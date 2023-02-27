# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import copy
import numpy as np
import tensorflow as tf

import evalu
import lrs
from data import Dataset
from models import model
from search import beam_search
from utils import parallel, cycle, util, queuer, saver, dtype
from modules import initializer


def tower_train_graph(train_features, optimizer, graph, params):
    # define multi-gpu training graph
    def _tower_train_graph(features):
        train_output = graph.train_fn(
            features, params, initializer=initializer.get_initializer(params.initializer, params.initializer_gain))

        tower_gradients = optimizer.compute_gradients(
            train_output["loss"] * tf.cast(params.loss_scale, tf.float32), colocate_gradients_with_ops=True)
        tower_gradients = [(g / tf.cast(params.loss_scale, tf.float32), v) for g, v in tower_gradients]

        return {
            "loss": train_output["loss"],
            "gradient": tower_gradients
        }

    # feed model to multiple gpus
    tower_outputs = parallel.parallel_model(
        _tower_train_graph, train_features,
        params.gpus, use_cpu=(len(params.gpus) == 0))

    loss = tf.add_n(tower_outputs['loss']) / len(tower_outputs['loss'])
    gradients = parallel.average_gradients(tower_outputs['gradient'])

    return loss, gradients


def tower_infer_graph(eval_features, graph, params):
    # define multi-gpu inferring graph
    def _tower_infer_graph(features):
        encoding_fn, decoding_fn = graph.infer_fn(params)
        beam_output = beam_search(features, encoding_fn, decoding_fn, params)

        return beam_output

    # feed model to multiple gpus
    eval_outputs = parallel.parallel_model(
        _tower_infer_graph, eval_features,
        params.gpus, use_cpu=(len(params.gpus) == 0))
    eval_seqs, eval_scores = eval_outputs['seq'], eval_outputs['score']

    return eval_seqs, eval_scores


def train(params):
    # status measure
    if params.recorder.estop or \
            params.recorder.epoch > params.epoches or \
            params.recorder.step > params.max_training_steps:
        tf.logging.info("Stop condition reached, you have finished training your model.")
        return 0.

    # loading dataset
    tf.logging.info("Begin Loading Training and Dev Dataset")
    start_time = time.time()
    train_dataset = Dataset(params, params.img_train_file, params.src_train_file, params.tgt_train_file,
                            params.src_vocab, params.tgt_vocab, params.max_len, params.max_img_len,
                            batch_or_token=params.batch_or_token,
                            data_leak_ratio=params.data_leak_ratio, target_size=params.target_size, f=params.f)
    dev_dataset = Dataset(params, params.img_dev_file, params.src_dev_file, params.src_dev_file,
                          params.src_vocab, params.src_vocab, params.eval_max_len, params.max_img_len,
                          batch_or_token='batch',
                          data_leak_ratio=params.data_leak_ratio, target_size=params.target_size, f=params.f)
    tf.logging.info(
        "End Loading dataset, within {} seconds".format(time.time() - start_time))

    # Build Graph
    with tf.Graph().as_default():
        lr = tf.placeholder(tf.as_dtype(dtype.floatx()), [], "learn_rate")

        # shift automatically sliced multi-gpu process into `zero` manner :)
        features = []
        for fidx in range(max(len(params.gpus), 1)):
            feature = {
                "source": tf.placeholder(tf.int32, [None, None], "source"),
                "target": tf.placeholder(tf.int32, [None, None], "target"),
                "image": tf.placeholder(tf.float32, [None, None, params.target_size], "image"),
                "mask": tf.placeholder(tf.float32, [None, None], "mask"),
                "is_img": tf.placeholder(tf.float32, [None], "is_img"),
                "label": tf.sparse_placeholder(tf.int32, name="label"),
            }
            features.append(feature)

        # session info
        sess = util.get_session(params.gpus)

        tf.logging.info("Begining Building Training Graph")
        start_time = time.time()

        # create global step
        global_step = tf.train.get_or_create_global_step()

        # set up optimizer
        optimizer = tf.train.AdamOptimizer(lr,
                                           beta1=params.beta1,
                                           beta2=params.beta2,
                                           epsilon=params.epsilon)

        # get graph
        graph = model.get_model(params.model_name)

        # set up training graph
        loss, gradients = tower_train_graph(features, optimizer, graph, params)

        # apply pseudo cyclic parallel operation
        vle, ops = cycle.create_train_op({"loss": loss}, gradients,
                                         optimizer, global_step, params)

        tf.logging.info("End Building Training Graph, within {} seconds".format(time.time() - start_time))

        tf.logging.info("Begin Building Inferring Graph")
        start_time = time.time()

        # set up infer graph
        eval_seqs, eval_scores = tower_infer_graph(features, graph, params)

        tf.logging.info("End Building Inferring Graph, within {} seconds".format(time.time() - start_time))

        # initialize the model
        sess.run(tf.global_variables_initializer())

        # log parameters
        util.variable_printer()

        # create saver
        train_saver = saver.Saver(
            checkpoints=params.checkpoints,
            output_dir=params.output_dir,
            best_checkpoints=params.best_checkpoints,
        )

        tf.logging.info("Training")
        cycle_counter = 0
        data_on_gpu = []
        cum_tokens = []

        # restore parameters
        tf.logging.info("Trying restore pretrained parameters")
        train_saver.restore(sess, path=params.pretrained_model)
        sess.run(tf.assign(global_step, 0))

        tf.logging.info("Trying restore existing parameters")
        # inception_utils.load_model(sess, params.pretrained_inception)
        train_saver.restore(sess)

        # setup learning rate
        params.lrate = params.recorder.lrate
        adapt_lr = lrs.get_lr(params)

        start_time = time.time()
        start_epoch = params.recorder.epoch
        for epoch in range(start_epoch, params.epoches + 1):

            params.recorder.epoch = epoch

            tf.logging.info("Training the model for epoch {}".format(epoch))
            size = params.batch_size if params.batch_or_token == 'batch' \
                else params.token_size

            train_queue = queuer.EnQueuer(
                train_dataset.batcher(size,
                                      buffer_size=params.buffer_size,
                                      shuffle=params.shuffle_batch,
                                      train=True),
                lambda x: x,
                worker_processes_num=params.process_num,
                input_queue_size=params.input_queue_size,
                output_queue_size=params.output_queue_size,
            )

            adapt_lr.before_epoch(eidx=epoch)

            for lidx, data in enumerate(train_queue):

                if params.train_continue:
                    if lidx <= params.recorder.lidx:
                        segments = params.recorder.lidx // 5
                        if params.recorder.lidx < 5 or lidx % segments == 0:
                            tf.logging.info(
                                "{} Passing {}-th index according to record".format(util.time_str(time.time()), lidx))

                        continue

                params.recorder.lidx = lidx

                data_on_gpu.append(data)
                # use multiple gpus, and data samples is not enough
                # make sure the data is fully added
                # The actual batch size: batch_size * num_gpus * update_cycle
                if len(params.gpus) > 0 and len(data_on_gpu) < len(params.gpus):
                    continue

                # increase the counter by 1
                cycle_counter += 1

                if cycle_counter == 1:
                    # calculate adaptive learning rate
                    adapt_lr.step(params.recorder.step)

                    # clear internal states
                    sess.run(ops["zero_op"])

                # data feeding to gpu placeholders
                feed_dicts = {}
                for fidx, shard_data in enumerate(data_on_gpu):
                    # define feed_dict
                    feed_dict = {
                        features[fidx]["source"]: shard_data["src"],
                        features[fidx]["target"]: shard_data["tgt"],
                        features[fidx]["image"]: shard_data["img"],
                        features[fidx]["mask"]: shard_data["mask"],
                        features[fidx]["is_img"]: shard_data["is_img"],
                        features[fidx]["label"]: shard_data["spar"],
                        lr: adapt_lr.get_lr(),
                    }
                    feed_dicts.update(feed_dict)

                    # collect target tokens
                    cum_tokens.append(np.sum(shard_data['tgt'] > 0))

                # reset data points on gpus
                data_on_gpu = []

                # internal accumulative gradient collection
                if cycle_counter < params.update_cycle:
                    sess.run(ops["collect_op"], feed_dict=feed_dicts)

                # at the final step, update model parameters
                if cycle_counter == params.update_cycle:
                    cycle_counter = 0

                    # directly update parameters, usually this works well
                    if not params.safe_nan:
                        _, loss, gnorm, pnorm, gstep = sess.run(
                            [ops["train_op"], vle["loss"], vle["gradient_norm"], vle["parameter_norm"],
                             global_step], feed_dict=feed_dicts)

                        if np.isnan(loss) or np.isinf(loss) or np.isnan(gnorm) or np.isinf(gnorm):
                            tf.logging.error("Nan or Inf raised! Loss {} GNorm {}.".format(loss, gnorm))
                            params.recorder.estop = True
                            break
                    else:
                        # Notice, applying safe nan can help train the big model, but sacrifice speed
                        loss, gnorm, pnorm, gstep = sess.run(
                            [vle["loss"], vle["gradient_norm"], vle["parameter_norm"], global_step],
                            feed_dict=feed_dicts)

                        if np.isnan(loss) or np.isinf(loss) or np.isnan(gnorm) or np.isinf(gnorm) \
                                or gnorm > params.gnorm_upper_bound:
                            tf.logging.error(
                                "Nan or Inf raised, GStep {} is passed! Loss {} GNorm {}.".format(gstep, loss, gnorm))
                            continue

                        sess.run(ops["train_op"], feed_dict=feed_dicts)

                    if gstep % params.disp_freq == 0:
                        end_time = time.time()
                        tf.logging.info(
                            "{} Epoch {}, GStep {}~{}, LStep {}~{}, "
                            "Loss {:.3f}, GNorm {:.3f}, PNorm {:.3f}, Lr {:.5f}, "
                            "Src {}, Tgt {}, Tokens {}, UD {:.3f} s".format(
                                util.time_str(end_time), epoch,
                                gstep - params.disp_freq + 1, gstep,
                                lidx - params.disp_freq + 1, lidx,
                                loss, gnorm, pnorm,
                                adapt_lr.get_lr(), data['src'].shape, data['tgt'].shape,
                                np.sum(cum_tokens), end_time - start_time)
                        )
                        start_time = time.time()
                        cum_tokens = []

                    # trigger model saver
                    if gstep > 0 and gstep % params.save_freq == 0:
                        train_saver.save(sess, gstep)
                        params.recorder.save_to_json(os.path.join(params.output_dir, "record.json"))

                    # trigger model evaluation
                    if gstep > 0 and gstep % params.eval_freq == 0:
                        if params.ema_decay > 0.:
                            sess.run(ops['ema_backup_op'])
                            sess.run(ops['ema_assign_op'])

                        tf.logging.info("Start Evaluating")
                        eval_start_time = time.time()
                        tranes, scores, indices = evalu.decoding(
                            sess, features, eval_seqs,
                            eval_scores, dev_dataset, params)
                        bleu = evalu.eval_metric(tranes, params.tgt_dev_file, indices=indices, remove_bpe=params.remove_bpe)
                        eval_end_time = time.time()
                        tf.logging.info("End Evaluating")

                        if params.ema_decay > 0.:
                            sess.run(ops['ema_restore_op'])

                        tf.logging.info(
                            "{} GStep {}, Scores {}, BLEU {}, Duration {:.3f} s".format(
                                util.time_str(eval_end_time), gstep, np.mean(scores),
                                bleu, eval_end_time - eval_start_time)
                        )

                        # save eval translation
                        evalu.dump_tanslation(
                            tranes,
                            os.path.join(params.output_dir, "eval-{}.trans.txt".format(gstep)),
                            indices=indices)

                        # save parameters
                        train_saver.save(sess, gstep, bleu)

                        # check for early stopping
                        valid_scores = [v[1] for v in params.recorder.valid_script_scores]
                        if len(valid_scores) == 0 or bleu > np.max(valid_scores):
                            params.recorder.bad_counter = 0
                        else:
                            params.recorder.bad_counter += 1

                            if params.recorder.bad_counter > params.estop_patience:
                                params.recorder.estop = True
                                break

                        params.recorder.history_scores.append((int(gstep), float(np.mean(scores))))
                        params.recorder.valid_script_scores.append((int(gstep), float(bleu)))
                        params.recorder.save_to_json(os.path.join(params.output_dir, "record.json"))

                        # handle the learning rate decay in a typical manner
                        adapt_lr.after_eval(float(bleu))

                    # trigger temporary sampling
                    if gstep > 0 and gstep % params.sample_freq == 0:
                        tf.logging.info("Start Sampling")
                        decode_seqs, decode_scores = sess.run(
                            [eval_seqs[:1], eval_scores[:1]], feed_dict={features[0]["image"]: data["img"][:5], features[0]["mask"]: data["mask"][:5]})
                        tranes, scores = evalu.decode_hypothesis(decode_seqs, decode_scores, params)

                        for sidx in range(min(5, len(scores))):
                            # sample_source = evalu.decode_target_token(data['src'][sidx], params.src_vocab)
                            # tf.logging.info("{}-th Source: {}".format(sidx, ' '.join(sample_source)))
                            sample_target = evalu.decode_target_token(data['tgt'][sidx], params.tgt_vocab)
                            tf.logging.info("{}-th Target: {}".format(sidx, ' '.join(sample_target)))
                            sample_trans = tranes[sidx]
                            tf.logging.info("{}-th Translation: {}".format(sidx, ' '.join(sample_trans)))

                        tf.logging.info("End Sampling")

                    # trigger stopping
                    if gstep >= params.max_training_steps:
                        # stop running by setting EStop signal
                        params.recorder.estop = True
                        break

                    # should be equal to global_step
                    params.recorder.step = int(gstep)

            if params.recorder.estop:
                tf.logging.info("Early Stopped!")
                break

            # reset to 0
            params.recorder.lidx = -1

            adapt_lr.after_epoch(eidx=epoch)

    # Final Evaluation
    tf.logging.info("Start Final Evaluating")
    if params.ema_decay > 0.:
        sess.run(ops['ema_backup_op'])
        sess.run(ops['ema_assign_op'])

    gstep = int(params.recorder.step + 1)
    eval_start_time = time.time()
    tranes, scores, indices = evalu.decoding(sess, features, eval_seqs, eval_scores, dev_dataset, params)
    bleu = evalu.eval_metric(tranes, params.tgt_dev_file, indices=indices, remove_bpe=params.remove_bpe)
    eval_end_time = time.time()
    tf.logging.info("End Evaluating")

    if params.ema_decay > 0.:
        sess.run(ops['ema_restore_op'])

    tf.logging.info(
        "{} GStep {}, Scores {}, BLEU {}, Duration {:.3f} s".format(
            util.time_str(eval_end_time), gstep, np.mean(scores), bleu, eval_end_time - eval_start_time)
    )

    # save eval translation
    evalu.dump_tanslation(
        tranes,
        os.path.join(params.output_dir, "eval-{}.trans.txt".format(gstep)),
        indices=indices)

    tf.logging.info("Your training is finished :)")

    return train_saver.best_score


def evaluate(params):
    # loading dataset
    tf.logging.info("Begin Loading Test Dataset")
    start_time = time.time()
    test_dataset = Dataset(params, params.img_test_file, params.src_test_file, params.src_test_file,
                           params.src_vocab, params.src_vocab, params.eval_max_len, params.max_img_len,
                           batch_or_token='batch',
                           data_leak_ratio=params.data_leak_ratio, target_size=params.target_size, f=params.f)
    tf.logging.info(
        "End Loading dataset, within {} seconds".format(time.time() - start_time))

    # Build Graph
    with tf.Graph().as_default():
        features = []
        for fidx in range(max(len(params.gpus), 1)):
            feature = {
                "source": tf.placeholder(tf.int32, [None, None], "source"),
                "image": tf.placeholder(tf.float32, [None, None, params.target_size], "image"),
                "mask": tf.placeholder(tf.float32, [None, None], "mask"),

            }
            features.append(feature)

        # session info
        sess = util.get_session(params.gpus)

        tf.logging.info("Begining Building Evaluation Graph")
        start_time = time.time()

        # get graph
        graph = model.get_model(params.model_name)

        # set up infer graph
        eval_seqs, eval_scores = tower_infer_graph(features, graph, params)

        tf.logging.info("End Building Inferring Graph, within {} seconds".format(time.time() - start_time))

        # set up ema
        if params.ema_decay > 0.:
            # recover from EMA
            ema = tf.train.ExponentialMovingAverage(decay=params.ema_decay)
            ema.apply(tf.trainable_variables())
            ema_assign_op = tf.group(*(tf.assign(var, ema.average(var).read_value())
                                       for var in tf.trainable_variables()))
        else:
            ema_assign_op = tf.no_op()

        # initialize the model
        sess.run(tf.global_variables_initializer())

        # log parameters
        util.variable_printer()

        # create saver
        eval_saver = saver.Saver(checkpoints=params.checkpoints, output_dir=params.output_dir)

        # restore parameters
        tf.logging.info("Trying restore existing parameters")
        eval_saver.restore(sess, params.output_dir)
        sess.run(ema_assign_op)

        tf.logging.info("Starting Evaluating")
        eval_start_time = time.time()
        tranes, scores, indices = evalu.decoding(sess, features, eval_seqs, eval_scores, test_dataset, params)
        bleu = evalu.eval_metric(tranes, params.tgt_test_file, indices=indices, remove_bpe=params.remove_bpe)
        eval_end_time = time.time()

        tf.logging.info(
            "{} Scores {}, BLEU {}, Duration {}s".format(
                util.time_str(eval_end_time), np.mean(scores), bleu, eval_end_time - eval_start_time)
        )

        # save translation
        evalu.dump_tanslation(tranes, params.test_output, indices=indices)

    return bleu
