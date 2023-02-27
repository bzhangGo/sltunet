# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import os
import codecs
import random
import socket
import json

import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc

import models
import main as graph
from utils.apply_bpe import BPE
from vocab import Vocab
from utils import dtype, util

logger = tf.get_logger()
logger.propagate = False


# define global initial parameters
global_params = tc.training.HParams(
    # whether share source and target word embedding
    shared_source_target_embedding=False,
    # whether share target and softmax word embedding
    shared_target_softmax_embedding=True,

    # separately encoding textual and sign video until `sep_layer`
    sep_layer=0,
    # source/target BPE codes and dropout rate => used for BPE-dropout
    src_codes='',
    tgt_codes='',
    src_bpe_dropout=0.,
    tgt_bpe_dropout=0.,
    bpe_dropout_stochastic_rate=0.6,

    # decoding maximum length: source length + decode_length
    decode_length=50,
    # beam size
    beam_size=4,
    # length penalty during beam search
    decode_alpha=0.6,
    # noise beam search with gumbel
    enable_noise_beam_search=False,
    # beam search temperature, sharp or flat prediction
    beam_search_temperature=1.0,
    # return top elements, not used
    top_beams=1,
    # remove BPE symbols at evaluation
    remove_bpe=False,

    # ctc setting for tf's ctc loss, handeling invalid paths
    ctc_repeated=False,
    # whether add ctc-loss during training
    ctc_enable=False,
    # ctc loss factor, corresponding to \alpha in Eq. (3)
    ctc_alpha=0.3,

    # learning rate setup
    # warmup steps: start point for learning rate stop increasing
    warmup_steps=400,
    # initial learning rate
    lrate=1e-5,
    # minimum learning rate
    min_lrate=0.0,
    # maximum learning rate
    max_lrate=1.0,

    # initialization
    # type of initializer
    initializer="uniform",
    # initializer range control
    initializer_gain=0.08,

    # parameters for transformer
    # encoder and decoder hidden size
    hidden_size=512,
    # source and target embedding size
    embed_size=512,
    # sign video feature size
    img_feature_size=2048,
    # sign video duplicate size
    img_aug_size=11,
    # ffn filter size for transformer
    filter_size=2048,
    # dropout value
    dropout=0.1,
    relu_dropout=0.1,
    residual_dropout=0.1,
    # scope name
    scope_name="transformer",
    # attention dropout
    attention_dropout=0.1,
    # the number of encoder layers, valid for deep nmt
    num_encoder_layer=6,
    # the number of decoder layers, valid for deep nmt
    num_decoder_layer=6,
    # the number of attention heads
    num_heads=8,

    # allowed maximum sentence length
    max_len=100,
    max_img_len=512,
    eval_max_len=1000,
    # constant batch size at 'batch' mode for batch-based batching
    batch_size=80,
    # constant token size at 'token' mode for token-based batching
    token_size=3000,
    # token or batch-based data iterator
    batch_or_token='token',
    # batch size for decoding, i.e. number of source sentences decoded at the same time
    eval_batch_size=32,
    # whether shuffle batches during training
    shuffle_batch=True,

    # whether use multiprocessing deal with data reading, default true
    process_num=1,
    # buffer size controls the number of sentences read in one time,
    buffer_size=100,
    # a unique queue in multi-thread reading process
    input_queue_size=100,
    output_queue_size=100,
    # data leak buffer threshold
    data_leak_ratio=0.5,

    # source vocabulary
    src_vocab_file="",
    # target vocabulary
    tgt_vocab_file="",
    # source train file
    src_train_file="",
    # target train file
    tgt_train_file="",
    # sign video train file
    img_train_file="",
    # source development file
    src_dev_file="",
    # target development file
    tgt_dev_file="",
    # sign video dev file
    img_dev_file="",
    # source test file
    src_test_file="",
    # target test file
    tgt_test_file="",
    # sign video test file
    img_test_file="",
    # working directory
    output_dir="",
    # test output file
    test_output="",
    # pretrained modeling
    pretrained_model="",

    # adam optimizer hyperparameters
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-9,
    # gradient clipping value
    clip_grad_norm=5.0,
    # gradient norm upper bound, avoid wired large gnorm, only works under safe-nan mode
    gnorm_upper_bound=1e20,
    # early stopping
    estop_patience=100,
    # label smoothing value
    label_smooth=0.1,
    # maximum epochs
    epoches=10,
    # the effective batch size is: batch/token size * update_cycle * num_gpus
    # sequential update cycle
    update_cycle=1,
    # the available gpus
    gpus=[0],
    # enable safely handle nan (only helpful for some wired large/nan norms)
    safe_nan=False,

    # enable training deep transformer
    deep_transformer_init=False,
    # which task to evaluate, supporting sign2text, sign2gloss, gloss2text
    eval_task="sign2text",

    # print information every disp_freq training steps
    disp_freq=100,
    # evaluate on the development file every eval_freq steps
    eval_freq=10000,
    # save the model parameters every save_freq steps
    save_freq=5000,
    # print sample translations every sample_freq steps
    sample_freq=1000,
    # saved checkpoint number
    checkpoints=5,
    best_checkpoints=1,
    # the maximum training steps, program with stop if epochs or max_training_steps is meet
    max_training_steps=1000,

    # number of threads for threaded reading, seems useless
    nthreads=6,
    # random control, not so well for tensorflow.
    random_seed=1234,
    # whether or not train from checkpoint
    train_continue=True,

    # support for float32/float16
    default_dtype="float32",
    dtype_epsilon=1e-8,
    dtype_inf=1e8,
    loss_scale=1.0,
)

flags = tf.flags
flags.DEFINE_string("config", "", "Additional Mergable Parameters")
flags.DEFINE_string("parameters", "", "Command Line Refinable Parameters")
flags.DEFINE_string("name", "model", "Description of the training process for distinguishing")
flags.DEFINE_string("mode", "train", "train or test or ensemble")


# saving model configuration
def save_parameters(params, output_dir):
    if not tf.gfile.Exists(output_dir):
        tf.gfile.MkDir(output_dir)

    param_name = os.path.join(output_dir, "param.json")
    with tf.gfile.Open(param_name, "w") as writer:
        tf.logging.info("Saving parameters into {}"
                        .format(param_name))
        writer.write(params.to_json())


# load model configuration
def load_parameters(params, output_dir):
    param_name = os.path.join(output_dir, "param.json")
    param_name = os.path.abspath(param_name)

    if tf.gfile.Exists(param_name):
        tf.logging.info("Loading parameters from {}"
                        .format(param_name))
        with tf.gfile.Open(param_name, 'r') as reader:
            json_str = reader.readline()
            params.parse_json(json_str)
    return params


class Recorder(object):
    def load_from_json(self, file_name):
        tf.logging.info("Loading recoder file from {}".format(file_name))
        with open(file_name, 'r', encoding='utf-8') as fh:
            self.__dict__.update(json.load(fh))

    def save_to_json(self, file_name):
        tf.logging.info("Saving recorder file into {}".format(file_name))
        with open(file_name, 'w', encoding='utf-8') as fh:
            json.dump(self.__dict__, fh, indent=2)


# build training process recorder
def setup_recorder(params):
    recorder = Recorder()
    # for early stopping
    recorder.bad_counter = 0    # start from 0
    recorder.estop = False

    recorder.lidx = -1      # local data index
    recorder.step = 0       # global step, start from 0
    recorder.epoch = 1      # epoch number, start from 1
    recorder.lrate = params.lrate     # running learning rate
    recorder.history_scores = []
    recorder.valid_script_scores = []

    # trying to load saved recorder
    record_path = os.path.join(params.output_dir, "record.json")
    record_path = os.path.abspath(record_path)
    if tf.gfile.Exists(record_path):
        recorder.load_from_json(record_path)

    params.add_hparam('recorder', recorder)
    return params


# print model configuration
def print_parameters(params):
    tf.logging.info("The Used Configuration:")
    for k, v in params.values().items():
        tf.logging.info("%s\t%s", k.ljust(20), str(v).ljust(20))
    tf.logging.info("")


def main(_):
    # set up logger
    tf.logging.set_verbosity(tf.logging.INFO)

    tf.logging.info("Welcome Using Zero :)")

    pid = os.getpid()
    tf.logging.info("Your pid is {0} and use the following command to force kill your running:\n"
                    "'pkill -9 -P {0}; kill -9 {0}'".format(pid))
    # On clusters, this could tell which machine you are running
    tf.logging.info("Your running machine name is {}".format(socket.gethostname()))

    # load registered models
    util.dynamic_load_module(models, prefix="models")

    params = global_params

    # try loading parameters
    # priority: command line > saver > default
    params.parse(flags.FLAGS.parameters)
    if os.path.exists(flags.FLAGS.config):
        params.override_from_dict(eval(open(flags.FLAGS.config).read()))
    params = load_parameters(params, params.output_dir)
    # override
    if os.path.exists(flags.FLAGS.config):
        params.override_from_dict(eval(open(flags.FLAGS.config).read()))
    params.parse(flags.FLAGS.parameters)

    # set up random seed
    random.seed(params.random_seed)
    np.random.seed(params.random_seed)
    tf.set_random_seed(params.random_seed)

    # loading vocabulary
    tf.logging.info("Begin Loading Vocabulary")
    start_time = time.time()
    params.src_vocab = Vocab(params.src_vocab_file)
    params.tgt_vocab = Vocab(params.tgt_vocab_file)
    params.src_bpe = BPE(codecs.open(params.src_codes, encoding='utf-8'), -1, '@@', None, None)
    params.tgt_bpe = BPE(codecs.open(params.tgt_codes, encoding='utf-8'), -1, '@@', None, None)
    tf.logging.info("End Loading Vocabulary, Source Vocab Size {}, "
                    "Target Vocab Size {}, within {} seconds"
                    .format(params.src_vocab.size(), params.tgt_vocab.size(), time.time() - start_time))

    # print parameters
    print_parameters(params)

    # set up the default datatype
    dtype.set_floatx(params.default_dtype)
    dtype.set_epsilon(params.dtype_epsilon)
    dtype.set_inf(params.dtype_inf)

    mode = flags.FLAGS.mode
    if mode == "train":
        # save parameters
        save_parameters(params, params.output_dir)
        # load the recorder
        params = setup_recorder(params)

        graph.train(params)
    elif mode == "test":
        graph.evaluate(params)
    else:
        tf.logging.error("Invalid mode: {}".format(mode))


if __name__ == '__main__':
    tf.app.run()
