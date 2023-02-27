# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import tensorflow as tf

import func
from models import model
from utils import util, dtype


def encoder(source, mask, params, is_training=False, mt=False, is_gloss=False):
    # - mt: if true, source is word ids and we need an embedding layer to extract source input
    #       if false, source is sign video features
    # - is_gloss: if true, translation into glosses
    #             if false, translation into text, we append an indicator vector to guide the model where to generate

    # remapping
    hidden_size = params.hidden_size

    if not mt:
        # features = tf.reshape(features, [sshp[0], sshp[1], util.shape_list(features)[-1]])
        features = func.linear(source, hidden_size, scope="premapper")

    else:
        mask = dtype.tf_to_float(tf.cast(source, tf.bool))
        initializer = tf.random_normal_initializer(0.0, hidden_size ** -0.5)

        embed_name = "embedding" if params.shared_source_target_embedding \
            else "src_embedding"
        src_emb = tf.get_variable(embed_name,
                              [params.src_vocab.size(), params.embed_size],
                              initializer=initializer)
        src_bias = tf.get_variable("bias", [params.embed_size])

        inputs = tf.gather(src_emb, source) * (hidden_size ** 0.5)
        features = tf.nn.bias_add(inputs, src_bias)

    # handle source or target generation
    gloss_indicator = tf.get_variable("gloss", [1, params.embed_size])
    trans_indicator = tf.get_variable("trans", [1, params.embed_size])
    indicator = gloss_indicator if is_gloss else trans_indicator

    mask = tf.pad(mask, [[0, 0], [1, 0]], constant_values=1)
    ishp = util.shape_list(features)
    features = tf.concat([util.expand_tile_dims(indicator, ishp[0], axis=0), features], 1)

    inputs = func.add_timing_signal(features)
    inputs = func.layer_norm(inputs)

    inputs = util.valid_apply_dropout(inputs, params.dropout)

    with tf.variable_scope("encoder"):
        x = inputs
        for layer in range(params.num_encoder_layer):
            if params.deep_transformer_init:
                layer_initializer = tf.variance_scaling_initializer(
                    params.initializer_gain * (layer + 1) ** -0.5,
                    mode="fan_avg",
                    distribution="uniform")
            else:
                layer_initializer = None
            # modality-specific layers: when layer <= params.sep_layer, we apply different transformer encoder layers to sign videos and texts
            with tf.variable_scope("layer_{}".format(layer) if layer > params.sep_layer else "layer_{}_{}".format(layer, 'mt' if mt else 'st'), initializer=layer_initializer):
                with tf.variable_scope("self_attention"):
                    y = func.dot_attention(
                        x,
                        None,
                        func.attention_bias(mask, "masking"),
                        hidden_size,
                        num_heads=params.num_heads,
                        dropout=params.attention_dropout,
                    )

                    y = y['output']
                    x = func.residual_fn(x, y, dropout=params.residual_dropout)
                    x = func.layer_norm(x)

                with tf.variable_scope("feed_forward"):
                    y = func.ffn_layer(
                        x,
                        params.filter_size,
                        hidden_size,
                        dropout=params.relu_dropout,
                    )

                    x = func.residual_fn(x, y, dropout=params.residual_dropout)
                    x = func.layer_norm(x)

    source_encodes = x
    x_shp = util.shape_list(x)

    return {
        "encodes": source_encodes,
        "decoder_initializer": {
            "layer_{}".format(l): {
                "k": dtype.tf_to_float(tf.zeros([x_shp[0], 0, hidden_size])),
                "v": dtype.tf_to_float(tf.zeros([x_shp[0], 0, hidden_size])),
            }
            for l in range(params.num_decoder_layer)
        },
        "mask": mask
    }


def decoder(target, state, params, labels=None, is_img=None):
    mask = dtype.tf_to_float(tf.cast(target, tf.bool))
    hidden_size = params.hidden_size
    initializer = tf.random_normal_initializer(0.0, hidden_size ** -0.5)

    is_training = ('decoder' not in state)

    if is_training:
        target, mask = util.remove_invalid_seq(target, mask)

    embed_name = "embedding" if params.shared_source_target_embedding \
        else "tgt_embedding"
    tgt_emb = tf.get_variable(embed_name,
                              [params.tgt_vocab.size(), params.embed_size],
                              initializer=initializer)
    tgt_bias = tf.get_variable("bias", [params.embed_size])

    inputs = tf.gather(tgt_emb, target) * (hidden_size ** 0.5)
    inputs = tf.nn.bias_add(inputs, tgt_bias)

    # shift
    if is_training:
        inputs = tf.pad(inputs, [[0, 0], [1, 0], [0, 0]])
        inputs = inputs[:, :-1, :]
        inputs = func.add_timing_signal(inputs)
    else:
        inputs = tf.cond(tf.reduce_all(tf.equal(target, params.tgt_vocab.pad())),
                         lambda: tf.zeros_like(inputs),
                         lambda: inputs)
        mask = tf.ones_like(mask)
        inputs = func.add_timing_signal(inputs, time=dtype.tf_to_float(state['time']))

    inputs = util.valid_apply_dropout(inputs, params.dropout)

    with tf.variable_scope("decoder"):
        x = inputs
        for layer in range(params.num_decoder_layer):
            if params.deep_transformer_init:
                layer_initializer = tf.variance_scaling_initializer(
                    params.initializer_gain * (layer + 1) ** -0.5,
                    mode="fan_avg",
                    distribution="uniform")
            else:
                layer_initializer = None
            with tf.variable_scope("layer_{}".format(layer), initializer=layer_initializer):
                with tf.variable_scope("self_attention"):
                    y = func.dot_attention(
                        x,
                        None,
                        func.attention_bias(tf.shape(mask)[1], "causal"),
                        hidden_size,
                        num_heads=params.num_heads,
                        dropout=params.attention_dropout,
                        cache=None if is_training else
                        state['decoder']['state']['layer_{}'.format(layer)],
                    )
                    if not is_training:
                        # k, v
                        state['decoder']['state']['layer_{}'.format(layer)] \
                            .update(y['cache'])

                    y = y['output']
                    x = func.residual_fn(x, y, dropout=params.residual_dropout)
                    x = func.layer_norm(x)

                with tf.variable_scope("cross_attention"):
                    y = func.dot_attention(
                        x,
                        state['encodes'],
                        func.attention_bias(state['mask'], "masking"),
                        hidden_size,
                        num_heads=params.num_heads,
                        dropout=params.attention_dropout,
                        cache=None if is_training else
                        state['decoder']['state']['layer_{}'.format(layer)],
                    )
                    if not is_training:
                        # mk, mv
                        state['decoder']['state']['layer_{}'.format(layer)] \
                            .update(y['cache'])

                    y = y['output']
                    x = func.residual_fn(x, y, dropout=params.residual_dropout)
                    x = func.layer_norm(x)

                with tf.variable_scope("feed_forward"):
                    y = func.ffn_layer(
                        x,
                        params.filter_size,
                        hidden_size,
                        dropout=params.relu_dropout,
                    )

                    x = func.residual_fn(x, y, dropout=params.residual_dropout)
                    x = func.layer_norm(x)
    feature = x
    if 'dev_decode' in state:
        feature = x[:, -1, :]

    embed_name = "tgt_embedding" if params.shared_target_softmax_embedding \
        else "softmax_embedding"
    embed_name = "embedding" if params.shared_source_target_embedding \
        else embed_name
    softmax_emb = tf.get_variable(embed_name,
                                  [params.tgt_vocab.size(), params.embed_size],
                                  initializer=initializer)
    feature = tf.reshape(feature, [-1, params.embed_size])
    logits = tf.matmul(feature, softmax_emb, False, True)

    logits = tf.cast(logits, tf.float32)

    soft_label, normalizer = util.label_smooth(
        target,
        util.shape_list(logits)[-1],
        factor=params.label_smooth)
    centropy = tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=logits,
        labels=soft_label
    )
    centropy -= normalizer
    centropy = tf.reshape(centropy, tf.shape(target))

    mask = tf.cast(mask, tf.float32)
    per_sample_loss = tf.reduce_sum(centropy * mask, -1) / tf.reduce_sum(mask, -1)

    # for sign-related tasks, we need is_img to distinguish which examples are sign examples
    if is_img is None:
        loss = tf.reduce_mean(per_sample_loss)
    else:
        loss = tf.reduce_sum(per_sample_loss * is_img) / (tf.reduce_sum(is_img) + 1e-8)

    # computing CTC regularization term
    # note we only retrain sign2text's CTC regularizer
    if is_training and params.ctc_enable and labels is not None:
        assert labels is not None

        # batch x seq x dim
        encoding = state['encodes']
        enc_logits = func.linear(encoding, params.src_vocab.size() + 1, scope="ctc_mapper")
        # seq dimension transpose
        enc_logits = tf.transpose(enc_logits, (1, 0, 2))

        enc_logits = tf.to_float(enc_logits)

        with tf.name_scope('loss'):
            ctc_loss = tf.nn.ctc_loss(labels, enc_logits, tf.cast(tf.reduce_sum(state['mask'], -1), tf.int32),
                                      ignore_longer_outputs_than_inputs=True,preprocess_collapse_repeated=params.ctc_repeated)
            ctc_loss /= tf.reduce_sum(mask, -1)

            if is_img is None:
                ctc_loss = tf.reduce_mean(ctc_loss)
            else:
                ctc_loss = tf.reduce_sum(ctc_loss * is_img) / (tf.reduce_sum(is_img) + 1e-8)

        loss = params.ctc_alpha * ctc_loss + loss

    return loss, logits, state, per_sample_loss


def train_fn(features, params, initializer=None):
    with tf.variable_scope(params.scope_name or "model",
                           initializer=initializer,
                           reuse=tf.AUTO_REUSE,
                           dtype=tf.as_dtype(dtype.floatx()),
                           custom_getter=dtype.float32_variable_storage_getter):
        # features contains five conents
        #  - image:  [batch, sign_video_len, feature_dim] (float) extracted sign video features based on SMKD
        #  - mask :  [batch, sign_video_len]              (float) mask for sign video features
        #  - source: [batch, src_seq_len] (int, ids) gloss or MT source inputs
        #  - target: [batch, tgt_seq_len] (int, ids) gloss translation or MT target
        #  - is_img: [batch] (float, like mask, 0.0 or 1.0) indicate whether the example comes from SLT
        #            for example, SLT example contains sign videos; but MT doesn't

        # for SLT examples, the training data is a triple (sign video, gloss, text)
        # for MT  examples, the training data is also a triple (dummy video, source, target)

        # sign translation
        state = encoder(features['image'], features['mask'], params, is_training=True, mt=False, is_gloss=False)
        loss_trans, _, _, _ = decoder(features['target'],  state, params,
                                         labels=features['label'] if params.ctc_enable else None, is_img=features["is_img"])

        # sign recognition
        state = encoder(features['image'], features['mask'], params, is_training=True, mt=False, is_gloss=True)
        loss_gloss, _, _, _ = decoder(features['source'],  state, params, labels=None, is_img=features["is_img"])

        # gloss2text translation & machine translation: both are text-to-text tasks
        state = encoder(features['source'], None, params, is_training=True, mt=True, is_gloss=False)
        loss_g2t, _, _, _ = decoder(features['target'],  state, params, labels=None, is_img=None)

        # sum-up all loss terms
        loss = loss_trans + loss_gloss + loss_g2t
 

        return {
            "loss": loss
        }


def score_fn(features, params, initializer=None):
    params = copy.copy(params)
    params = util.closing_dropout(params)
    params.label_smooth = 0.0
    with tf.variable_scope(params.scope_name or "model",
                           initializer=initializer,
                           reuse=tf.AUTO_REUSE,
                           dtype=tf.as_dtype(dtype.floatx()),
                           custom_getter=dtype.float32_variable_storage_getter):
        state = encoder(features['image'], features['mask'], params, is_training=False, mt=False, is_gloss=False)
        _, _, _, scores = decoder(features['target'], state, params)

        return {
            "score": scores
        }


def infer_fn(params):
    params = copy.copy(params)
    params = util.closing_dropout(params)

    def encoding_fn(image, mask):
        with tf.variable_scope(params.scope_name or "model",
                               reuse=tf.AUTO_REUSE,
                               dtype=tf.as_dtype(dtype.floatx()),
                               custom_getter=dtype.float32_variable_storage_getter):
            state = encoder(image, mask, params, is_training=False, mt=False, is_gloss=False)
            state["decoder"] = {
                "state": state["decoder_initializer"]
            }
            return state

    def decoding_fn(target, state, time):
        with tf.variable_scope(params.scope_name or "model",
                               reuse=tf.AUTO_REUSE,
                               dtype=tf.as_dtype(dtype.floatx()),
                               custom_getter=dtype.float32_variable_storage_getter):
            state['time'] = time
            step_loss, step_logits, step_state, _ = decoder(
                target, state, params)
            del state['time']

            return step_logits, step_state

    return encoding_fn, decoding_fn


# register the model, with a unique name
model.model_register("transformer", train_fn, score_fn, infer_fn)
