import codecs
import collections
import time

import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import tensorflow.contrib.seq2seq as seq2seq
import tensorflow.contrib.training as tf_training
from tensorflow.contrib.learn import ModeKeys

from utils.iterator import TrainIterator, EvalIterator, InferIterator
from utils.param_utils import get_model_params
from utils.vocabulary import Vocabulary, load_vocab

# If a vocab size is greater than this value, put the embedding on cpu instead
VOCAB_SIZE_THRESHOLD_CPU = 50000


def _get_embed_device(vocab_size):
    """Decide on which device to place an embed matrix given its vocab size."""
    if vocab_size > VOCAB_SIZE_THRESHOLD_CPU:
        return "/cpu:0"
    else:
        return "/gpu:0"


def get_initializer(init_op, seed=None, init_weight=None):
    """Create an initializer. init_weight is only for uniform."""
    if init_op == "uniform":
        assert init_weight
        return tf.random_uniform_initializer(
            -init_weight, init_weight, seed=seed)
    elif init_op == "glorot_normal":
        return tf.keras.initializers.glorot_normal(
            seed=seed)
    elif init_op == "glorot_uniform":
        return tf.keras.initializers.glorot_uniform(
            seed=seed)
    else:
        raise ValueError("Unknown init_op %s" % init_op)


def load_embed_txt(embed_file):
    """Load embed_file into a python dictionary.

    Note: the embed_file should be a Glove formated txt file. Assuming
    embed_size=5, for example:

    the -0.071549 0.093459 0.023738 -0.090339 0.056123
    to 0.57346 0.5417 -0.23477 -0.3624 0.4037
    and 0.20327 0.47348 0.050877 0.002103 0.060547

    Args:
      embed_file: file path to the embedding file.
    Returns:
      a dictionary that maps word to vector, and the size of embedding dimensions.
    """
    emb_dict = dict()
    emb_size = None
    with codecs.getreader("utf-8")(tf.gfile.GFile(embed_file, 'rb')) as f:
        for line in f:
            tokens = line.strip().split(" ")
            word = tokens[0]
            vec = list(map(float, tokens[1:]))
            emb_dict[word] = vec
            if emb_size:
                assert emb_size == len(vec), "All embedding size should be same."
            else:
                emb_size = len(vec)
    return emb_dict, emb_size


def _create_pretrained_emb_from_txt(
        vocab_file,
        embed_file,
        num_trainable_tokens=3,
        dtype=tf.float32,
        scope=None):
    """Load pretrain embeding from embed_file, and return an embedding matrix.

    Args:
      embed_file: Path to a Glove formated embedding txt file.
      num_trainable_tokens: Make the first n tokens in the vocab file as trainable
        variables. Default is 3, which is "<unk>", "<s>" and "</s>".
    """
    vocab = load_vocab(vocab_file)
    trainable_tokens = vocab[:num_trainable_tokens]

    print("# Using pretrained embedding: %s." % embed_file)
    print("  with trainable tokens: ")

    emb_dict, emb_size = load_embed_txt(embed_file)
    for token in trainable_tokens:
        print("    %s" % token)
        if token not in emb_dict:
            emb_dict[token] = [0.0] * emb_size  # trainable token with embedding [0,0,..,0]

    emb_mat = np.array(
        [emb_dict[token] for token in vocab], dtype=dtype.as_numpy_dtype())
    emb_mat = tf.constant(emb_mat)
    emb_mat_const = tf.slice(emb_mat, [num_trainable_tokens, 0], [-1, -1])
    with tf.variable_scope(scope or "pretrained_embeddings", dtype=dtype):
        emb_mat_var = tf.get_variable(
            "emb_mat_var", [num_trainable_tokens, emb_size])
    return tf.concat([emb_mat_var, emb_mat_const], 0)


def create_or_load_embedding(embed_name, vocab_file, embed_file,
                             vocab_size, embed_size, dtype=tf.float32):
    """Create a new or load an existing embedding matrix."""
    if vocab_file and embed_file:
        embedding = _create_pretrained_emb_from_txt(vocab_file, embed_file)
    else:
        with tf.device(_get_embed_device(vocab_size)):
            embedding = tf.get_variable(
                embed_name, [vocab_size, embed_size], dtype)
    return embedding


def load_model(model, ckpt, session, name):
    start_time = time.time()
    model.saver.restore(session, ckpt)
    session.run(tf.tables_initializer())
    print("  loaded %s model parameters from %s, time %.2fs" %
          (name, ckpt, time.time() - start_time))
    return model


def create_or_load_model(model, model_dir, session, name):
    """Create translation model and initialize or load parameters in session."""
    latest_ckpt = tf.train.latest_checkpoint(model_dir)
    if latest_ckpt:
        model = load_model(model, latest_ckpt, session, name)
    else:
        start_time = time.time()
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())
        print("# Created %s model with fresh parameters, time %.2fs" %
              (name, time.time() - start_time))

    global_step = model.global_step.eval(session=session)
    return model, global_step


def gradient_clip(gradients, max_gradient_norm):
    """Clipping gradients of a model."""
    clipped_gradients, gradient_norm = tf.clip_by_global_norm(
        gradients, max_gradient_norm)
    gradient_norm_summary = [tf.summary.scalar("grad_norm", gradient_norm),
                             tf.summary.scalar("clipped_gradient", tf.global_norm(clipped_gradients))]

    return clipped_gradients, gradient_norm_summary, gradient_norm


def get_learning_rate_warmup(global_step,
                             learning_rate,
                             warmup_steps,
                             warmup_scheme="t2t"):
    """Get learning rate warmup."""
    print("  learning_rate=%g, warmup_steps=%d, warmup_scheme=%s" %
          (learning_rate, warmup_steps, warmup_scheme))

    # Apply inverse decay if global steps less than warmup steps.
    # Inspired by https://arxiv.org/pdf/1706.03762.pdf (Section 5.3)
    # When step < warmup_steps,
    #   learing_rate *= warmup_factor ** (warmup_steps - step)
    if warmup_scheme == "t2t":
        # 0.01^(1/warmup_steps): we start with a lr, 100 times smaller
        warmup_factor = tf.exp(tf.log(0.01) / warmup_steps)
        inv_decay = warmup_factor ** (
            tf.to_float(warmup_steps - global_step))
    else:
        raise ValueError("Unknown warmup scheme %s" % warmup_scheme)

    return tf.cond(
        global_step < warmup_steps,
        lambda: inv_decay * learning_rate,
        lambda: learning_rate,
        name="learning_rate_warump_cond")


def get_learning_rate_decay(num_train_steps,
                            global_step,
                            learning_rate,
                            decay_scheme=""):
    """Get learning rate decay."""
    start_decay_step = 0
    decay_steps = 0
    decay_factor = 0
    if decay_scheme == "luong10":
        start_decay_step = int(num_train_steps / 2)
        remain_steps = num_train_steps - start_decay_step
        decay_steps = int(remain_steps / 10)  # decay 10 times
        decay_factor = 0.5
    elif decay_scheme == "luong234":
        start_decay_step = int(num_train_steps * 2 / 3)
        remain_steps = num_train_steps - start_decay_step
        decay_steps = int(remain_steps / 4)  # decay 4 times
        decay_factor = 0.5
    elif not decay_scheme:  # no decay
        start_decay_step = num_train_steps
        decay_steps = 0
        decay_factor = 1.0
    elif decay_scheme:
        raise ValueError("Unknown decay scheme %s" % decay_scheme)
    print("  decay_scheme=%s, start_decay_step=%d, decay_steps %d, "
          "decay_factor %g" % (decay_scheme, start_decay_step,
                               decay_steps, decay_factor))
    return tf.cond(
        global_step < start_decay_step,
        lambda: learning_rate,
        lambda: tf.train.exponential_decay(
            learning_rate,
            (global_step - start_decay_step),
            decay_steps, decay_factor, staircase=True),
        name="learning_rate_decay_cond")


def _single_cell(unit_type, num_units, forget_bias, dropout, mode,
                 residual_connection=False, device_str=None, residual_fn=None):
    """Create an instance of a single RNN cell."""
    # dropout (= 1 - keep_prob) is set to 0 during eval and infer
    dropout = dropout if mode == tf.contrib.learn.ModeKeys.TRAIN else 0.0

    # Cell Type
    if unit_type == "lstm":
        print("  LSTM, forget_bias=%g" % forget_bias, end='')
        single_cell = rnn.BasicLSTMCell(
            num_units,
            forget_bias=forget_bias)
    elif unit_type == "gru":
        print("  GRU", end='')
        single_cell = rnn.GRUCell(num_units)
    elif unit_type == "layer_norm_lstm":
        print("  Layer Normalized LSTM, forget_bias=%g" % forget_bias, end='')
        single_cell = rnn.LayerNormBasicLSTMCell(
            num_units,
            forget_bias=forget_bias,
            layer_norm=True)
    elif unit_type == "nas":
        print("  NASCell", end='')
        single_cell = rnn.NASCell(num_units)
    else:
        raise ValueError("Unknown unit type %s!" % unit_type)

    # Dropout (= 1 - keep_prob)
    if dropout > 0.0:
        single_cell = rnn.DropoutWrapper(
            cell=single_cell, input_keep_prob=(1.0 - dropout))
        print("  %s, dropout=%g " % (type(single_cell).__name__, dropout), end='')

    # Residual
    if residual_connection:
        single_cell = rnn.ResidualWrapper(
            single_cell, residual_fn=residual_fn)
        print("  %s" % type(single_cell).__name__, end='')

    # Device Wrapper
    if device_str:
        single_cell = rnn.DeviceWrapper(single_cell, device_str)
        print("  %s, device=%s" %
              (type(single_cell).__name__, device_str), end='')

    return single_cell


def create_rnn_cell(unit_type, num_units, num_layers, num_residual_layers,
                    forget_bias, dropout, mode):
    """
    创建RNNCell
    :param unit_type: 隐层单元类型：'lstm', 'gru', 'nas'
    :param num_units: 隐层单元个数
    :param num_layers: rnn cell层数
    :param num_residual_layers:  残差层数
    :param forget_bias: 遗忘门偏置
    :param dropout: dropout
    :param mode: tensorflow.contrib.learn.ModeKeys
    :return:
    """
    cell_list = []
    for i in range(num_layers):
        print("  cell %d" % i, end='')
        residual_connection = i >= (num_layers - num_residual_layers)
        single_cell = _single_cell(
            unit_type=unit_type,
            num_units=num_units,
            forget_bias=forget_bias,
            dropout=dropout,
            mode=mode,
            residual_connection=residual_connection)
        print()
        cell_list.append(single_cell)

    if len(cell_list) == 1:  # Single layer.
        return cell_list[0]
    else:  # Multi layers
        return rnn.MultiRNNCell(cell_list)


def create_attention_mechanism(attention_option, num_units, memory,
                               source_sequence_length):
    """
    Create attention mechanism based on the attention_option.
    :param attention_option: "luong","scaled_luong","bahdanau","normed_bahdanau"
    :param num_units:
    :param memory: The memory to query; usually the output of an RNN encoder.  This
        tensor should be shaped `[batch_size, max_time, ...]`.
    :param source_sequence_length: (optional) Sequence lengths for the batch entries
        in memory.  If provided, the memory tensor rows are masked with zeros
        for values past the respective sequence lengths.
    :return:
    """
    # Mechanism
    if attention_option == "luong":
        attention_mechanism = seq2seq.LuongAttention(
            num_units, memory, memory_sequence_length=source_sequence_length)
    elif attention_option == "scaled_luong":
        attention_mechanism = seq2seq.LuongAttention(
            num_units,
            memory,
            memory_sequence_length=source_sequence_length,
            scale=True)
    elif attention_option == "bahdanau":
        attention_mechanism = seq2seq.BahdanauAttention(
            num_units, memory, memory_sequence_length=source_sequence_length)
    elif attention_option == "normed_bahdanau":
        attention_mechanism = seq2seq.BahdanauAttention(
            num_units,
            memory,
            memory_sequence_length=source_sequence_length,
            normalize=True)
    else:
        raise ValueError("Unknown attention option %s" % attention_option)

    return attention_mechanism


def _create_attention_images_summary(final_context_state):
    """create attention image and attention summary."""
    attention_images = (final_context_state.alignment_history.stack())
    # Reshape to (batch, src_seq_len, tgt_seq_len,1)
    attention_images = tf.expand_dims(
        tf.transpose(attention_images, [1, 2, 0]), -1)
    # Scale to range [0, 255]
    attention_images *= 255
    attention_summary = tf.summary.image("attention_images", attention_images)
    return attention_summary


class TrainModel(
    collections.namedtuple("TrainModel",
                           ("graph",
                            "model",
                            "skip_count_placeholder",
                            "iterator"))):
    pass


def create_train_model(hparams,
                       model_creator,
                       scope=None):
    """Create train graph, model, and iterator."""
    print("# Creating TrainModel...")

    src_train_file = "%s/%s.%s" % (hparams.data_dir, hparams.train_prefix, hparams.src_suffix)
    tgt_train_file = "%s/%s.%s" % (hparams.data_dir, hparams.train_prefix, hparams.tgt_suffix)
    src_vocab_file = "%s/%s.%s" % (hparams.data_dir, hparams.vocab_prefix, hparams.src_suffix)
    tgt_vocab_file = "%s/%s.%s" % (hparams.data_dir, hparams.vocab_prefix, hparams.tgt_suffix)
    batch_size = hparams.batch_size
    num_buckets = hparams.num_buckets

    graph = tf.Graph()

    with graph.as_default(), tf.container(scope or "train"):
        skip_count_placeholder = tf.placeholder(shape=(), dtype=tf.int64)

        vocabulary = Vocabulary(
            src_vocab_file=src_vocab_file,
            tgt_vocab_file=tgt_vocab_file)

        iterator = TrainIterator(
            vocabulary=vocabulary,
            src_data_file=src_train_file,
            tgt_data_file=tgt_train_file,
            batch_size=batch_size,
            num_buckets=num_buckets,
            skip_count=skip_count_placeholder)

        assert isinstance(hparams, tf_training.HParams)

        model_params = get_model_params(
            hparams=hparams,
            vocabulary=vocabulary,
            iterator=iterator)
        model_params.add_hparam('mode', ModeKeys.TRAIN)

        model = model_creator(**model_params.values())

    return TrainModel(
        graph=graph,
        model=model,
        iterator=iterator,
        skip_count_placeholder=skip_count_placeholder)


class EvalModel(
    collections.namedtuple("EvalModel",
                           ("graph",
                            "model",
                            "src_file_placeholder",
                            "tgt_file_placeholder",
                            "iterator"))):
    pass


def create_eval_model(hparams,
                      model_creator,
                      scope=None):
    """Create eval graph, model, src/tgt file holders, and iterator."""
    print("# Creating EvalModel...")

    src_vocab_file = "%s/%s.%s" % (hparams.data_dir, hparams.vocab_prefix, hparams.src_suffix)
    tgt_vocab_file = "%s/%s.%s" % (hparams.data_dir, hparams.vocab_prefix, hparams.tgt_suffix)
    batch_size = hparams.batch_size
    num_buckets = hparams.num_buckets

    graph = tf.Graph()

    with graph.as_default(), tf.container(scope or "eval"):
        src_eval_file_placeholder = tf.placeholder(shape=(), dtype=tf.string)
        tgt_eval_file_placeholder = tf.placeholder(shape=(), dtype=tf.string)

        vocabulary = Vocabulary(
            src_vocab_file=src_vocab_file,
            tgt_vocab_file=tgt_vocab_file)

        iterator = EvalIterator(
            vocabulary=vocabulary,
            src_data_file=src_eval_file_placeholder,
            tgt_data_file=tgt_eval_file_placeholder,
            batch_size=batch_size,
            num_buckets=num_buckets)

        assert isinstance(hparams, tf_training.HParams)

        model_params = get_model_params(
            hparams=hparams,
            vocabulary=vocabulary,
            iterator=iterator)
        model_params.add_hparam('mode', ModeKeys.EVAL)

        model = model_creator(**model_params.values())

    return EvalModel(
        graph=graph,
        model=model,
        src_file_placeholder=src_eval_file_placeholder,
        tgt_file_placeholder=tgt_eval_file_placeholder,
        iterator=iterator)


class InferModel(
    collections.namedtuple("InferModel",
                           ("graph",
                            "model",
                            "src_data_placeholder",
                            "batch_size_placeholder",
                            "iterator"))):
    pass


def create_infer_model(hparams,
                       model_creator,
                       scope=None):
    """Create inference model."""
    print("# Creating InferModel...")

    src_vocab_file = "%s/%s.%s" % (hparams.data_dir, hparams.vocab_prefix, hparams.src_suffix)
    tgt_vocab_file = "%s/%s.%s" % (hparams.data_dir, hparams.vocab_prefix, hparams.tgt_suffix)

    graph = tf.Graph()

    with graph.as_default(), tf.container(scope or "infer"):
        src_data_placeholder = tf.placeholder(shape=[None], dtype=tf.string)
        batch_size_placeholder = tf.placeholder(shape=[], dtype=tf.int64)

        vocabulary = Vocabulary(
            src_vocab_file=src_vocab_file,
            tgt_vocab_file=tgt_vocab_file)

        iterator = InferIterator(
            vocabulary=vocabulary,
            src_data=src_data_placeholder,
            batch_size=batch_size_placeholder)

        assert isinstance(hparams, tf_training.HParams)

        model_params = get_model_params(
            hparams=hparams,
            vocabulary=vocabulary,
            iterator=iterator)
        model_params.add_hparam('mode', ModeKeys.INFER)

        model = model_creator(**model_params.values())

    return InferModel(
        graph=graph,
        model=model,
        src_data_placeholder=src_data_placeholder,
        batch_size_placeholder=batch_size_placeholder,
        iterator=iterator)
