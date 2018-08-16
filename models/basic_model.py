import abc

import tensorflow as tf
import tensorflow.contrib.seq2seq as seq2seq
from tensorflow.contrib.learn import ModeKeys
from tensorflow.python.layers.core import Dense

from utils.iterator import DataSetIterator
from utils.vocabulary import Vocabulary
from . import model_helper

__all__ = ['BasicModel']


class Base:
    def __init__(self,
                 num_train_steps,
                 vocabulary,
                 embedding_size,
                 iterator,
                 src_vocab_file=None,
                 src_embed_file=None,
                 tgt_vocab_file=None,
                 tgt_embed_file=None,
                 mode=ModeKeys.TRAIN,
                 unit_type='lstm',
                 num_units=128,
                 forget_bias=1.0,
                 dropout=0.2,
                 encoder_type='bi',
                 num_encoder_layers=1,
                 num_encoder_residual_layers=0,
                 num_decoder_layers=1,
                 num_decoder_residual_layers=0,
                 optimizer='adam',
                 learning_rate=0.001,
                 decay_scheme='',
                 beam_width=0,
                 length_penalty_weight=0,
                 attention_option=None,
                 output_attention=True,
                 pass_hidden_state=True,
                 var_initializer=None,
                 init_weight=0.1,
                 time_major=False,
                 max_gradient_norm=5,
                 num_keep_ckpts=5,
                 random_seed=None):
        """
        :param num_train_steps: 训练step总数（迭代mini-batch总数）
        :param vocabulary: Vocabulary对象，当前词汇表
        :param embedding_size: 词向量维数
        :param iterator: DataSetIterator对象，为模型提供数据以及处理过程
        :param src_vocab_file: 源语料词汇表文件路径
        :param src_embed_file: 源语料词向量文件路径
        :param tgt_vocab_file: 目标语料词汇表文件路径
        :param tgt_embed_file: 目标料词向量文件路径
        :param mode: 训练模式, 'train' | 'eval' | 'infer'
        :param unit_type: RNN网络类型，'lstm' | 'gru'
        :param num_units: 隐层神经元个数（即编码器输出语义向量的大小）
        :param forget_bias: 遗忘门偏置
        :param dropout: dropout层比例
        :param encoder_type: 编码器网络类型，'uni' | 'bi' | 'gnmt'
        :param num_encoder_layers: 编码器RNN层数，如果是'bi' | 'gnmt'，必须是偶数
        :param num_encoder_residual_layers: 编码器残差层数
        :param num_decoder_layers: 解码器RNN层数
        :param num_decoder_residual_layers: 解码器残差层数
        :param optimizer: 优化器，'adam' | 'rmpprop' | 'sgd'
        :param learning_rate: 学习率
        :param decay_scheme: 权重衰减规则
        :param beam_width: Beamsearch带宽
        :param length_penalty_weight: 长度惩罚因子
        :param attention_option: 注意力机制类型，'luong' | 'scaled_luong' | 'bahdanau' | 'normed_bahdanau'
        :param output_attention: 是否使用注意力权重作为编码器每一步的输出。
                Python bool.  If `True` (default), the output at each
                time step is the attention value.  This is the behavior of Luong-style
                attention mechanisms.  If `False`, the output at each time step is
                the output of `cell`.  This is the behavior of Bhadanau-style
                attention mechanisms.  In both cases, the `attention` tensor is
                propagated to the next time step via the state and is used there.
                This flag only controls whether the attention mechanism is propagated
                up to the next cell in an RNN stack or to the top RNN output.
        :param pass_hidden_state: 是否把编码器的隐层状态传给解码器作为输入，false的话解码器从0向量开始解码
        :param var_initializer: 参数初始化器
        :param init_weight: 如果是均匀分布初始化器，该参数指定参数初始化范围（-w, w）
        :param time_major: 是否使用[time，batch_size]的输入维度顺序
        :param max_gradient_norm: 梯度截断阈值
        :param num_keep_ckpts: 最大检查点个数
        :param random_seed: 随机种子
        """
        self.num_train_steps = num_train_steps

        assert vocabulary is not None
        self.vocabulary = vocabulary
        self.src_vocab_size = vocabulary.src_vocab_size
        self.tgt_vocab_size = vocabulary.tgt_vocab_size
        self.embedding_size = embedding_size

        assert isinstance(iterator, DataSetIterator)
        self.iterator = iterator

        # Pretrained embed
        self.src_vocab_file = src_vocab_file
        self.src_embed_file = src_embed_file
        self.tgt_vocab_file = tgt_vocab_file
        self.tgt_embed_file = tgt_embed_file

        # RNN cell
        self.unit_type = unit_type
        self.num_units = num_units
        self.forget_bias = forget_bias
        self.dropout = dropout
        self.encoder_type = encoder_type
        self.num_encoder_layers = num_encoder_layers
        self.num_encoder_residual_layers = num_encoder_residual_layers
        self.num_decoder_layers = num_decoder_layers
        self.num_decoder_residual_layers = num_decoder_residual_layers

        # Learning_rate
        self.learning_rate = learning_rate
        self.decay_scheme = decay_scheme

        # Other params
        self.mode = mode
        self.optimizer = optimizer
        self.beam_width = beam_width
        self.length_penalty_weight = length_penalty_weight
        self.attention_option = attention_option
        self.output_attention = output_attention
        self.pass_hidden_state = pass_hidden_state
        self.var_initializer = var_initializer
        self.init_weight = init_weight  # only for uniform initializer
        self.max_gradient_norm = max_gradient_norm
        self.time_major = time_major
        self.num_keep_ckpts = num_keep_ckpts
        self.random_seed = random_seed

        # Count global step
        self.global_step = tf.Variable(0, trainable=False)

        # Variable initializer
        initializer = model_helper.get_initializer(
            self.var_initializer, self.random_seed, self.init_weight)
        tf.get_variable_scope().set_initializer(initializer)

        # Compute graph
        result = self._build_graph()

        if self.mode == ModeKeys.TRAIN:
            self.train_loss = result[0]
            self.word_count = tf.reduce_sum(
                self.encoder_input_length) + tf.reduce_sum(
                self.decoder_target_length)
        elif self.mode == ModeKeys.EVAL:
            self.eval_loss = result[0]
        elif self.mode == ModeKeys.INFER:
            _, self.infer_logits, self.sample_id, self.final_context_state = result
            self.sample_words = self.vocabulary.reverse_tgt_vocab_table.lookup(
                tf.to_int64(self.sample_id))

        if self.mode != ModeKeys.INFER:
            # Count the number of predicted words for compute ppl.
            self.predict_count = tf.reduce_sum(
                self.decoder_target_length)

        variables = tf.trainable_variables()

        # Gradients update and summary combination.
        if self.mode == ModeKeys.TRAIN:
            # Learning rate
            self.learning_rate = tf.constant(self.learning_rate)

            # warm-up
            # self.learning_rate = model_helper.get_learning_rate_warmup(hparams)

            # decay
            self.learning_rate = model_helper.get_learning_rate_decay(
                num_train_steps=self.num_train_steps,
                global_step=self.global_step,
                learning_rate=self.learning_rate,
                decay_scheme=self.decay_scheme)

            # Optimizer
            if self.optimizer == "sgd":
                opt = tf.train.GradientDescentOptimizer(self.learning_rate)
                tf.summary.scalar("lr", self.learning_rate)
            elif self.optimizer == 'rmsprop':
                opt = tf.train.RMSPropOptimizer(self.learning_rate)
            else:
                opt = tf.train.AdamOptimizer(self.learning_rate)

            # Gradients
            gradients = tf.gradients(self.train_loss, variables)

            clipped_grads, grad_norm_summary, grad_norm = model_helper.gradient_clip(
                gradients, max_gradient_norm=self.max_gradient_norm)
            self.grad_norm = grad_norm

            self.update = opt.apply_gradients(
                grads_and_vars=zip(clipped_grads, variables),
                global_step=self.global_step)

            # Summary
            self.train_summary = tf.summary.merge(
                [tf.summary.scalar("lr", self.learning_rate),
                 tf.summary.scalar("train_loss", self.train_loss)] +
                grad_norm_summary)

        # Saver
        self.saver = tf.train.Saver(
            tf.global_variables(), max_to_keep=self.num_keep_ckpts)

        # Print trainable variables
        print("# Trainable variables")
        for var in variables:
            print("  %s, %s, %s" % (var.name, str(var.get_shape()),
                                    var.op.device))

    def train(self, sess):
        assert self.mode == ModeKeys.TRAIN
        train_res = sess.run(
            [self.update,
             self.train_loss,
             self.predict_count,
             self.train_summary,
             self.global_step,
             self.word_count,
             self.batch_size,
             self.grad_norm,
             self.learning_rate])
        return {'train_loss': train_res[1],
                'predict_count': train_res[2],
                'train_summary': train_res[3],
                'global_step': train_res[4],
                'word_count': train_res[5],
                'batch_size': train_res[6],
                'grad_norm': train_res[7],
                'learning_rate': train_res[8]}

    def eval(self, sess):
        assert self.mode == ModeKeys.EVAL
        eval_res = sess.run(
            [self.eval_loss,
             self.predict_count,
             self.batch_size])
        return {'eval_loss': eval_res[0],
                'predict_count': eval_res[1],
                'batch_size': eval_res[2]}

    def infer(self, sess):
        assert self.mode == ModeKeys.INFER
        infer_res = sess.run(
            [self.infer_logits,
             self.sample_id,
             self.sample_words])
        return {'infer_logits': infer_res[0],
                'sample_id': infer_res[1],
                'sample_words': infer_res[2]}

    def decode(self, sess):
        """Decode a batch.

        Args:
          sess: tensorflow session to use.

        Returns:
          A tuple consiting of outputs, infer_summary.
            outputs: of size [batch_size, time]
        """
        infer_res = self.infer(sess)
        sample_words = infer_res['sample_words']

        # make sure outputs is of shape [batch_size, time] or
        # [beam_width, batch_size, time] when using beam search.
        if self.time_major:
            sample_words = sample_words.transpose()
        elif sample_words.ndim == 3:
            # beam search output in [batch_size, time, beam_width] shape.
            sample_words = sample_words.transpose([2, 0, 1])

        return sample_words

    def _build_graph(self):
        # Data inputs and outputs.
        self._init_io()

        # Batch size
        self.batch_size = tf.size(self.encoder_input_length)

        # Embedding
        self._init_embeddings()

        # Encoder
        encoder_outputs, encoder_state = self._build_encoder()

        # Decoder
        logits, sample_id, final_state = self._build_decoder(
            encoder_outputs, encoder_state)

        # Compute loss
        loss = None
        if self.mode != ModeKeys.INFER:
            loss = self._compute_loss(logits)

        return loss, logits, sample_id, final_state

    def _init_io(self):
        # Encoder
        self.encoder_input = self.iterator.source_input
        self.encoder_input_length = self.iterator.source_sequence_length
        if self.time_major:
            # [batch_size, time] -> [time, batch_size]
            self.encoder_input = tf.transpose(self.encoder_input)

        # Decoder
        if self.mode != ModeKeys.INFER:
            # 训练和评估模式需要target
            self.decoder_target_input = self.iterator.target_input
            self.decoder_target_output = self.iterator.target_output
            self.decoder_target_length = self.iterator.target_sequence_length
            if self.time_major:
                # [batch_size, time] -> [time, batch_size]
                self.decoder_target_input = tf.transpose(self.decoder_target_input)

    def _init_embeddings(self):
        with tf.variable_scope("Embeddings"):
            # Encoder embedding
            self.encoder_embedding = model_helper.create_or_load_embedding(
                embed_name='encoder_embedding',
                vocab_file=self.src_vocab_file,
                embed_file=self.src_embed_file,
                vocab_size=self.src_vocab_size,
                embed_size=self.embedding_size)
            # Decoder embedding
            self.decoder_embedding = model_helper.create_or_load_embedding(
                embed_name='decoder_embedding',
                vocab_file=self.tgt_vocab_file,
                embed_file=self.tgt_embed_file,
                vocab_size=self.tgt_vocab_size,
                embed_size=self.embedding_size)

        with tf.variable_scope("embedded_inputs"):
            # encoder input
            self.encoder_input_embedded = tf.nn.embedding_lookup(
                self.encoder_embedding, self.encoder_input)

            if self.mode != ModeKeys.INFER:
                # decoder input
                self.decoder_target_input_embedded = tf.nn.embedding_lookup(
                    self.decoder_embedding, self.decoder_target_input)

    @abc.abstractmethod
    def _build_encoder(self):
        """This function must be implemented by subclass
        :rtype: tuple
        """
        pass

    def _create_encoder_cell(self, num_layers, num_residual_layers):
        """Build a multi-layer RNN cell that can be used by encoder."""
        return model_helper.create_rnn_cell(
            unit_type=self.unit_type,
            num_units=self.num_units,
            num_layers=num_layers,
            num_residual_layers=num_residual_layers,
            forget_bias=self.forget_bias,
            dropout=self.dropout,
            mode=self.mode)

    def _build_decoder(self, encoder_outputs, encoder_state):
        # TODO init decoder_initial_state
        encoder_outputs = encoder_outputs
        encoder_state = encoder_state

        with tf.variable_scope("Decoder") as scope:
            output_layer = Dense(
                units=self.tgt_vocab_size,
                use_bias=False,
                name='infer_output_layer')

            tgt_sos_id = tf.cast(tf.constant(Vocabulary.SOS_ID), tf.int32)
            tgt_eos_id = tf.cast(tf.constant(Vocabulary.EOS_ID), tf.int32)

            batch_size = self.batch_size
            start_tokens = tf.fill(dims=[batch_size], value=tgt_sos_id)
            end_token = tgt_eos_id

            print("# Building decoder...")

            # Create decoder cell
            decoder_cell, decoder_initial_state = self._create_decoder_cell(
                encoder_outputs=encoder_outputs,
                encoder_state=encoder_state,
                source_sequence_length=self.encoder_input_length)

            if self.mode != ModeKeys.INFER:
                # Dynamic decode for training
                train_helper = seq2seq.TrainingHelper(
                    inputs=self.decoder_target_input_embedded,
                    sequence_length=self.decoder_target_length,
                    time_major=self.time_major,
                    name='decoder_train_helper')

                train_decoder = seq2seq.BasicDecoder(
                    cell=decoder_cell,
                    helper=train_helper,
                    initial_state=decoder_initial_state,
                    output_layer=output_layer)

                train_outputs, final_context_state, _ = seq2seq.dynamic_decode(
                    decoder=train_decoder,
                    output_time_major=self.time_major,
                    swap_memory=True,
                    scope=scope)

                logits = train_outputs.rnn_output
                sample_id = train_outputs.sample_id
            else:
                # Dynamic decode for inference
                if self.beam_width > 0:
                    # The encoder state must be tiled to beam_width via tf.contrib.seq2seq.tile_batch
                    infer_decoder = seq2seq.BeamSearchDecoder(
                        cell=decoder_cell,
                        embedding=self.decoder_embedding,
                        start_tokens=start_tokens,
                        end_token=end_token,
                        initial_state=decoder_initial_state,
                        beam_width=self.beam_width,
                        output_layer=output_layer,
                        length_penalty_weight=self.length_penalty_weight)
                else:
                    infer_helper = seq2seq.GreedyEmbeddingHelper(
                        embedding=self.decoder_embedding,
                        start_tokens=start_tokens,
                        end_token=end_token)

                    infer_decoder = seq2seq.BasicDecoder(
                        cell=decoder_cell,
                        helper=infer_helper,
                        initial_state=decoder_initial_state,
                        output_layer=output_layer)

                infer_outputs, final_context_state, _ = seq2seq.dynamic_decode(
                    decoder=infer_decoder,
                    output_time_major=self.time_major,
                    maximum_iterations=tf.reduce_max(self.encoder_input_length) + 100,
                    swap_memory=True,
                    scope=scope)

                if self.beam_width > 0:
                    logits = tf.no_op()
                    sample_id = infer_outputs.predicted_ids
                else:
                    logits = infer_outputs.rnn_output
                    sample_id = infer_outputs.sample_id

            return logits, sample_id, final_context_state

    @abc.abstractmethod
    def _create_decoder_cell(self, encoder_outputs, encoder_state,
                             source_sequence_length) -> object:
        """This function must be implemented by subclass"""
        pass

    def _get_max_time(self, tensor):
        time_axis = 0 if self.time_major else 1
        return tensor.shape[time_axis].value or tf.shape(tensor)[time_axis]

    # def _compute_loss(self, logits):
    #     # [batch_size, time, embedding_size]
    #     logits = tf.transpose(logits, [1, 0, 2]) if self.time_major else logits
    #     # [batch_size, time]
    #     targets = tf.transpose(self.decoder_target_output, [1, 0]) \
    #         if self.time_major else self.decoder_target_output
    #
    #     loss_weights = tf.sequence_mask(
    #         lengths=self.decoder_target_length,
    #         maxlen=self._get_max_time(targets),
    #         dtype=tf.float32)
    #
    #     loss = seq2seq.sequence_loss(
    #         logits=logits, targets=targets,
    #         weights=loss_weights)
    #
    #     return loss

    def _compute_loss(self, logits):
        """Compute optimization loss."""
        target_output = self.iterator.target_output
        if self.time_major:
            target_output = tf.transpose(target_output)
        max_time = self._get_max_time(target_output)
        cross_ent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=target_output, logits=logits)
        target_weights = tf.sequence_mask(
            self.iterator.target_sequence_length, max_time, dtype=logits.dtype)
        if self.time_major:
            target_weights = tf.transpose(target_weights)

        loss = tf.reduce_sum(
            cross_ent * target_weights) / tf.to_float(self.batch_size)
        return loss


class BasicModel(Base):

    def _build_encoder(self):
        """Build an encoder."""
        encoder_type = self.encoder_type
        num_layers = self.num_encoder_layers
        num_residual_layers = self.num_encoder_residual_layers

        with tf.variable_scope("Encoder") as scope:
            dtype = scope.dtype
            print("# Building encoder...")

            # Encoder_outputs: [max_time, batch_size, num_units]
            if encoder_type == "uni":
                print("  encoder_type=%s, num_layers=%d, num_residual_layers=%d" %
                      (encoder_type, num_layers, num_residual_layers))
                cell = self._create_encoder_cell(num_layers, num_residual_layers)

                encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
                    cell=cell,
                    inputs=self.encoder_input_embedded,
                    dtype=dtype,
                    sequence_length=self.encoder_input_length,
                    time_major=self.time_major,
                    swap_memory=True)
            elif encoder_type == "bi":
                if num_layers % 2 != 0:
                    raise ValueError('Encoder layers must be even number when use bidirectional encoder.')
                num_bi_layers = num_layers // 2
                num_bi_residual_layers = num_residual_layers // 2
                print("  encoder_type=%s, num_layers=%d, num_residual_layers=%d" %
                      (encoder_type, num_layers, num_residual_layers))

                encoder_outputs, bi_encoder_state = self._create_bidirectional_rnn(
                    inputs=self.encoder_input_embedded,
                    sequence_length=self.encoder_input_length,
                    num_bi_layers=num_bi_layers,
                    num_bi_residual_layers=num_bi_residual_layers,
                    dtype=dtype)

                if num_bi_layers == 1:
                    encoder_state = bi_encoder_state
                else:
                    # alternatively concat forward and backward states
                    encoder_state = []
                    for layer_id in range(num_bi_layers):
                        encoder_state.append(bi_encoder_state[0][layer_id])  # forward
                        encoder_state.append(bi_encoder_state[1][layer_id])  # backward
                    encoder_state = tuple(encoder_state)
            else:
                raise ValueError("Unknown encoder_type %s" % encoder_type)

        return encoder_outputs, encoder_state

    def _create_bidirectional_rnn(self, inputs, sequence_length,
                                  num_bi_layers, num_bi_residual_layers,
                                  dtype):
        """
        Create and call bidirectional RNN cells.
        :param inputs: data inputs
        :param sequence_length: sequence length
        :param dtype: data type
        :return: The concatenated bidirectional output and the bidirectional RNN cell"s
                 state.
        """
        # Construct forward and backward encoder cells
        fw_cell = self._create_encoder_cell(num_bi_layers, num_bi_residual_layers)
        bw_cell = self._create_encoder_cell(num_bi_layers, num_bi_residual_layers)

        bi_outputs, bi_state = tf.nn.bidirectional_dynamic_rnn(
            fw_cell,
            bw_cell,
            inputs,
            dtype=dtype,
            sequence_length=sequence_length,
            time_major=self.time_major,
            swap_memory=True)

        return tf.concat(bi_outputs, -1), bi_state

    def _create_decoder_cell(self, encoder_outputs, encoder_state,
                             source_sequence_length):
        """Build an RNN cell that can be used by decoder."""

        # We only make use of encoder_outputs in attention-based models
        if self.attention_option:
            raise ValueError("BasicModel doesn't support attention.")

        cell = model_helper.create_rnn_cell(
            unit_type=self.unit_type,
            num_units=self.num_units,
            num_layers=self.num_decoder_layers,
            num_residual_layers=self.num_decoder_residual_layers,
            forget_bias=self.forget_bias,
            dropout=self.dropout,
            mode=self.mode)

        if self.mode == ModeKeys.INFER and self.beam_width > 0:
            # For beam search, we need to replicate encoder state `beam_width` times
            decoder_initial_state = seq2seq.tile_batch(
                encoder_state, multiplier=self.beam_width)
        else:
            decoder_initial_state = encoder_state

        return cell, decoder_initial_state
