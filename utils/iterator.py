import tensorflow as tf

from .vocabulary import Vocabulary

__all__ = ["DataSetIterator",
           "TrainIterator",
           "EvalIterator",
           "InferIterator"]


class DataSetIterator:

    def __init__(self,
                 vocabulary,
                 random_seed=123,
                 batch_size=1,
                 num_buckets=1,
                 src_max_len=None,
                 tgt_max_len=None,
                 delimiter=' ',
                 num_parallel_calls=4,
                 output_buffer_size=None,
                 skip_count=None,
                 reshuffle_each_iteration=True):
        """
        创建一个数据迭代器
        :param vocabulary: Vocabulary对象
        :param random_seed: 随机种子
        :param batch_size: 数据批大小
        :param num_buckets: 把整个数据集按长度进行bucket处理的bucket数量
        :param src_max_len: 源序列最大长度，如果被设置，超出长度的序列将被过滤
        :param tgt_max_len: 目标序列最大长度，如果被设置，超出长度的序列将被过滤
        :param delimiter: 原始数据文件的定界符，由数据预处理程序确定
        :param num_parallel_calls: 数据并行处理的数量，如未被设置，数据将被顺序处理
        :param output_buffer_size: 数据集最大缓冲数，限制输出数据集的大小
        :param skip_count: 跳过数据集开始skip_count个数据，如值-1则跳过整个数据集
        :param reshuffle_each_iteration: 是否每次迭代都重新混洗数据
        """
        assert vocabulary and isinstance(vocabulary, Vocabulary)

        self._vocabulary = vocabulary
        self._random_seed = random_seed
        self._batch_size = batch_size
        self._num_buckets = num_buckets
        self._src_max_len = src_max_len
        self._tgt_max_len = tgt_max_len
        self._delimiter = delimiter
        self._num_parallel_calls = num_parallel_calls
        self._output_buffer_size = output_buffer_size
        self._skip_count = skip_count
        self._reshuffle_each_iteration = reshuffle_each_iteration

        # 通过子类的_init_iterator方法初始化
        self.initializer = None
        self.source_input = None
        self.target_input = None
        self.target_output = None
        self.source_sequence_length = None
        self.target_sequence_length = None

        self._init_iterator()

    def _init_iterator(self):
        """初始化数据集迭代器，由子类实现"""
        pass


class TrainIterator(DataSetIterator):
    def __init__(self,
                 src_data_file,
                 tgt_data_file,
                 **kwargs):
        self._src_dataset = tf.data.TextLineDataset(src_data_file)
        self._tgt_dataset = tf.data.TextLineDataset(tgt_data_file)
        super(TrainIterator, self).__init__(**kwargs)

    def _init_iterator(self):
        """创建训练数据集迭代器"""
        src_eos_id = tf.cast(self._vocabulary.EOS_ID, tf.int32)
        tgt_sos_id = tf.cast(self._vocabulary.SOS_ID, tf.int32)
        tgt_eos_id = tf.cast(self._vocabulary.EOS_ID, tf.int32)

        if not self._output_buffer_size:
            self._output_buffer_size = self._batch_size * 1000

        src_tgt_dataset = tf.data.Dataset.zip((self._src_dataset, self._tgt_dataset))

        # 用于支持多个worker训练，每个worker可以获得一个唯一的数据集，
        # 该数据集包含原始数据集的1/num_shards条数据
        # src_tgt_dataset = src_tgt_dataset.shard(self.num_shards, self._shard_index)

        # 跳过数据集的前skip_count个数据
        if self._skip_count is not None:
            src_tgt_dataset = src_tgt_dataset.skip(self._skip_count)

        # 数据混洗
        src_tgt_dataset = src_tgt_dataset.shuffle(
            self._output_buffer_size, self._random_seed, self._reshuffle_each_iteration)

        # 对切好词的数据集进行切割，src_tgt_dataset:((num_samples,None),(num_samples, None))
        src_tgt_dataset = src_tgt_dataset.map(
            lambda src, tgt: (tf.string_split([src], delimiter=self._delimiter).values,
                              tf.string_split([tgt], delimiter=self._delimiter).values),
            num_parallel_calls=self._num_parallel_calls).prefetch(self._output_buffer_size)

        # 过滤长度为0的数据条目
        src_tgt_dataset = src_tgt_dataset.filter(
            lambda src, tgt: tf.logical_and(tf.size(src) > 0, tf.size(tgt) > 0))

        # 过滤长度超出最大长度的数据条目
        if self._src_max_len:
            src_tgt_dataset = src_tgt_dataset.filter(
                lambda src, tgt: tf.size(src) < self._src_max_len).prefetch(self._output_buffer_size)
        if self._tgt_max_len:
            src_tgt_dataset = src_tgt_dataset.filter(
                lambda src, tgt: tf.size(tgt) < self._tgt_max_len).prefetch(self._output_buffer_size)

        # 把原始数据集中的字符转为词汇表中的id
        src_tgt_dataset = src_tgt_dataset.map(
            lambda src, tgt: (tf.cast(self._vocabulary.src_vocab_table.lookup(src), tf.int32),
                              tf.cast(self._vocabulary.tgt_vocab_table.lookup(tgt), tf.int32)),
            num_parallel_calls=self._num_parallel_calls).prefetch(self._output_buffer_size)

        # 为数据集的每个target前后加上<s>和</s>,表示句子的开始和结束
        # src_tgt_dataset:((num_src, None),(num_tgt_input, None),(num_tgt_output, None))
        src_tgt_dataset = src_tgt_dataset.map(
            lambda src, tgt: (src,
                              tf.concat(([tgt_sos_id], tgt), axis=0),
                              tf.concat((tgt, [tgt_eos_id]), axis=0)),
            num_parallel_calls=self._num_parallel_calls).prefetch(self._output_buffer_size)

        # 计算数据集每个条目的长度
        # src_tgt_dataset: ((num_src, None), (num_tgt_in, None), (num_tgt_out, None),
        # (num_src, 1), (num_tgt, 1))
        src_tgt_dataset = src_tgt_dataset.map(
            lambda src, tgt_in, tgt_out: (
                src, tgt_in, tgt_out, tf.size(src), tf.size(tgt_in)),
            num_parallel_calls=self._num_parallel_calls).prefetch(self._output_buffer_size)

        # 对数据集进行Bucket处理，根据源序列的长度划分为形如[[0-9], [10-19], ...]
        def batching_func(x):
            return x.padded_batch(
                self._batch_size,
                # The first three entries are the source and target line rows;
                # these have unknown-length vectors.  The last two entries are
                # the source and target row sizes; these are scalars.
                padded_shapes=(
                    tf.TensorShape([None]),  # src
                    tf.TensorShape([None]),  # tgt_input
                    tf.TensorShape([None]),  # tgt_output
                    tf.TensorShape([]),      # src_len
                    tf.TensorShape([])),     # tgt_len
                # Pad the source and target sequences with eos tokens.
                # (Though notice we don't generally need to do this since
                # later on we will be masking out calculations past the true sequence.
                padding_values=(
                    src_eos_id,  # src
                    tgt_eos_id,  # tgt_input
                    tgt_eos_id,  # tgt_output
                    0,           # src_len -- unused
                    0))          # tgt_len -- unused

        if self._num_buckets > 1:
            def key_func(unused_1, unused_2, unused_3, src_len, tgt_len):
                # 计算每个bucket的宽度，把数据集分为[0, bucket_width)，
                # [bucket_width, 2 * bucket_width)...的数据块，超过((num_bucket-1) * bucket_width)
                # 长度的数据都存入最后一个bucket
                if self._src_max_len:
                    bucket_width = (self._src_max_len + self._num_buckets - 1) // self._num_buckets
                else:
                    bucket_width = 10

                # 根据源序列和目标序列长度计算每个数据项所属的bucket_id
                bucket_id = tf.maximum(src_len // bucket_width, tgt_len // bucket_width)
                return tf.to_int64(tf.minimum(self._num_buckets, bucket_id))

            def reduce_func(unused_key, windowed_data):
                return batching_func(windowed_data)

            batched_dataset = src_tgt_dataset.apply(
                tf.contrib.data.group_by_window(
                    key_func=key_func, reduce_func=reduce_func,
                    window_size=self._batch_size))
        else:
            batched_dataset = batching_func(src_tgt_dataset)

        iterator = batched_dataset.make_initializable_iterator()
        next_batch = iterator.get_next()
        self.initializer = iterator.initializer
        self.source_input = next_batch[0]
        self.target_input = next_batch[1]
        self.target_output = next_batch[2]
        self.source_sequence_length = next_batch[3]
        self.target_sequence_length = next_batch[4]


class EvalIterator(TrainIterator):
    pass


class InferIterator(DataSetIterator):
    def __init__(self,
                 src_data,
                 **kwargs):
        self._src_dataset = tf.data.Dataset.from_tensor_slices(src_data)
        super(InferIterator, self).__init__(**kwargs)

    def _init_iterator(self):
        """创建推断数据集迭代器"""
        src_eos_id = tf.cast(self._vocabulary.EOS_ID, tf.int32)
        src_dataset = self._src_dataset.map(lambda src: tf.string_split([src]).values)

        if self._src_max_len:
            src_dataset = src_dataset.filter(lambda src: tf.size(src) < self._src_max_len)

        # 把原始数据集中的字符转为词汇表中的id
        src_dataset = src_dataset.map(
            lambda src: tf.cast(self._vocabulary.src_vocab_table.lookup(src), tf.int32))

        # 计算数据集每个条目的长度.
        src_dataset = src_dataset.map(lambda src: (src, tf.size(src)))

        # 用于推断的iterator不用bucket处理，直接batching
        def batching_func(x):
            return x.padded_batch(
                self._batch_size,
                # The entry is the source line rows;
                # this has unknown-length vectors.  The last entry is
                # the source row size; this is a scalar.
                padded_shapes=(
                    tf.TensorShape([None]),  # src(unknown-length vector)
                    tf.TensorShape([])),  # src_len(scalar)
                # Pad the source sequences with eos tokens.
                # (Though notice we don't generally need to do this since
                # later on we will be masking out calculations past the true sequence.
                padding_values=(
                    src_eos_id,  # src
                    0))  # src_len -- unused

        batched_dataset = batching_func(src_dataset)
        iterator = batched_dataset.make_initializable_iterator()
        next_batch = iterator.get_next()
        self.initializer = iterator.initializer
        self.source_input = next_batch[0]
        self.source_sequence_length = next_batch[1]
