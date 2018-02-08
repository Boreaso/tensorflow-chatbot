import codecs
import os

import tensorflow as tf
from tensorflow.python.ops import lookup_ops

__all__ = ["load_vocab",
           "Vocabulary"]


def load_vocab(vocab_file):
    """加载词汇文件"""
    vocab = []
    with codecs.getreader("utf-8")(tf.gfile.GFile(vocab_file, "rb")) as file:
        vocab_size = 0
        for _word in file:
            vocab_size += 1
            vocab.append(_word.strip())
    return vocab, vocab_size


class Vocabulary:
    UNK = "<unk>"
    SOS = "<s>"
    EOS = "</s>"
    UNK_ID = 0
    SOS_ID = 1
    EOS_ID = 2

    def __init__(self,
                 src_vocab_file,
                 tgt_vocab_file,
                 share_vocab=False):
        self.share_vocab = share_vocab

        # src_vocab_file, src_vocab_size
        validated_result = self._validate_vocab_file(src_vocab_file)
        self._src_vocab_file, self.src_vocab_size = validated_result

        # tgt_vocab_file, tgt_vocab_size
        validated_result = self._validate_vocab_file(tgt_vocab_file)
        self._tgt_vocab_file, self.tgt_vocab_size = validated_result

        # generate tables
        self.src_vocab_table, self.tgt_vocab_table = self._get_vocab_tables()
        self.reverse_src_vocab_table, self.reverse_tgt_vocab_table = self._get_reverse_vocab_table()

    def _validate_vocab_file(self, vocab_file):
        """合法化词汇文件，文件开头三个词汇必须依次为 <unk>, <s>, </s>"""

        print("# Validating file '%s' " % vocab_file)

        if tf.gfile.Exists(vocab_file):
            print("  Vocab file %s exists" % vocab_file)
            vocab, vocab_size = load_vocab(vocab_file)

            assert len(vocab) >= 3
            if vocab[0] != self.UNK or vocab[1] != self.SOS or vocab[2] != self.EOS:
                print("  The first 3 vocab words [%s, %s, %s]"
                      " are not [%s, %s, %s]" %
                      (vocab[0], vocab[1], vocab[2], self.UNK, self.SOS, self.EOS))
                vocab = [self.UNK, self.SOS, self.EOS] + vocab
                vocab_size += 3
                validated_vocab_file = os.path.join(os.path.dirname(vocab_file),
                                                    "validated_" + os.path.basename(vocab_file))
                with codecs.getwriter("utf-8")(
                        tf.gfile.GFile(validated_vocab_file, "wb")) as f:
                    for word in vocab:
                        f.write("%s\n" % word)
                vocab_file = validated_vocab_file
        else:
            raise ValueError("  vocab_file '%s' does not exist." % vocab_file)

        vocab_size = len(vocab)
        return vocab_file, vocab_size

    def _get_vocab_tables(self):
        # 根据文件生成词汇-index映射
        assert self._src_vocab_file and self._tgt_vocab_file

        if tf.gfile.Exists(self._src_vocab_file):
            src_vocab_table = lookup_ops.index_table_from_file(
                self._src_vocab_file, default_value=self.UNK_ID)
            if self.share_vocab:
                tgt_vocab_table = src_vocab_table
            else:
                if tf.gfile.Exists(self._tgt_vocab_file):
                    tgt_vocab_table = lookup_ops.index_table_from_file(
                        self._tgt_vocab_file, default_value=self.UNK_ID)
                else:
                    raise ValueError("tgt_vocab_file '%s' does not exists" % self._tgt_vocab_file)
        else:
            raise ValueError("src_vocab_file '%s' does not exists" % self._src_vocab_file)

        return src_vocab_table, tgt_vocab_table

    def _get_reverse_vocab_table(self):
        # 根据文件生成index-词汇映射
        assert self._src_vocab_file and self._tgt_vocab_file

        if tf.gfile.Exists(self._src_vocab_file):
            reverse_src_vocab_table = lookup_ops.index_to_string_table_from_file(
                self._src_vocab_file, default_value=self.UNK)
            if self.share_vocab:
                reverse_tgt_vocab_table = reverse_src_vocab_table
            else:
                if tf.gfile.Exists(self._tgt_vocab_file):
                    reverse_tgt_vocab_table = lookup_ops.index_to_string_table_from_file(
                        self._tgt_vocab_file, default_value=self.UNK)
                else:
                    raise ValueError("tgt_vocab_file '%s' does not exists" % self._tgt_vocab_file)
        else:
            raise ValueError("src_vocab_file '%s' does not exists" % self._src_vocab_file)

        return reverse_src_vocab_table, reverse_tgt_vocab_table
