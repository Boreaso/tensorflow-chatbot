import os
import sys
from collections import Counter

import jieba

from utils import misc_utils as utils
from utils.vocabulary import Vocabulary as Vocab

PREFIX = [Vocab.UNK, Vocab.SOS, Vocab.EOS]


def split_to_qa_files(corpus_original,
                      questions_path,
                      answers_path,
                      delimiter='|',
                      filters=None):
    if filters is None:
        filters = [r'\r\n']
    with open(corpus_original, encoding='utf-8') as ori_file, \
            open(questions_path, mode='w', encoding='utf-8') as q_file, \
            open(answers_path, mode='w', encoding='utf-8') as a_file:
        count = 0
        for ori_line in ori_file:
            if len(ori_line.strip('\n')) > 0 and delimiter in ori_line:
                splits = ori_line.split(delimiter)
                if len(splits) != 2:
                    continue
                q_line, a_line = splits

                if q_line.strip('\n').strip(' ') in filters or \
                        a_line.strip('\n').strip(' ') in filters:
                    continue
                q_file.writelines(q_line.strip(' ') + '\n')
                a_file.writelines(a_line.strip(' '))
                count += 1
        print('Total num: %d' % count)


def split_xhj_to_qa_files(src_corpus_file,
                          tgt_ques_file,
                          tgt_ans_file,
                          max_size=sys.maxsize):
    with open(src_corpus_file, encoding='utf-8') as corpus_file, \
            open(tgt_ques_file, mode='w', encoding='utf-8') as ques_file, \
            open(tgt_ans_file, mode='w', encoding='utf-8') as ans_file:
        count = 0
        while True:
            if count >= max_size:
                break
            line = corpus_file.readline()
            if not line:
                break
            if line and line.startswith('M'):
                q_line = line[2:]
                a_line = corpus_file.readline()
                if a_line and a_line.startswith('M'):
                    a_line = a_line[2:]
                    ques_file.writelines(q_line)
                    ans_file.writelines(a_line)
                    count += 1


def _get_data_iterator(src_corpus_file, tgt_corpus_file):
    with open(src_corpus_file, mode='r', encoding='utf-8') as src_f, \
            open(tgt_corpus_file, mode='r', encoding='utf-8') as tgt_f:
        while True:
            src_line = src_f.readline().strip()
            tgt_line = tgt_f.readline().strip()

            if not src_line and not tgt_line:
                break
            elif not src_line or not tgt_line:
                continue

            yield src_line, tgt_line


def segment(src_corpus_file,
            tgt_corpus_file,
            src_seg_file,
            tgt_seg_file,
            max_size=sys.maxsize,
            delimiter=' '):
    utils.ensure_dir_exist(src_seg_file)
    utils.ensure_dir_exist(tgt_seg_file)

    with open(src_seg_file, 'w', encoding='utf-8') as src_f, \
            open(tgt_seg_file, 'w', encoding='utf-8') as tgt_f:
        iterator = _get_data_iterator(src_corpus_file, tgt_corpus_file)
        for i, pair in enumerate(iterator):
            if i >= max_size:
                break
            src_line = pair[0]
            tgt_line = pair[1]
            src_segmented = jieba.lcut(src_line.strip())
            tgt_segmented = jieba.lcut(tgt_line.strip())
            src_f.writelines(delimiter.join(src_segmented) + '\n')
            tgt_f.writelines(delimiter.join(tgt_segmented) + '\n')


def _vocab_to_file(corpus_file, vocab_file):
    """
    加载样本文件，全部切词后统计词频，按词频由高到低排序后存储
    """
    with open(corpus_file, 'r', encoding='utf-8') as corpus_f, \
            open(vocab_file, 'w', encoding='utf-8') as vocab_f:
        content = corpus_f.read()
        words_cut = jieba.lcut(content)
        words_count = Counter(words_cut)
        sorted_list = [[kv[1], kv[0]] for kv in words_count.items()]
        sorted_list.sort(reverse=True)
        sorted_list = PREFIX + [item[1] for item in sorted_list if item[1] != '\n']
        vocab_f.writelines('\n'.join(sorted_list))


def generate_vocab(src_corpus_file,
                   tgt_corpus_file,
                   src_vocab_file,
                   tgt_vocab_file):
    utils.ensure_dir_exist(src_vocab_file)
    utils.ensure_dir_exist(tgt_vocab_file)

    _vocab_to_file(src_corpus_file, src_vocab_file)
    _vocab_to_file(tgt_corpus_file, tgt_vocab_file)


def make_data_set(src_data_file,
                  tgt_data_file,
                  prefixs,
                  sizes):
    assert len(prefixs) == len(sizes)

    dst_files = []
    for prefix in prefixs:
        dst_src_file = os.path.join(os.path.dirname(src_data_file),
                                    prefix + '.' + os.path.basename(src_data_file))
        dst_tgt_file = os.path.join(os.path.dirname(tgt_data_file),
                                    prefix + '.' + os.path.basename(tgt_data_file))
        dst_files.append((open(dst_src_file, mode='w', encoding='utf-8'),
                          open(dst_tgt_file, mode='w', encoding='utf-8')))

    with open(src_data_file, encoding='utf-8') as src_f, \
            open(tgt_data_file, encoding='utf-8') as tgt_f:
        src_lines = src_f.readlines()
        tgt_lines = tgt_f.readlines()

        assert len(src_lines) >= sum(sizes) and len(tgt_lines) >= sum(sizes)

        last_index = 0
        for i, size in enumerate(sizes):
            dst_files[i][0].writelines(src_lines[last_index:last_index + size])
            dst_files[i][1].writelines(tgt_lines[last_index:last_index + size])
            last_index = last_index + size

    for dst_file in dst_files:
        dst_file[0].close()
        dst_file[1].close()


if __name__ == '__main__':
    # split_to_qa_files(corpus_original='../corpus/group_chat1.txt',
    #                   questions_path='../corpus/questions.txt',
    #                   answers_path='../corpus/answers.txt')
    split_xhj_to_qa_files(
        src_corpus_file='../corpus/xhj_50w.txt',
        tgt_ques_file='../corpus/questions.txt',
        tgt_ans_file='../corpus/answers.txt',
        max_size=sys.maxsize)
    segment(
        src_corpus_file='../corpus/questions.txt',
        tgt_corpus_file='../corpus/answers.txt',
        src_seg_file='../data/questions',
        tgt_seg_file='../data/answers',
        max_size=200000)
    generate_vocab(
        src_corpus_file='../data/questions',
        tgt_corpus_file='../data/answers',
        src_vocab_file='../data/vocab.questions',
        tgt_vocab_file='../data/vocab.answers')
    make_data_set(
        src_data_file="../data/questions",
        tgt_data_file="../data/answers",
        prefixs=['train', 'dev', 'test'],
        sizes=[198000, 1000, 1000])
