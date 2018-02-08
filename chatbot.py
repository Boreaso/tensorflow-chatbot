import argparse
import json
import os
import random
import time

import jieba
import numpy as np
import tensorflow as tf

from models import model_helper
from models.attention_model import AttentionModel
from models.basic_model import BasicModel
from utils import eval_utils
from utils import misc_utils as utils
from utils import param_utils
from utils import train_utils
from utils.vocabulary import Vocabulary

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class ChatBot:

    def __init__(self, hparams):
        self.hparams = hparams

        # Data locations
        self.out_dir = hparams.out_dir
        self.model_dir = os.path.join(self.out_dir, 'ckpts')
        if not tf.gfile.Exists(self.model_dir):
            tf.gfile.MakeDirs(self.model_dir)

        self.train_src_file = os.path.join(hparams.data_dir, hparams.train_prefix + '.' + hparams.src_suffix)
        self.train_tgt_file = os.path.join(hparams.data_dir, hparams.train_prefix + '.' + hparams.tgt_suffix)
        self.test_src_file = os.path.join(hparams.data_dir, hparams.test_prefix + '.' + hparams.src_suffix)
        self.test_tgt_file = os.path.join(hparams.data_dir, hparams.test_prefix + '.' + hparams.tgt_suffix)
        self.dev_src_file = os.path.join(hparams.data_dir, hparams.dev_prefix + '.' + hparams.src_suffix)
        self.dev_tgt_file = os.path.join(hparams.data_dir, hparams.dev_prefix + '.' + hparams.tgt_suffix)
        self.infer_out_file = os.path.join(self.out_dir, 'infer_output')
        self.eval_out_file = os.path.join(self.out_dir, 'eval_output')

        # Create models
        attention_option = hparams.attention_option

        if attention_option:
            model_creator = AttentionModel
        else:
            model_creator = BasicModel

        self.train_model = model_helper.create_train_model(
            hparams=hparams,
            model_creator=model_creator)
        self.eval_model = model_helper.create_eval_model(
            hparams=hparams,
            model_creator=model_creator)
        self.infer_model = model_helper.create_infer_model(
            hparams=hparams,
            model_creator=model_creator)

        # Sessions
        config_proto = utils.get_config_proto()
        self.train_sess = tf.Session(config=config_proto, graph=self.train_model.graph)
        self.eval_sess = tf.Session(config=config_proto, graph=self.eval_model.graph)
        self.infer_sess = tf.Session(config=config_proto, graph=self.infer_model.graph)

        # EOS
        self.tgt_eos = Vocabulary.EOS.encode("utf-8")

    def train(self):
        hparams = self.hparams
        train_model = self.train_model
        train_sess = self.train_sess
        model_dir = self.model_dir

        steps_per_stats = hparams.steps_per_stats
        num_train_steps = hparams.num_train_steps

        summary_name = "train_log"

        # Load train model
        with self.train_model.graph.as_default():
            loaded_train_model, global_step = model_helper.create_or_load_model(
                self.train_model.model, self.model_dir, self.train_sess, "train")

        # Summary writer
        summary_writer = tf.summary.FileWriter(
            os.path.join(self.out_dir, summary_name), train_model.graph)

        # Initialize dataset iterator
        train_sess.run(
            train_model.iterator.initializer,
            feed_dict={train_model.skip_count_placeholder: 0})

        loss_track = []
        training_start_time = time.time()
        epoch_count = 0
        last_stats_step = global_step
        stats = train_utils.init_stats()
        best_bleu_score = 0

        while global_step < num_train_steps:
            # Run a training step
            start_time = time.time()
            try:
                train_result = loaded_train_model.train(train_sess)
            except tf.errors.OutOfRangeError:
                # Finished going through the training dataset. Go to next epoch.
                epoch_count += 1
                print("# Finished epoch %d, step %d." %
                      (epoch_count, global_step))

                # Save model params
                loaded_train_model.saver.save(
                    train_sess,
                    os.path.join(model_dir, "chatbot.ckpt"),
                    global_step=global_step)

                # Do evaluation
                self.eval(best_bleu_score)

                train_sess.run(
                    train_model.iterator.initializer,
                    feed_dict={train_model.skip_count_placeholder: 0})
                continue

            # Write step summary and accumulate statistics
            global_step = train_utils.update_stats(
                stats, summary_writer, start_time,
                train_result.values(), best_bleu_score)

            loss_track.append(train_result['train_loss'])

            if global_step - last_stats_step >= steps_per_stats:
                last_stats_step = global_step
                is_overflow = train_utils.check_stats(stats, global_step, steps_per_stats)
                if is_overflow:
                    break

                # Reset statistics
                stats = train_utils.init_stats()

        # Training done.
        loaded_train_model.saver.save(
            train_sess,
            os.path.join(model_dir, "chatbot.ckpt"),
            global_step=global_step)

        summary_writer.close()

        print('Training done. Total time: %.4f' % (time.time() - training_start_time))

    def eval(self, best_bleu_score=0):
        print('# Doing evaluation...')
        # inference to file 'out_dir/infer_output'
        # self.infer()

        if best_bleu_score == 0 and \
                os.path.exists(self.eval_out_file):
            eval_json = json.load(open(file=self.eval_out_file))
            best_bleu_score = eval_json['best_bleu']

        bleu_score = eval_utils.bleu_score(
            ref_file=self.test_tgt_file,
            trans_file=self.infer_out_file)

        if bleu_score > best_bleu_score:
            best_bleu_score = bleu_score
            json.dump({'best_bleu': best_bleu_score},
                      open(file=self.eval_out_file, mode='w'))

        print('bleu score: ', best_bleu_score)

        # Sample decode
        self.sample_decode()

        return bleu_score

    def sample_decode(self, num_sentences=1):
        """Sample decode num_sentences random sentence from src_data."""
        model_dir = self.model_dir
        infer_model = self.infer_model
        infer_sess = self.infer_sess
        train_src_file = self.train_src_file
        train_tgt_file = self.train_tgt_file
        beam_width = self.hparams.beam_width

        start_time = time.time()

        # Load infer model
        with infer_model.graph.as_default():
            loaded_infer_model, global_step = model_helper.create_or_load_model(
                infer_model.model, model_dir, infer_sess, "infer")

        src_data = open(train_src_file, encoding='utf-8').readlines()
        tgt_data = open(train_tgt_file, encoding='utf-8').readlines()

        for _ in range(num_sentences):
            decode_id = random.randint(0, len(src_data) - 1)
            print("# Decoding sentence %d" % decode_id)

            iterator_feed_dict = {
                infer_model.src_data_placeholder: [src_data[decode_id]],
                infer_model.batch_size_placeholder: 1
            }
            infer_sess.run(
                self.infer_model.iterator.initializer,
                feed_dict=iterator_feed_dict)

            sample_words = loaded_infer_model.decode(infer_sess)

            if beam_width > 0:
                # get the top translation.
                sample_words = sample_words[0]

            response = self._get_response(sample_words)

            print("  src: %s" % src_data[decode_id], end='')
            print("  ref: %s" % tgt_data[decode_id], end='')
            print("  bot: %s" % response)
            print("  tim: %.4fs" % (time.time() - start_time))

    def infer(self, num_print_per_batch=0):
        model_dir = self.model_dir
        out_dir = self.out_dir
        dev_src_file = self.dev_src_file
        dev_tgt_file = self.dev_tgt_file
        infer_batch_size = self.hparams.infer_batch_size
        beam_width = self.hparams.beam_width
        infer_model = self.infer_model
        infer_sess = self.infer_sess

        infer_output_file = os.path.join(out_dir, 'infer_output')

        start_time = time.time()
        print('# Decoding to %s' % infer_output_file)

        # Load infer model
        with infer_model.graph.as_default():
            loaded_infer_model, global_step = model_helper.create_or_load_model(
                infer_model.model, model_dir, infer_sess, "infer")

        with open(dev_src_file, encoding='utf-8') as in_src_file, \
                open(dev_tgt_file, encoding='utf-8') as in_tgt_file, \
                open(infer_output_file, mode='w', encoding='utf-8') as out_file:
            infer_src_data = in_src_file.readlines()
            infer_tgt_data = in_tgt_file.readlines()

            iterator_feed_dict = {
                infer_model.src_data_placeholder: infer_src_data,
                infer_model.batch_size_placeholder: infer_batch_size
            }
            infer_sess.run(
                infer_model.iterator.initializer,
                feed_dict=iterator_feed_dict)

            num_sentences = 0
            while True:
                try:
                    # The shape of sample_words is [batch_size, time] or
                    # [beam_width, batch_size, time] when using beam search.
                    sample_words = loaded_infer_model.decode(infer_sess)

                    if beam_width == 0:
                        sample_words = np.expand_dims(sample_words, 0)

                    batch_size = sample_words.shape[1]

                    for sent_id in range(batch_size):
                        beam_id = random.randint(0, beam_width - 1) if beam_width > 0 else 0
                        response = self._get_response(sample_words[beam_id][sent_id])
                        out_file.write(response + '\n')

                        if sent_id < num_print_per_batch:
                            sent_id += num_sentences
                            print("  sentence %d" % sent_id)
                            print("  src: %s" % infer_src_data[sent_id], end='')
                            print("  ref: %s" % infer_tgt_data[sent_id], end='')
                            print("  bot: %s" % response)

                    num_sentences += batch_size
                except tf.errors.OutOfRangeError:
                    utils.print_time(
                        "  done, num sentences %d, beam width %d" %
                        (num_sentences, beam_width), start_time)
                    break

    def chat(self):
        """Accept a input str and get response by trained model."""
        model_dir = self.model_dir
        infer_model = self.infer_model
        infer_sess = self.infer_sess
        beam_width = self.hparams.beam_width

        # Load infer model
        with infer_model.graph.as_default():
            loaded_infer_model, global_step = model_helper.create_or_load_model(
                infer_model.model, model_dir, infer_sess, "infer")

        # Warm up jieba
        jieba.lcut("jieba")

        while True:
            input_str = input('Me > ')
            if not input_str.strip():
                continue

            input_seg = jieba.lcut(input_str)
            start_time = time.time()

            iterator_feed_dict = {
                infer_model.src_data_placeholder: input_seg,
                infer_model.batch_size_placeholder: 1
            }
            infer_sess.run(
                self.infer_model.iterator.initializer,
                feed_dict=iterator_feed_dict)

            sample_words = loaded_infer_model.decode(infer_sess)

            if beam_width > 0:
                # Get a random answer.
                beam_id = random.randint(0, beam_width - 1)
                sample_words = sample_words[beam_id]

            response = self._get_response(sample_words)

            print("AI > %s (%.4fs)" % (response, time.time() - start_time))

    def _get_eval_perplexity(self, name):
        model_dir = self.model_dir
        eval_model = self.eval_model
        eval_sess = self.eval_sess

        with eval_model.graph.as_default():
            loaded_eval_model, global_step = model_helper.create_or_load_model(
                eval_model.model, model_dir, eval_sess, 'eval')

        dev_eval_iterator_feed_dict = {
            eval_model.src_file_placeholder: self.dev_src_file,
            eval_model.tgt_file_placeholder: self.dev_tgt_file
        }

        dev_ppl = eval_utils.internal_eval(
            eval_model, global_step, eval_sess, eval_model.iterator,
            dev_eval_iterator_feed_dict, name)

        return dev_ppl

    def _get_response(self, sample_words):
        tgt_eos = self.tgt_eos
        # Make sure sample_words has 1 dim.
        sample_words = sample_words.flatten().tolist()

        if tgt_eos and tgt_eos in sample_words:
            sample_words = sample_words[:sample_words.index(tgt_eos)]

        response = ' '.join([word.decode() for word in sample_words])

        return response


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    param_utils.add_arguments(parser)
    FLAGS, unused = parser.parse_known_args()

    hparams = param_utils.create_hparams(FLAGS)

    json_str = open('hparams/chatbot_xhj.json').read()
    loaded_hparams = param_utils.create_hparams(FLAGS)
    loaded_hparams.parse_json(json_str)

    param_utils.combine_hparams(hparams, loaded_hparams)

    hparams.mode = 'infer'

    chatbot = ChatBot(hparams)
    if hparams.mode == 'train':
        chatbot.train()
    elif hparams.mode == 'sample':
        chatbot.sample_decode()
    elif hparams.mode == 'infer':
        chatbot.infer(30)
    elif hparams.mode == 'eval':
        chatbot.eval()
    elif hparams.mode == 'chat':
        chatbot.chat()
    else:
        raise ValueError("Invalid value of 'mode' param "
                         "(train | sample | infer | eval | chat).")
