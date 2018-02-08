import math
import os
import sys
import time

import tensorflow as tf


def file_line_count(file_path):
    count = 0
    with open(file_path, encoding='utf-8') as file:
        for _ in file:
            count += 1
    return count


def ensure_dir_exist(file_path):
    file_dir = os.path.dirname(file_path)
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)


def safe_exp(value):
    """Exponentiation with catching of overflow error."""
    try:
        ans = math.exp(value)
    except OverflowError:
        ans = float("inf")
    return ans


def add_summary(summary_writer, global_step, tag, value):
    """Add a new summary to the current summary_writer.
    Useful to log things that are not part of the training graph, e.g., tag=BLEU.
    """
    summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
    summary_writer.add_summary(summary, global_step)


def print_time(s, start_time):
    """Take a start time, print elapsed duration, and return a new time."""
    print("%s, time %ds, %s." % (s, (time.time() - start_time), time.ctime()))
    sys.stdout.flush()
    return time.time()


def get_config_proto(log_device_placement=False, allow_soft_placement=True,
                     num_intra_threads=0, num_inter_threads=0):
    # GPU options:
    # https://www.tensorflow.org/versions/r0.10/how_tos/using_gpu/index.html
    config_proto = tf.ConfigProto(
        log_device_placement=log_device_placement,
        allow_soft_placement=allow_soft_placement)
    config_proto.gpu_options.allow_growth = True

    # CPU threads options
    if num_intra_threads:
        config_proto.intra_op_parallelism_threads = num_intra_threads
    if num_inter_threads:
        config_proto.inter_op_parallelism_threads = num_inter_threads

    return config_proto


def print_infer(index,
                inference,
                source=None,
                target=None,
                time=None):
    format_infer = '{:>4} '.format(index)
    if source:
        format_infer += 'Q  > {}\n'.format(source)
    if target:
        format_infer += ' ' * 5 + 'A  > {}\n'.format(target)
    if time:
        format_infer += ' ' * 5 + 'AI > {inf} ({t}s)'.format(
            inf=inference, t=time)
    else:
        format_infer += ' ' * 5 + 'AI > {}'.format(inference)

    print(format_infer)
