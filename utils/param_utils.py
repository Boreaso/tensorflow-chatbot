import argparse
import os

import tensorflow.contrib.training as tf_training

from utils.vocabulary import Vocabulary


def add_arguments(parser: argparse.ArgumentParser):
    """Build ArgumentParser."""
    assert isinstance(parser, argparse.ArgumentParser)

    parser.register("type", "bool", lambda v: v.lower() == "true")

    # Data
    parser.add_argument("--mode", type=str, default='train',
                        help=""""Specify the mode of program.Options include:
                        train: Training chatbot with train files.
                        sample: Select a sentence randomly from train source file and decode.
                        infer: Do inference with test source file, the results are saved to file 
                               "out_dir/infer_output".
                        eval: Inference and compute bleu score.
                        chat: Accept a input string and output a inference result.""")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Data directory, e.g., en.")
    parser.add_argument("--src_suffix", type=str, default=None,
                        help="Source file suffix, e.g., de.")
    parser.add_argument("--tgt_suffix", type=str, default=None,
                        help="Target file suffix, e.g., de.")
    parser.add_argument("--train_prefix", type=str, default=None,
                        help="Train file prefix.")
    parser.add_argument("--dev_prefix", type=str, default=None,
                        help="Develop file prefix.")
    parser.add_argument("--test_prefix", type=str, default=None,
                        help="Test file prefix.")
    parser.add_argument("--vocab_prefix", type=str, default=None,
                        help="Vocab file prefix.")
    parser.add_argument("--embed_prefix", type=str, default=None,
                        help="Embed file prefix.")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Misc data output directory")

    # Network
    parser.add_argument("--embedding_size", type=int, default=128,
                        help="Word embedding size(word vector length)")
    parser.add_argument("--unit_type", type=str, default='lstm',
                        help="Optional: 'lstm', 'gru' or 'nas'")
    parser.add_argument("--num_units", type=int, default=32,
                        help="The number of hidden units(network size).")
    parser.add_argument("--forget_bias", type=float, default=1.0,
                        help="Forget bias for BasicLSTMCell.")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="Dropout rate (not keep_prob)")
    parser.add_argument("--encoder_type", type=str, default="uni",
                        help="""uni | bi | gnmt. For bi, we build num_layers/2 
                        bi-directional layers.For gnmt, we build 1 bi-directional layer, 
                        and (num_layers - 1) uni-directional layers.""")
    parser.add_argument("--num_encoder_layers", type=int, default=2,
                        help="The number of encoder layers(encoder depth).")
    parser.add_argument("--num_encoder_residual_layers", type=int, default=0, nargs="?",
                        const=True, help="The number of encoder's residual connections.")
    parser.add_argument("--num_decoder_layers", type=int, default=2,
                        help="The number of decoder layers(decoder depth).")
    parser.add_argument("--num_decoder_residual_layers", type=int, default=0, nargs="?",
                        const=True, help="The number of decoder's residual connections.")
    parser.add_argument("--time_major", type="bool", nargs="?", const=True,
                        default=True, help="Whether to use time-major mode for dynamic RNN.")

    # Attention mechanisms
    parser.add_argument("--attention_option", type=str, default="", help="""\
      luong | scaled_luong | bahdanau | normed_bahdanau or set to "" for no attention""")
    parser.add_argument("--output_attention", type="bool", nargs="?", const=True, default=True,
                        help="""Only used in standard attention_architecture. Whether use attention as
                        the cell output at each timestep.""")
    parser.add_argument("--pass_hidden_state", type="bool", nargs="?", const=True, default=True,
                        help="""Whether to pass encoder's hidden state to decoder when using an attention
                        based model.""")

    # Train
    parser.add_argument("--num_train_steps", type=int, default=10000,
                        help="Num steps to train.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for inference mode.")
    parser.add_argument("--optimizer", type=str, default="sgd", help="sgd | rmsprop | adam")
    parser.add_argument("--learning_rate", type=float, default=1.0,
                        help="Learning rate. Adam: 0.001 | 0.0001")
    parser.add_argument("--decay_scheme", type=str, default="",
                        help="""How we decay learning rate. Options include:
                        luong234: after 2/3 num train steps, we start halving the learning rate
                        for 4 times before finishing.
                        luong10: after 1/2 num train steps, we start halving the learning rate
                        for 10 times before finishing.""")

    # initializer
    parser.add_argument("--var_initializer", type=str, default="uniform",
                        help="uniform | glorot_normal | glorot_uniform")
    parser.add_argument("--init_weight", type=float, default=0.1,
                        help=("for uniform init_op, initialize weights "
                              "between [-this, this]."))

    # Sequence lengths
    parser.add_argument("--src_max_len", type=int, default=50,
                        help="Max length of src sequences during training.")
    parser.add_argument("--tgt_max_len", type=int, default=50,
                        help="Max length of tgt sequences during training.")
    parser.add_argument("--src_max_len_infer", type=int, default=None,
                        help="Max length of src sequences during inference.")
    parser.add_argument("--tgt_max_len_infer", type=int, default=None,
                        help="""Max length of tgt sequences during inference.  
                        Also use to restrict the maximum decoding length.""")

    # Inference
    parser.add_argument("--beam_width", type=int, default=0,
                        help=("""beam width when using beam search decoder. 
                        If 0 (default), use standard decoder with greedy helper."""))
    parser.add_argument("--length_penalty_weight", type=float, default=0.0,
                        help="Length penalty for beam search.")
    parser.add_argument("--infer_batch_size", type=int, default=32,
                        help="Batch size for inference mode.")

    # Misc
    parser.add_argument("--sos", type=str, default="<s>",
                        help="Start-of-sentence symbol.")
    parser.add_argument("--eos", type=str, default="</s>",
                        help="End-of-sentence symbol.")
    parser.add_argument("--max_gradient_norm", type=float, default=5.0,
                        help="Clip gradients to this norm.")
    parser.add_argument("--num_buckets", type=int, default=5,
                        help="Put data into similar-length buckets.")
    parser.add_argument("--steps_per_stats", type=int, default=500,
                        help="How many training steps to do per logging.")
    parser.add_argument("--share_vocab", type="bool", nargs="?", const=True, default=False,
                        help="""Whether to use the source vocab and embeddings for both source 
                        and target.""")
    parser.add_argument("--random_seed", type=int, default=None,
                        help="Random seed (>0, set a specific seed).")
    parser.add_argument("--num_keep_ckpts", type=int, default=0,
                        help="Max number of checkpoints to save.")


def create_hparams(flags):
    """Create training hparams."""
    return tf_training.HParams(
        # mode
        mode=flags.mode,

        # Data
        data_dir=flags.data_dir,
        src_suffix=flags.src_suffix,
        tgt_suffix=flags.tgt_suffix,
        train_prefix=flags.train_prefix,
        dev_prefix=flags.dev_prefix,
        test_prefix=flags.test_prefix,
        vocab_prefix=flags.vocab_prefix,
        embed_prefix=flags.embed_prefix,
        out_dir=flags.out_dir,

        # Networks
        embedding_size=flags.embedding_size,
        unit_type=flags.unit_type,
        num_units=flags.num_units,
        forget_bias=flags.forget_bias,
        dropout=flags.dropout,
        encoder_type=flags.encoder_type,
        num_encoder_layers=flags.num_encoder_layers,
        num_encoder_residual_layers=flags.num_encoder_residual_layers,
        num_decoder_layers=flags.num_decoder_layers,
        num_decoder_residual_layers=flags.num_decoder_residual_layers,
        time_major=flags.time_major,

        # Attention mechanisms
        attention_option=flags.attention_option,
        output_attention=flags.output_attention,
        pass_hidden_state=flags.pass_hidden_state,

        # Train
        optimizer=flags.optimizer,
        num_train_steps=flags.num_train_steps,
        batch_size=flags.batch_size,
        var_initializer=flags.var_initializer,
        init_weight=flags.init_weight,
        max_gradient_norm=flags.max_gradient_norm,
        learning_rate=flags.learning_rate,
        decay_scheme=flags.decay_scheme,

        # Data constraints
        num_buckets=flags.num_buckets,
        src_max_len=flags.src_max_len,
        tgt_max_len=flags.tgt_max_len,

        # Inference
        src_max_len_infer=flags.src_max_len_infer,
        tgt_max_len_infer=flags.tgt_max_len_infer,
        infer_batch_size=flags.infer_batch_size,
        beam_width=flags.beam_width,
        length_penalty_weight=flags.length_penalty_weight,

        # Vocab
        sos=flags.sos if flags.sos else Vocabulary.SOS,
        eos=flags.eos if flags.eos else Vocabulary.EOS,

        # Misc
        steps_per_stats=flags.steps_per_stats,
        share_vocab=flags.share_vocab,
        random_seed=flags.random_seed,
        num_keep_ckpts=5  # saves 5 checkpoints by default.
    )


def get_model_params(hparams,
                     vocabulary,
                     iterator):
    assert hparams.data_dir and hparams.src_suffix and hparams.tgt_suffix

    src_vocab_file = os.path.join(
        hparams.data_dir, hparams.vocab_prefix + '.' + hparams.src_suffix) \
        if hparams.vocab_prefix else None
    tgt_vocab_file = os.path.join(
        hparams.data_dir, hparams.vocab_prefix + '.' + hparams.tgt_suffix) \
        if hparams.vocab_prefix else None
    src_embed_file = os.path.join(
        hparams.data_dir, hparams.embed_prefix + '.' + hparams.src_suffix) \
        if hparams.embed_prefix else None
    tgt_embed_file = os.path.join(
        hparams.data_dir, hparams.embed_prefix + '.' + hparams.tgt_suffix) \
        if hparams.embed_prefix else None

    return tf_training.HParams(
        vocabulary=vocabulary,
        iterator=iterator,
        src_vocab_file=src_vocab_file,
        src_embed_file=src_embed_file,
        tgt_vocab_file=tgt_vocab_file,
        tgt_embed_file=tgt_embed_file,
        num_train_steps=hparams.num_train_steps,
        embedding_size=hparams.embedding_size,
        unit_type=hparams.unit_type,
        num_units=hparams.num_units,
        forget_bias=hparams.forget_bias,
        dropout=hparams.dropout,
        encoder_type=hparams.encoder_type,
        num_encoder_layers=hparams.num_encoder_layers,
        num_encoder_residual_layers=hparams.num_encoder_residual_layers,
        num_decoder_layers=hparams.num_decoder_layers,
        num_decoder_residual_layers=hparams.num_decoder_residual_layers,
        optimizer=hparams.optimizer,
        learning_rate=hparams.learning_rate,
        decay_scheme=hparams.decay_scheme,
        beam_width=hparams.beam_width,
        attention_option=hparams.attention_option,
        output_attention=hparams.output_attention,
        pass_hidden_state=hparams.pass_hidden_state,
        var_initializer=hparams.var_initializer,
        init_weight=hparams.init_weight,
        time_major=hparams.time_major,
        max_gradient_norm=hparams.max_gradient_norm,
        num_keep_ckpts=hparams.num_keep_ckpts,
        random_seed=hparams.random_seed)


def combine_hparams(hparams, new_hparams):
    assert hparams and new_hparams

    loaded_config = hparams.values()
    for key in loaded_config:
        if getattr(new_hparams, key) != loaded_config[key]:
            print("# Updating hparams.%s: %s -> %s" %
                  (key, str(loaded_config[key]),
                   str(getattr(new_hparams, key))))
            setattr(hparams, key, getattr(new_hparams, key))
    return hparams


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    FLAGS, unused = parser.parse_known_args()

    loaded_hparams = create_hparams(FLAGS)
    json_str = open('../hparams/chatbot_xhj.json').read()
    loaded_hparams.parse_json(json_str)

    hparams = create_hparams(FLAGS)

    combine_hparams(hparams, loaded_hparams)

    print(loaded_hparams.values())
