import tensorflow as tf
import tensorflow.contrib.seq2seq as seq2seq

from . import model_helper
from .basic_model import BasicModel

__all__ = ["AttentionModel"]


class AttentionModel(BasicModel):
    """Sequence-to-sequence dynamic model with attention.

    This class implements a multi-layer recurrent neural network as encoder,
    and an attention-based decoder. This is the same as the model described in
    (Luong et al., EMNLP'2015) paper: https://arxiv.org/pdf/1508.04025v5.pdf.
    This class also allows to use GRU cells in addition to LSTM cells with
    support for dropout.
    """

    def _create_decoder_cell(self, encoder_outputs, encoder_state,
                             source_sequence_length):
        """Build a RNN cell with attention mechanism that can be used by decoder."""
        if not self.attention_option:
            raise ValueError("Parameter 'attention' must be set.")

        attention_option = self.attention_option
        output_attention = self.output_attention

        num_units = self.num_units
        num_layers = self.num_decoder_layers
        num_residual_layers = self.num_decoder_residual_layers
        beam_width = self.beam_width

        dtype = tf.float32

        # Ensure memory is batch-major
        if self.time_major:
            memory = tf.transpose(encoder_outputs, [1, 0, 2])
        else:
            memory = encoder_outputs

        if self.mode == tf.contrib.learn.ModeKeys.INFER and beam_width > 0:
            # The encoder output has been tiled to beam_width via
            # tf.contrib.seq2seq.tile_batch(NOT tf.tile) when use beam search.
            memory = seq2seq.tile_batch(
                memory, multiplier=beam_width)
            source_sequence_length = seq2seq.tile_batch(
                source_sequence_length, multiplier=beam_width)
            encoder_state = seq2seq.tile_batch(
                encoder_state, multiplier=beam_width)
            batch_size = self.batch_size * beam_width
        else:
            batch_size = self.batch_size

        attention_mechanism = model_helper.create_attention_mechanism(
            attention_option, num_units, memory, source_sequence_length)

        cell = model_helper.create_rnn_cell(
            unit_type=self.unit_type,
            num_units=num_units,
            num_layers=num_layers,
            num_residual_layers=num_residual_layers,
            forget_bias=self.forget_bias,
            dropout=self.dropout,
            mode=self.mode)

        # Only generate alignment in greedy INFER mode.
        alignment_history = (self.mode == tf.contrib.learn.ModeKeys.INFER and
                             beam_width == 0)
        cell = seq2seq.AttentionWrapper(
            cell,
            attention_mechanism,
            attention_layer_size=num_units,
            alignment_history=alignment_history,
            output_attention=output_attention,
            name="attention")

        if self.pass_hidden_state:
            decoder_initial_state = cell.zero_state(batch_size, dtype).clone(
                cell_state=encoder_state)
        else:
            decoder_initial_state = cell.zero_state(batch_size, dtype)

        return cell, decoder_initial_state
