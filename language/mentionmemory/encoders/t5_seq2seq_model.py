# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Contains T5 sequence-to-sequence model."""

import flax.linen as nn
import functools
import jax.numpy as jnp
from typing import List

from language.mentionmemory.encoders import base_encoder
from language.mentionmemory.encoders import encoder_registry
from language.mentionmemory.modules import embedding
from language.mentionmemory.modules import transformer
from language.mentionmemory.utils import default_values
from language.mentionmemory.utils import jax_utils as jut
from language.mentionmemory.utils.custom_types import Array, Dtype, InitType  # pylint: disable=g-multiple-import

from flaxformer.architectures.t5 import t5_architecture
from flaxformer.architectures.t5 import t5_common_layers
from flaxformer.components import dense


ACTIVATIONS = ('gelu', 'linear')
HEAD_DIM = 64


@encoder_registry.register_encoder('t5')
class T5Seq2SeqModel(base_encoder.BaseEncoder):
  """T5 sequence-to-sequence model.

  The T5 sequence-to-sequence model (as in https://arxiv.org/abs/1910.10683).

  Attributes:
    vocab_size: size of token vocabulary.
    hidden_size: dimensionality of token representations.
    intermediate_dim: dimensionality of intermediate representations in MLP.
    entity_dim: dimensionality of entity embeddings.
    num_attention_heads: number of attention heads in Transformer layers.
    num_encoder_layers: number of Transformer layers in the encoder.
    num_decoder_layers: number of Transformer layers in the encoder.
    dtype: data type of encoding (bfloat16 or float32). Parameters and certain
      parts of computation (i.e. loss) are always in float32.
    max_length: maximal number of tokens for pre-training.
    dropout_rate: dropout rate in Transformer layers.
    head_dim: The dimension of the attention head. Default 64.
    activations: The activations to use for the MLP. Default ('gelu', 'linear').
    kernel_init: initialization function for model kernels.
    bias_init: initialization function for model biases.
    layer_norm_epsilon: layer norm constant for numerical stability.
  """
  vocab_size: int
  hidden_size: int
  intermediate_dim: int
  num_attention_heads: int
  num_encoder_layers: int
  num_decoder_layers: int
  dtype: Dtype
  # TODO(urikz): Move this argument out of model parameters
  max_length: int
  dropout_rate: float

  head_dim: int = 64
  activations: List[str] = ('gelu', 'linear')

  kernel_init: InitType = default_values.kernel_init
  bias_init: InitType = default_values.bias_init
  layer_norm_epsilon: float = default_values.layer_norm_epsilon

  def setup(self):

    output_logits_factory = functools.partial(
        dense.DenseGeneral,
        use_bias=False,
        features=self.vocab_size,
        dtype='float32',
        kernel_init=t5_common_layers.MLP_KERNEL_INIT,
        bias_init=t5_common_layers.BIAS_INIT,
    )
    decoder_factory = functools.partial(
        t5_common_layers.decoder,
        num_heads=self.num_attention_heads,
        head_dim=self.head_dim,
        mlp_dim=self.intermediate_dim,
        num_layers=self.num_decoder_layers,
        dropout_rate=self.dropout_rate,
        activations=self.activations,
        output_logits_factory=output_logits_factory,
        dtype=self.dtype)
    encoder_factory = functools.partial(
        t5_common_layers.encoder,
        num_heads=self.num_attention_heads,
        head_dim=self.head_dim,
        mlp_dim=self.intermediate_dim,
        num_layers=self.num_encoder_layers,
        dropout_rate=self.dropout_rate,
        activations=self.activations,
        dtype=self.dtype)
    embedding_factory = functools.partial(
        t5_common_layers.embedding,
        vocabulary_size=self.vocab_size,
        embedding_dim=self.hidden_size,
        dtype=self.dtype)

    self.encoder_decoder = t5_architecture.EncoderDecoder(
        encoder_factory=encoder_factory,
        decoder_factory=decoder_factory,
        shared_token_embedder_factory=embedding_factory,
        dtype=self.dtype)  # pytype: disable=wrong-keyword-args

  def forward(
      self,
      batch,
      deterministic,
  ):
    loss_helpers = {}
    logging_helpers = {}

    # Some default argments for T5
    enable_dropout = not deterministic
    # decode: Whether to prepare and use an autoregressive cache.
    decode = False
    # max_decode_length: An optional integer specifying the maximum decoding
    # length. Note that this is only used for defining the relative position
    # embedding parameters.
    max_decode_length = None

    encoded = self.encoder_decoder.encode(
        batch['source_text_ids'],
        encoder_segment_ids=None,
        encoder_positions=batch['source_position_ids'],
        enable_dropout=enable_dropout)

    target_logits = self.encoder_decoder.decode(
        encoded,
        batch['source_text_ids'],  # Only used for masks.
        batch['target_input_text_ids'],
        batch['target_output_text_ids'],
        encoder_segment_ids=None,
        decoder_segment_ids=None,
        decoder_positions=batch['target_position_ids'],
        enable_dropout=enable_dropout,
        decode=decode,
        max_decode_length=max_decode_length)

    return target_logits, loss_helpers, logging_helpers
