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
"""Tests for T5 seq2seq model."""

import copy
import itertools

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from language.mentionmemory.encoders import t5_seq2seq_model
from language.mentionmemory.encoders import encoder_registry
from language.mentionmemory.utils import test_utils
import ml_collections
import numpy as np


class T5Seq2SeqModelTest(parameterized.TestCase):
  """Tests for T5 seq2seq model."""

  encoder_name = 't5'

  encoder_config = {
      'dtype': 'float32',
      'vocab_size': 1000,
      'source_max_length': 32,
      'target_max_length': 32,
      'hidden_size': 3,
      'head_dim': 1,
      'intermediate_dim': 7,
      'num_attention_heads': 3,
      'num_encoder_layers': 2,
      'num_decoder_layers': 2,
      'dropout_rate': 0.1,
  }

  def _gen_batch(self, batch_size, text_length, vocab_size):
    positions = np.tile(np.arange(text_length), [batch_size, 1])
    batch = {
        'source_text_ids':
            np.random.randint(1, vocab_size, size=(batch_size, text_length)),
        'source_position_ids':
            positions,
        'target_input_text_ids':
            np.random.randint(1, vocab_size, size=(
                batch_size,
                text_length,
            )),
        'target_output_text_ids':
            np.random.randint(1, vocab_size, size=(
                batch_size,
                text_length,
            )),
        'target_input_position_ids':
            positions,
    }
    batch = jax.tree_map(jnp.asarray, batch)
    return batch

  @parameterized.parameters((2, 13), (1, 20), (17, 5))
  def test_model_shape(
      self,
      batch_size,
      text_length,
  ):
    """Test model forward runs and produces expected shape."""
    vocab_size = self.encoder_config['vocab_size']

    encoder_class = encoder_registry.get_registered_encoder(self.encoder_name)
    encoder = encoder_class(**self.encoder_config)

    batch = self._gen_batch(batch_size, text_length, vocab_size)

    init_rng = jax.random.PRNGKey(0)
    (output_logits, loss_helpers, _), _ = encoder.init_with_output(
        init_rng,
        batch,
        deterministic=True,
        method=encoder.forward,
    )

    self.assertEqual(output_logits.shape, (batch_size, text_length, vocab_size))


if __name__ == '__main__':
  absltest.main()
