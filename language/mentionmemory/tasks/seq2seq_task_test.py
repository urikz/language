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
"""Tests for EaE task."""

import copy
import itertools
import json

from absl.testing import absltest
from absl.testing import parameterized
import jax
from language.mentionmemory.encoders import import_encoders  # pylint: disable=unused-import
from language.mentionmemory.tasks import task_registry
from language.mentionmemory.tasks import import_tasks  # pylint: disable=unused-import
from language.mentionmemory.utils import test_utils
import ml_collections
import numpy as np

# def gen_eae_test_list():

#   text_lengths = [0, 50, 128]
#   n_mention_list = [0, 5, 10, 15]
#   n_linked_mention_list = [0, 3, 5, 8, 10, 12, 15]
#   no_entity_attention = [True, False]

#   # pylint: disable=g-complex-comprehension
#   test_list = [
#       (text_length, n_mentions, n_linked_mentions, no_entity_attention)
#       for (
#           text_length,
#           n_mentions,
#           n_linked_mentions,
#           no_entity_attention,
#       ) in itertools.product(text_lengths, n_mention_list,
#                              n_linked_mention_list, no_entity_attention)
#       if not (n_mentions *
#               MENTION_SIZE >= text_length or n_linked_mentions > n_mentions)
#   ]

#   return test_list


def generate_sequence(vocab_size, max_length, actual_length):
  s = np.random.randint(vocab_size, size=(max_length,))
  if actual_length < max_length:
    s[actual_length:] = 0
  return s


def generate_sample(vocab_size, source_max_length, source_actual_length,
                    target_max_length, target_actual_length):
  source = generate_sequence(vocab_size, source_max_length,
                             source_actual_length)
  target = generate_sequence(vocab_size, target_max_length,
                             target_actual_length)
  return {
      'source_text_ids': source,
      'source_text_mask': (source > 0).astype(np.int64),
      'target_text_ids': target,
      'target_text_mask': (target > 0).astype(np.int64),
  }


class Seq2SeqTaskTest(test_utils.TestCase):
  """Tests for Seq2Seq task."""

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

  model_config = {
      'encoder_config': encoder_config,
      'encoder_name': encoder_name,
  }

  config = {
      'model_config': model_config,
      'task_name': 'seq2seq',
      'seed': 0,
      'per_device_batch_size': 2,
      'samples_per_example': 1,
  }

  @parameterized.parameters((10, 5), (2, 2), (5, 10))
  def test_loss_fn(self, source_text_length, target_text_length):
    """Test loss function runs and produces expected values."""

    model_config = copy.deepcopy(self.model_config)
    model_config = ml_collections.FrozenConfigDict(model_config)
    config = ml_collections.FrozenConfigDict(self.config)

    vocab_size = model_config.encoder_config.vocab_size
    source_max_length = model_config.encoder_config.source_max_length
    target_max_length = model_config.encoder_config.target_max_length

    task = task_registry.get_registered_task(config.task_name)

    preprocess_fn = task.make_preprocess_fn(config)
    collater_fn = task.make_collater_fn(config)

    model = task.build_model(model_config)

    dummy_input = task.dummy_input(config)
    init_rng = jax.random.PRNGKey(0)
    init_parameters = model.init(init_rng, dummy_input, True)

    samples = []
    for _ in range(config.per_device_batch_size):
      sample = generate_sample(vocab_size, source_max_length,
                               source_text_length, target_max_length,
                               target_text_length)
      sample = preprocess_fn(sample)
      samples.append(sample)

    batch = {k: np.stack([s[k] for s in samples]) for k in samples[0]}
    batch = collater_fn(batch)
    batch = jax.tree_map(np.asarray, batch)

    loss_fn = task.make_loss_fn(config)

    _, metrics, auxiliary_output = loss_fn(
        model_config=model_config,
        model_params=init_parameters['params'],
        model_vars={},
        batch=batch,
        deterministic=True,
    )

    self.assertEqual(metrics['agg']['denominator'],
                     config.per_device_batch_size * (target_text_length - 1))
    self.assertEqual(metrics['exact_match']['denominator'],
                     config.per_device_batch_size)


if __name__ == '__main__':
  absltest.main()
