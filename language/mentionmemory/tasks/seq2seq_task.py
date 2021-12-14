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
"""Base sequence to sequence task."""

import flax.linen as nn
import jax.numpy as jnp
from language.mentionmemory.encoders import encoder_registry
from language.mentionmemory.tasks import base_task
from language.mentionmemory.tasks import task_registry
from language.mentionmemory.utils import metric_utils
from language.mentionmemory.utils.custom_types import Array, Dtype, MetricGroups  # pylint: disable=g-multiple-import
import ml_collections
import numpy as np
import tensorflow.compat.v2 as tf


class Seq2SeqModel(nn.Module):

  encoder_name: str
  encoder_config: ml_collections.FrozenConfigDict

  def setup(self):
    self.encoder = encoder_registry.get_registered_encoder(
        self.encoder_name)(**self.encoder_config)

  def __call__(self, batch, deterministic):
    loss_helpers, logging_helpers = {}, {}
    output_logits, loss_helpers, logging_helpers = self.encoder.forward(
        batch, deterministic)
    loss_helpers['output_logits'] = output_logits
    return loss_helpers, logging_helpers


@task_registry.register_task('seq2seq')
class Seq2SeqTask(base_task.BaseTask):
  """Sequence to sequence task."""

  model_class = Seq2SeqModel

  @classmethod
  def make_loss_fn(cls, config):
    """Creates task loss function."""

    def loss_fn(
        model_config,
        model_params,
        model_vars,
        batch,
        deterministic,
        dropout_rng=None,
    ):
      """Model-specific loss function.

      See BaseTask.

      Args:
        model_config: contains model config hyperparameters.
        model_params: contains model parameters.
        model_vars: contains model variables (not optimized).
        batch: model input.
        deterministic: whether dropout etc should be applied.
        dropout_rng: seed for dropout randomness.

      Returns:
        Loss and metrics.
      """
      variable_dict = {'params': model_params}
      variable_dict.update(model_vars)
      loss_helpers, logging_helpers = cls.build_model(model_config).apply(
          variable_dict, batch, deterministic=deterministic, rngs=dropout_rng)

      weights = (batch['target_output_text_ids'] > 0).astype(jnp.float32)

      # target_output_text_ids
      loss, denom = metric_utils.compute_weighted_cross_entropy(
          loss_helpers['output_logits'], batch['target_output_text_ids'],
          weights)

      acc, _ = metric_utils.compute_weighted_accuracy(
          loss_helpers['output_logits'], batch['target_output_text_ids'],
          weights)

      per_token_errors = jnp.not_equal(
          jnp.argmax(loss_helpers['output_logits'], axis=-1),
          batch['target_output_text_ids'])
      per_token_errors = per_token_errors * weights
      exact_match = (per_token_errors.sum(axis=-1) == 0).astype(jnp.int32)
      exact_match_denom = (weights.sum(axis=-1) > 0).sum()

      metrics = {
          'agg': {
              'loss': loss,
              'denominator': denom,
              'acc': acc,
          },
          'exact_match': {
              'total': exact_match,
              'denominator': exact_match_denom,
          }
      }

      auxiliary_output = {}
      return loss, metrics, auxiliary_output

    return loss_fn

  @staticmethod
  def make_preprocess_fn(config):
    """Produces function to preprocess samples. See BaseTask."""
    encoder_config = config.model_config.encoder_config

    def preprocess_fn(example):
      source_text_ids = example['source_text_ids'] * example['source_text_mask']
      target_text_ids = example['target_text_ids'] * example['target_text_mask']
      new_example = {
          'source_text_ids':
              source_text_ids,
          'source_position_ids':
              np.arange(encoder_config.source_max_length),
          'target_input_text_ids':
              target_text_ids[:-1],
          'target_output_text_ids':
              target_text_ids[1:],
          'target_input_position_ids':
              np.arange(encoder_config.target_max_length - 1),
      }

      return new_example

    return preprocess_fn

  @staticmethod
  def get_name_to_features(config):
    """Return feature dict for decoding purposes. See BaseTask."""

    encoder_config = config.model_config.encoder_config
    source_max_length = encoder_config.source_max_length
    target_max_length = encoder_config.target_max_length
    name_to_features = {
        'source_text_ids': tf.io.FixedLenFeature(source_max_length, tf.int64),
        'source_text_mask': tf.io.FixedLenFeature(source_max_length, tf.int64),
        'target_text_ids': tf.io.FixedLenFeature(target_max_length, tf.int64),
        'target_text_mask': tf.io.FixedLenFeature(target_max_length, tf.int64),
    }

    return name_to_features

  @staticmethod
  def dummy_input(config):
    """Produces model-specific dummy input batch. See BaseTask."""

    batch_size = config.per_device_batch_size
    source_text_shape = (batch_size,
                         config.model_config.encoder_config.source_max_length)
    target_text_shape = (batch_size,
                         config.model_config.encoder_config.target_max_length -
                         1)
    int_type = jnp.int32

    dummy_input = {
        'source_text_ids': jnp.ones(source_text_shape, int_type),
        'source_position_ids': jnp.ones(source_text_shape, int_type),
        'target_input_text_ids': jnp.ones(target_text_shape, int_type),
        'target_output_text_ids': jnp.ones(target_text_shape, int_type),
        'target_input_position_ids': jnp.ones(target_text_shape, int_type),
    }

    return dummy_input
