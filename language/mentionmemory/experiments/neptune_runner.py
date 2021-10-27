# coding=utf-8
# Copyright 2018 Yury Zemlyanskiy.
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
"""Run experiments using Neptune."""

import json
import os
import socket
import tempfile

from absl import app
from absl import flags
from absl import logging
import os
import jax
from language.mentionmemory.training.trainer import train
import ml_collections
import tensorflow.compat.v1 as tf
import neptune

FLAGS = flags.FLAGS

# Training config.
flags.DEFINE_string(
    "config_file", None,
    "Path to file that contains JSON serialized model configuration. If this"
    "is specified, ignores 'config' parameter.")
flags.DEFINE_string("exp_name", None, "Experiment name")
flags.DEFINE_string(
    "model_base_dir", None,
    "The output directory where the model checkpoints will be written.")
flags.DEFINE_boolean(
    "debug", False, "Whether to run training without connecting to Neptune.ai")

# Hyper parameters
flags.DEFINE_float("learning_rate", None, "Learning rate")
flags.DEFINE_integer("per_device_batch_size", None, "Per device batch size")
flags.DEFINE_integer("num_train_steps", None, "Number of training steps")
flags.DEFINE_integer("warmup_steps", None, "Number of warmup training steps")


def validate_config(config):
  """Verify all paths in the config exist."""
  for value in config.values():
    if isinstance(value, str) and value.startswith('/'):
      if len(tf.io.gfile.glob(value)) == 0:
        raise ValueError('Invalid path %s' % value)
    elif isinstance(value, dict):
      validate_config(value)


def get_and_validate_config():
  with tf.io.gfile.GFile(FLAGS.config_file, "r") as reader:
    config = json.load(reader)

  if FLAGS.learning_rate is not None:
    config["learning_rate"] = FLAGS.learning_rate
  if FLAGS.per_device_batch_size is not None:
    config["per_device_batch_size"] = FLAGS.per_device_batch_size
  if FLAGS.num_train_steps is not None:
    config["num_train_steps"] = FLAGS.num_train_steps
  if FLAGS.warmup_steps is not None:
    config["warmup_steps"] = FLAGS.warmup_steps

  validate_config(config)

  return config


def get_tags(config):
  TAGS = {
      'task': 'task_name',
      'lr': 'learning_rate',
  }

  def get_tag(prefix, name):
    if name in config:
      return prefix + ':' + str(config[name])
    else:
      return None

  tags = [get_tag(k, v) for (k, v) in TAGS.items()]
  tags = list(filter(None, tags))
  return tags


def get_neptune_experiment(config):
  if not FLAGS.debug:
    neptune.init()
    return neptune.create_experiment(name=FLAGS.exp_name,
                                     params=config,
                                     tags=get_tags(config),
                                     git_info=neptune.utils.get_git_info(),
                                     hostname=socket.gethostname())
  else:
    return None


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  # Hide any GPUs form TensorFlow. Otherwise TF might reserve memory and make
  # it unavailable to JAX.
  tf.config.experimental.set_visible_devices([], "GPU")

  logging.info("JAX process: %d / %d", jax.process_index(), jax.process_count())
  logging.info("JAX local devices: %r", jax.local_devices())

  config = get_and_validate_config()
  experiment = get_neptune_experiment(config)

  if experiment is not None:
    exp_id = experiment.id
    model_dir = os.path.join(FLAGS.model_base_dir, exp_id)
  else:
    exp_id = None
    model_dir = tempfile.mkdtemp(prefix='debug_', dir=FLAGS.model_base_dir)

  config['exp_name'] = FLAGS.exp_name
  config['exp_id'] = exp_id
  config['model_dir'] = model_dir
  tf.io.gfile.makedirs(model_dir)

  train(ml_collections.ConfigDict(config))

  if experiment is not None:
    experiment.stop()


if __name__ == "__main__":
  flags.mark_flags_as_required(['config_file', 'exp_name', 'model_base_dir'])
  app.run(main)
