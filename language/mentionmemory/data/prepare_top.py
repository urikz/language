# coding=utf-8
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
"""Prepare Ulrta Fine Entity Typing dataset for evaluation."""

import json
import os
import re

from absl import app
from absl import flags
from absl import logging

import sentencepiece.sentencepiece_pb2 as sentencepiece_pb2
import sentencepiece as spm
import collections
import numpy as np
import tensorflow.compat.v2 as tf

FLAGS = flags.FLAGS

flags.DEFINE_string('save_dir', None, 'save directory')
flags.DEFINE_string(
    'data_dir', None, 'Directory where raw data is located. '
    'Download from https://www.cs.utexas.edu/~eunsol/html_pages/open_entity.html'
)
flags.DEFINE_string('vocab_path', None, 'Path to tokenizer vocabulary')
flags.DEFINE_integer('source_max_length', 128, 'max nr of tokens')
flags.DEFINE_integer('target_max_length', 128, 'max nr of tokens')

FileConfig = collections.namedtuple(
    'FileConfig', ['input_path', 'output_path', 'allow_to_skip'])

FILES = [
    FileConfig('train.txt', 'train', True),
    FileConfig('test.txt', 'valid', False),
    # FileConfig('eval.txt', 'test', False),
]

EOS_ID = 1


def _pad(x, max_length):
  assert len(x) < max_length, len(x)
  pad_shape = (0, max_length - len(x))
  return np.pad(x, pad_shape, mode='constant')


def _convert_to_tf_feature(value):
  if isinstance(value, str):
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[value.encode('utf8')]))
  else:
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _convert_to_tf_example(sample):
  features = tf.train.Features(feature={
      key: _convert_to_tf_feature(value) for key, value in sample.items()
  })
  return tf.train.Example(features=features)


class Processor:

  def __init__(
      self,
      vocab_path,
      source_max_length,
      target_max_length,
  ):
    self.vocab_path = vocab_path
    self.source_max_length = source_max_length
    self.target_max_length = target_max_length

    self.tokenizer = spm.SentencePieceProcessor()
    logging.info('Loading sentence piece model from %s', self.vocab_path)
    # Handle cases where SP can't load the file, but gfile can.
    sp_model_ = tf.io.gfile.GFile(self.vocab_path, 'rb').read()
    self.tokenizer.LoadFromSerializedProto(sp_model_)

  def process_text(self, text):
    processed_text = sentencepiece_pb2.SentencePieceText.FromString(
        self.tokenizer.EncodeAsSerializedProto(text))
    token_ids = [piece.id for piece in processed_text.pieces] + [EOS_ID]
    token_ids = np.array(token_ids, dtype=np.int64)
    return token_ids

  def process_target(self, text):

    def process(match):
      return ' '.join(
          [match.group(1),
           match.group(2).lower().replace('_', ' '), '='])

    text = re.sub(r'(\[IN):([A-Z_]*)', process, text)
    text = re.sub(r'(\[SL):([A-Z_]*)', process, text)
    return text

  def process_line(self, line):
    sample_id, _, _, source_text, _, _, target_text, _ = line[:-1].split('\t')

    processed_source_text = self.process_text(source_text)
    if len(processed_source_text) >= self.source_max_length:
      return None, 1, 0

    target_text = self.process_target(target_text)
    processed_target_text = self.process_text(target_text)

    if len(processed_target_text) >= self.target_max_length:
      return None, 0, 1

    processed_source_mask = np.ones(len(processed_source_text), dtype=np.int64)
    processed_source_mask = _pad(processed_source_mask, self.source_max_length)
    processed_target_mask = np.ones(len(processed_target_text), dtype=np.int64)
    processed_target_mask = _pad(processed_target_mask, self.target_max_length)

    processed_source_text = _pad(processed_source_text, self.source_max_length)
    processed_target_text = _pad(processed_target_text, self.target_max_length)

    sample = {
        'source_text_ids': processed_source_text,
        'source_text_mask': processed_source_mask,
        'target_text_ids': processed_target_text,
        'target_text_mask': processed_target_mask,
        'sample_id': sample_id,
    }

    tf_example = _convert_to_tf_example(sample)
    return tf_example, 0, 0

  def process_file(self, input_path, output_path, allow_to_skip):
    n_samples, n_long_source, n_long_target = 0, 0, 0
    writer = tf.io.TFRecordWriter(output_path)
    with tf.io.gfile.GFile(input_path) as in_f:
      for line in in_f:
        tf_example, is_long_source, is_long_target = self.process_line(line)
        n_long_source += is_long_source
        n_long_target += is_long_target
        if tf_example is not None:
          n_samples += 1
          writer.write(tf_example.SerializeToString())
        assert allow_to_skip or n_long_source + n_long_target == 0

    writer.close()
    logging.info('Processed %d samples. Skipped %d and %d', n_samples,
                 n_long_source, n_long_target)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  processor = Processor(
      FLAGS.vocab_path,
      FLAGS.source_max_length,
      FLAGS.target_max_length,
  )

  for file_config in FILES:
    input_full_path = os.path.join(FLAGS.data_dir, file_config.input_path)
    output_full_path = os.path.join(FLAGS.save_dir, file_config.output_path)
    logging.info('Processing %s to %s', input_full_path, output_full_path)
    processor.process_file(input_full_path, output_full_path,
                           file_config.allow_to_skip)


if __name__ == '__main__':
  app.run(main)
