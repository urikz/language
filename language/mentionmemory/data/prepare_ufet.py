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

from absl import app
from absl import flags
from absl import logging

import bert.tokenization as bert_tokenization
import collections
from language.mentionmemory.utils import tokenization_utils
import multiprocessing
import numpy as np
import spacy
import tensorflow.compat.v2 as tf
import tqdm

FLAGS = flags.FLAGS

flags.DEFINE_string('save_dir', None, 'save directory')
flags.DEFINE_string(
    'data_dir', None, 'Directory where raw data is located. '
    'Download from https://www.cs.utexas.edu/~eunsol/html_pages/open_entity.html'
)
flags.DEFINE_string('vocab_path', None, 'Path to tokenizer vocabulary')
flags.DEFINE_string('label_vocab_path', None, 'Path to tokenizer vocabulary')
flags.DEFINE_integer('max_length', 128, 'max nr of tokens')
flags.DEFINE_integer('max_mentions', 32, 'max nr of mentions')
flags.DEFINE_integer('max_num_labels_per_sample', 32, 'max nr of labels')

flags.DEFINE_integer('nworkers', 1, 'Number of parallel workers.')

FileConfig = collections.namedtuple(
    'FileConfig', ['input_path', 'output_path', 'allow_to_skip'])

FILES = [
    FileConfig('crowd/dev.json', 'valid', False),
    FileConfig('crowd/train.json', 'train_crowd', True),
    FileConfig('distant_supervision/el_train.json', 'train_el', True),
    FileConfig('distant_supervision/headword_train.json', 'train_headword',
               True),
]


class UFETSampleProcessor(object):

  def __init__(
      self,
      vocab_path,
      label_vocab_path,
      max_length,
      max_mentions,
      max_num_labels_per_sample,
      spacy_model_name='en_core_web_lg',
  ):
    self.vocab_path = vocab_path
    self.label_vocab_path = label_vocab_path
    self.max_length = max_length
    self.max_mentions = max_mentions
    self.max_num_labels_per_sample = max_num_labels_per_sample
    self.spacy_model_name = spacy_model_name

  def initializer(self):
    global tokenizer
    tokenizer = bert_tokenization.FullTokenizer(self.vocab_path,
                                                do_lower_case=True)
    global label_vocab
    with open(self.label_vocab_path) as f:
      label_vocab = {line.strip(): index for index, line in enumerate(f)}
    global spacy_model
    spacy_model = spacy.load(self.spacy_model_name)

  def _get_mention_char_spans(self, text, shift):
    parsed_text = spacy_model(text)
    mention_char_spans = []
    for chunk in parsed_text.noun_chunks:
      span_start_char = parsed_text[chunk.start].idx
      span_last_token = parsed_text[chunk.end - 1]
      span_end_char = span_last_token.idx + len(span_last_token.text) - 1
      mention_char_spans.append(
          (span_start_char + shift, span_end_char + shift))
    return mention_char_spans

  def _get_mention_features(self, mention_spans):
    mention_spans = np.array(mention_spans)
    mention_start_positions = mention_spans[:, 0]
    mention_end_positions = mention_spans[:, 1]

    mention_start_positions = mention_start_positions[:self.max_mentions]
    mention_end_positions = mention_end_positions[:self.max_mentions]

    mention_pad_shape = (0, self.max_mentions - len(mention_start_positions))

    mention_mask = np.ones(len(mention_start_positions), dtype=np.int64)
    mention_mask = np.pad(mention_mask, mention_pad_shape, mode='constant')
    mention_start_positions = np.pad(mention_start_positions,
                                     mention_pad_shape,
                                     mode='constant')
    mention_end_positions = np.pad(mention_end_positions,
                                   mention_pad_shape,
                                   mode='constant')

    return {
        'mention_start_positions': mention_start_positions,
        'mention_end_positions': mention_end_positions,
        'mention_mask': mention_mask,
    }

  def _get_target_features(self, labels):
    sample = {}
    target = [label_vocab[l] for l in labels if l in label_vocab]
    sample['target'] = np.zeros(self.max_num_labels_per_sample, dtype=np.int64)
    sample['target'][:len(target)] = target

    sample['target_mask'] = np.zeros(self.max_num_labels_per_sample,
                                     dtype=np.int64)
    sample['target_mask'][:len(target)] = 1
    return sample

  def _convert_to_tf_example(self, sample):
    features = tf.train.Features(
        feature={
            key: tf.train.Feature(int64_list=tf.train.Int64List(value=value))
            for key, value in sample.items()
        })
    return tf.train.Example(features=features)

  def __call__(self, line):
    global tokenizer
    global spacy_model
    sample = json.loads(line)

    left_context = ' '.join(sample['left_context_token'])
    if len(left_context) > 0:
      left_context = left_context + ' '
    mention_char_spans = self._get_mention_char_spans(left_context, 0)

    if len(sample['mention_span']) == 0:
      return None, 0

    mention_target_index = len(mention_char_spans)
    mention_char_spans.append(
        (len(left_context),
         len(left_context) + len(sample['mention_span']) - 1))
    text = left_context + sample['mention_span']

    right_context = ' '.join(sample['left_context_token'])
    if len(right_context) > 0:
      right_context = ' ' + right_context

    mention_char_spans.extend(
        self._get_mention_char_spans(right_context, len(text)))
    text = text + right_context

    assert (text[mention_char_spans[mention_target_index][0]:
                 mention_char_spans[mention_target_index][1] +
                 1] == sample['mention_span'])

    _, text_ids, text_mask, mention_spans, span_indexes = tokenization_utils.tokenize_with_mention_spans(
        tokenizer=tokenizer,
        sentence=text,
        spans=mention_char_spans,
        max_length=self.max_length,
        add_bert_tokens=True,
        allow_truncated_spans=True)

    if mention_target_index not in span_indexes:
      return None, 0
    is_long_example = int(len(span_indexes) < len(mention_char_spans))
    mention_target_index = span_indexes.index(mention_target_index)

    new_sample = {}
    new_sample['text_ids'] = text_ids
    new_sample['text_mask'] = text_mask
    new_sample.update(self._get_mention_features(mention_spans))
    new_sample.update(self._get_target_features(sample['y_str']))
    new_sample['mention_target_indices'] = np.array([mention_target_index])

    if new_sample['target_mask'].sum() == 0:
      return None, 0

    tf_example = self._convert_to_tf_example(new_sample)
    return tf_example.SerializeToString(), is_long_example


def map_maybe_in_parallel(processor, inputs, nworkers: int):
  assert nworkers > 0
  if nworkers == 1:
    processor.initializer()
    for result in map(processor, inputs):
      yield result
  else:
    with multiprocessing.Pool(processes=nworkers,
                              initializer=processor.initializer) as pool:
      for result in pool.imap(processor, inputs):
        yield result


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  processor = UFETSampleProcessor(
      FLAGS.vocab_path,
      FLAGS.label_vocab_path,
      FLAGS.max_length,
      FLAGS.max_mentions,
      FLAGS.max_num_labels_per_sample,
  )

  for file_config in FILES:
    input_full_path = os.path.join(FLAGS.data_dir, file_config.input_path)
    output_full_path = os.path.join(FLAGS.save_dir, file_config.output_path)
    logging.info('Processing %s to %s', input_full_path, output_full_path)

    n_samples = 0
    n_skipped_samples = 0
    n_long_samples = 0
    pbar = tqdm.tqdm(desc=file_config.input_path)
    pbar.set_postfix({
        'n_samples': n_samples,
        'n_skipped_samples': n_skipped_samples,
        'n_long_samples': n_long_samples,
    })

    writer = tf.io.TFRecordWriter(output_full_path)
    with open(input_full_path) as in_file:
      for example_bytes, is_long_sample in map_maybe_in_parallel(
          processor, in_file, nworkers=FLAGS.nworkers):
        if example_bytes is None:
          n_skipped_samples += 1
          assert file_config.allow_to_skip
          continue
        writer.write(example_bytes)
        n_samples += 1
        n_long_samples += is_long_sample
        pbar.update()
        pbar.set_postfix({
            'n_samples': n_samples,
            'n_skipped_samples': n_skipped_samples,
            'n_long_samples': n_long_samples,
        })
    writer.close()
    pbar.close()


if __name__ == '__main__':
  app.run(main)
