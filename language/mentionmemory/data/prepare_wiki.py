import dataclasses
import logging
import numpy as np
import time
import math
from typing import Optional

import json
import os

from absl import app
from absl import flags
from absl import logging

import bert.tokenization as bert_tokenization
from language.mentionmemory.utils import tokenization_utils
import multiprocessing
import numpy as np
import spacy
import tensorflow.compat.v2 as tf
import tqdm

import codecs
from collections import Counter
import dataclasses
import glob
import json
import numpy as np
import os
import spacy
import stat
import sys
import tqdm
from typing import Any, List
from ncls import NCLS
import urllib.parse

FLAGS = flags.FLAGS

flags.DEFINE_string('output', None, 'Output path.')
flags.DEFINE_string('data', None, 'Input files pattern.')
flags.DEFINE_string('vocab_path', None, 'Path to tokenizer vocabulary')
flags.DEFINE_string('target_entity', None,
                    'Keep only passages that contain this entity.')
flags.DEFINE_integer('max_length', 128, 'max nr of tokens')
flags.DEFINE_integer('nworkers', 1, 'Number of parallel workers.')
flags.DEFINE_integer('min_words_per_doc', None,
                     'Minimal number of words per document.')

TRAINING_TQDM_BAD_FORMAT = (
    '{l_bar}{bar}| '
    '{n_fmt}/{total_fmt} [{elapsed}<{remaining} {postfix}]')


def strip_wiki_domain(url):
  assert url.startswith('http://en.wikipedia.org/wiki/')
  url = url[len('http://en.wikipedia.org/wiki/'):]
  return urllib.parse.unquote(url)


@dataclasses.dataclass
class AnnotationFilterStats:
  num_annotations: int = 0
  num_human_annotations: int = 0
  num_el_annotations: int = 0
  num_noun_annotations: int = 0
  num_filtered_xao: int = 0
  num_filtered_by_candidate_set: int = 0
  num_filtered_by_human_annotations: int = 0
  num_filtered_by_self_overlaps: int = 0
  num_filtered_by_crossing_block_boundaries: int = 0
  num_filtered_by_entity_vocab: int = 0

  def add(self, other_filter_stats):
    for field in dataclasses.fields(other_filter_stats):
      old_value = getattr(self, field.name)
      add_value = getattr(other_filter_stats, field.name)
      new_value = old_value + add_value
      setattr(self, field.name, new_value)

  def get_num_annotations(self):
    return self.num_annotations

  def get_num_filtered_annotations(self):
    return (
        self.num_filtered_xao +
        # `num_filtered_by_candidate_set` were not really filtered.
        # We have just modified their entity ID to [UNK].
        # self.num_filtered_by_candidate_set +
        self.num_filtered_by_human_annotations +
        self.num_filtered_by_self_overlaps +
        self.num_filtered_by_crossing_block_boundaries +
        self.num_filtered_by_entity_vocab)

  def dump(self, path):
    with open(path, 'w') as f:
      json.dump(dataclasses.asdict(self), f, indent=4)


def get_start_end(mention):
  return mention['offset'], mention['offset'] + len(mention['surface_form'])


def intersect(s1, e1, s2, e2):
  return (s1 <= s2 and s2 < e1) or (s2 <= s1 and s1 < e2)


def get_intervals(annotations):
  start, end = [], []
  for annotation in annotations:
    s, e = get_start_end(annotation)
    start.append(s)
    end.append(e)
  return np.array(
      start, dtype=np.int64), np.array(
          end, dtype=np.int64), np.arange(len(annotations))


def load_vocab(path):
  assert os.path.exists(path)
  entities = {}
  with codecs.open(path, 'r', 'utf-8') as f:
    for index, line in enumerate(f):
      columns = line[:-1].split(' ')
      assert len(columns) <= 2
      entity = columns[0]
      entities[entity] = int(index)
  return entities


@dataclasses.dataclass
class Annotation(object):
  start: int
  end: int
  text: str
  name: Optional[str] = None
  start_word: Optional[int] = None
  end_word: Optional[int] = None


@dataclasses.dataclass
class AnnotatedArticle(object):
  text: str
  annotations: List[Annotation]

  @staticmethod
  def build(text, annotations, shall_normalize_whitespaces=True):
    text = text.replace('\xa0', ' ').replace('\n', ' ')
    sorted_annotations = sorted(annotations, key=lambda a: (a.start, a.end))
    for a in sorted_annotations:
      a.text = a.text.replace('\xa0', ' ').replace('\n', ' ')
    article = AnnotatedArticle(text, sorted_annotations)
    article.validate()
    if shall_normalize_whitespaces:
      article.strip_whitespaces()
      article.strip_double_whitespaces()
    # article.add_margin_to_annotations()
    return article

  def validate(self):
    for a in self.annotations:
      assert a.end <= len(self.text)
      assert self.text[a.start:a.end] == a.text

  def __len__(self):
    return len(self.text)

  def add_margin_to_annotations(self):
    for i in range(len(self.annotations)):
      start = self.annotations[i].start
      if start > 0 and self.text[start - 1] != ' ':
        self.text = self.text[:start] + ' ' + self.text[start:]
        for j in range(i, len(self.annotations)):
          self.annotations[j].start += 1
          self.annotations[j].end += 1

      end = self.annotations[i].end
      if end < len(self.text) and self.text[end] != ' ':
        self.text = self.text[:end] + ' ' + self.text[end:]
        for j in range(i + 1, len(self.annotations)):
          self.annotations[j].start += 1
          self.annotations[j].end += 1
    self.validate()

  def strip_whitespaces(self):
    self.text = self.text.rstrip()
    num_whitespaces = 0
    while num_whitespaces < len(
        self.text) and self.text[num_whitespaces] == ' ':
      num_whitespaces += 1
    assert num_whitespaces < len(self.text)
    if num_whitespaces > 0:
      self.text = self.text[num_whitespaces:]
      for i in range(len(self.annotations)):
        self.annotations[i].start -= num_whitespaces
        self.annotations[i].end -= num_whitespaces
    self.validate()

  def strip_double_whitespaces(self):
    idx = self.text.find('  ')
    while idx >= 0:
      self.text = self.text[:idx] + self.text[idx + 1:]
      for i in range(len(self.annotations)):
        assert self.annotations[i].start != idx
        if self.annotations[i].end > idx:
          self.annotations[i].end -= 1
        if self.annotations[i].start > idx:
          self.annotations[i].start -= 1
      idx = self.text.find('  ')
    for i in range(len(self.annotations)):
      idx = self.annotations[i].text.find('  ')
      while idx >= 0:
        self.annotations[i].text = (
            self.annotations[i].text[:idx] + self.annotations[i].text[idx + 1:])
        idx = self.annotations[i].text.find('  ')
    self.validate()

  def set_word_based_offsets(self):
    word_index = 0
    current_annotation_index, next_annotation_index = None, 0
    assert self.text[0] != ' ', self.text
    for i in range(len(self.text)):
      if next_annotation_index >= len(self.annotations):
        break
      if self.text[i] == ' ':
        word_index += 1
      else:
        if current_annotation_index is None:
          if i == self.annotations[next_annotation_index].start:
            current_annotation_index = next_annotation_index
            self.annotations[current_annotation_index].start_word = word_index

        if current_annotation_index is not None:
          if i == self.annotations[current_annotation_index].end - 1:
            self.annotations[current_annotation_index].end_word = word_index + 1
            next_annotation_index += 1
            current_annotation_index = None

    words = self.text.split(' ')
    for a in self.annotations:
      assert ' '.join(words[a.start_word:a.end_word]) == a.text


def _get_mention_features(mention_spans, mention_ids, max_length,
                          length_per_example):
  new_mention_spans, new_mention_ids = [], []
  for index in range(len(mention_spans)):
    mention_span_start, mention_span_end = mention_spans[index]
    mention_within_example = (
        mention_span_start // length_per_example == mention_span_end //
        length_per_example)
    if mention_within_example:
      new_mention_spans.append(mention_spans[index])
      new_mention_ids.append(mention_ids[index])
  mention_spans = new_mention_spans
  mention_ids = new_mention_ids

  mention_spans = np.array(mention_spans)
  dense_span_starts = np.zeros(max_length, dtype=np.int64)
  dense_span_starts[mention_spans[:, 0]] = 1
  dense_span_ends = np.zeros(max_length, dtype=np.int64)
  dense_span_ends[mention_spans[:, 1]] = 1

  dense_mention_mask = np.zeros(max_length, dtype=np.int64)
  dense_mention_ids = np.zeros(max_length, dtype=np.int64)

  for index in range(len(mention_spans)):
    mention_span_start, mention_span_end = mention_spans[index]
    if mention_ids[index] > 0:
      dense_mention_mask[mention_span_start:mention_span_end] = 1
      dense_mention_ids[mention_span_start:mention_span_end] = mention_ids[
          index]

  return {
      'dense_span_starts': dense_span_starts,
      'dense_span_ends': dense_span_ends,
      'dense_mention_mask': dense_mention_mask,
      'dense_mention_ids': dense_mention_ids,
  }


def _convert_to_tf_examples(sample, max_length):
  total_length = len(sample['text_ids'])
  assert total_length % max_length == 0
  tf_examples = []
  for index in range(total_length // max_length):
    start = index * max_length
    end = (index + 1) * max_length

    features = {}
    for key, value in sample.items():
      if key == 'text_ids':
        feature = np.concatenate([[101], value[start:end], [0]])
        fisrt_non_zero = np.nonzero(feature == 0)[0][0]
        feature[fisrt_non_zero] = 102
      elif key == 'text_mask':
        feature = np.concatenate([[1], value[start:end], [0]])
        fisrt_non_zero = np.nonzero(feature == 0)[0][0] = 1
      else:
        feature = np.concatenate([[0], value[start:end], [0]])
      features[key] = feature

    features = tf.train.Features(
        feature={
            key: tf.train.Feature(int64_list=tf.train.Int64List(value=value))
            for key, value in features.items()
        })
    tf_example = tf.train.Example(features=features)
    tf_examples.append(tf_example.SerializeToString())
  return tf_examples


class WikiProcessor(object):

  def __init__(
      self,
      vocab_path,
      entity_vocab,
      max_length,
      build_entity_vocab_mode,
      min_words_per_doc,
      target_entity,
  ):
    self.vocab_path = vocab_path
    self.entity_vocab = entity_vocab
    self.max_length = max_length
    self.build_entity_vocab_mode = build_entity_vocab_mode
    self.min_words_per_doc = min_words_per_doc
    self.target_entity = target_entity

  def initializer(self):
    global tokenizer
    tokenizer = bert_tokenization.FullTokenizer(
        self.vocab_path, do_lower_case=True)
    global spacy_model
    spacy_model = spacy.load('en_core_web_lg')

    global entities
    if not self.build_entity_vocab_mode:
      entities = load_vocab(self.entity_vocab)

  def fix_annotations(self, annotations, text_len):
    new_annotations = []
    num_filtered = 0
    for annotation in annotations:
      if u'\xa0' in annotation['surface_form'] or '\n' in annotation[
          'surface_form'] or '\n' in annotation['uri']:
        num_filtered += 1
        continue
      if annotation['offset'] + len(annotation['surface_form']) > text_len:
        num_filtered += 1
        continue
      while annotation['surface_form'].startswith(' '):
        annotation['surface_form'] = annotation['surface_form'][1:]
        annotation['offset'] += 1
      while annotation['surface_form'].endswith(' '):
        annotation['surface_form'] = annotation['surface_form'][:-1]
      assert len(annotation['surface_form']) > 0
      new_annotations.append(annotation)
    return new_annotations, num_filtered

  def filter_by_candidate_set(self, article, annotations):
    main_entity = strip_wiki_domain(article['url'])
    original_entities_set = set(
        [original_entity['uri'] for original_entity in article['annotations']])
    original_entities_set.add(main_entity)

    new_annotations = []
    num_filtered = 0
    for annotation in annotations:
      if annotation['uri'] in original_entities_set:
        new_annotations.append(annotation)
      else:
        annotation['uri'] = '[UNK]'
        new_annotations.append(annotation)
        num_filtered += 1
    return new_annotations, num_filtered

  def filter_by_human_annotations(self, existing_annotations, annotations):
    ncls = NCLS(*get_intervals(existing_annotations))
    new_annotations = []
    num_filtered = 0
    for annotation in annotations:
      entity_start, entity_end = get_start_end(annotation)
      matched_human_annotation = list(
          ncls.find_overlap(entity_start, entity_end))
      if len(matched_human_annotation) == 0:
        new_annotations.append(annotation)
      else:
        human_annotation = existing_annotations[matched_human_annotation[0][2]]
        human_annotation_start, human_annotation_end = get_start_end(
            human_annotation)
        assert intersect(human_annotation_start, human_annotation_end,
                         entity_start, entity_end)
        num_filtered += 1
    assert len(new_annotations) + num_filtered == len(annotations)
    return new_annotations, num_filtered

  def filter_by_self_overlaps(self, annotations):
    new_annotations = []
    num_filtered = 0
    start, end, index = get_intervals(annotations)
    events = sorted([(x[0], 1, x[1]) for x in zip(start, index)] +
                    [(x[0], 0, x[1]) for x in zip(end, index)])
    current_open_interval = None
    for event in events:
      if event[1] == 0:  # close interval
        if event[2] == current_open_interval:
          current_open_interval = None
      else:
        assert event[1] == 1  # open interval
        if current_open_interval is None:
          current_open_interval = event[2]
          new_annotations.append(annotations[current_open_interval])
        else:
          num_filtered += 1
    assert len(new_annotations) + num_filtered == len(annotations)
    # TODO: Only for debugging
    # for i in range(len(new_annotations)):
    #     start_i, end_i = get_start_end(new_annotations[i])
    #     for j in range(i + 1, len(new_annotations)):
    #         start_j, end_j = get_start_end(new_annotations[j])
    #         assert not intersect(start_i, end_i, start_j, end_j)
    return new_annotations, num_filtered

  def filter_by_unk_annotations(self, annotations):
    global entities
    assert entities is not None
    new_annotations = []
    num_filtered = 0
    for annotation in annotations:
      if annotation['uri'] in entities and annotation['uri'] != '[UNK]':
        new_annotations.append(annotation)
      else:
        num_filtered += 1
    return new_annotations, num_filtered

  def annotate_nouns(self, text):
    global spacy_model
    doc = spacy_model(text)
    annotations = []
    for np in doc.noun_chunks:
      offset = np.start_char
      surface_form = np.text
      assert text[offset:offset + len(surface_form)] == surface_form
      annotations.append({
          'offset': offset,
          'surface_form': surface_form,
          'uri': '[NOUN]'
      })
    return annotations

  def tokenize(self, article, max_length):
    global tokenizer
    _, text_ids, text_mask, mention_spans, span_indices = tokenization_utils.tokenize_with_mention_spans(
        tokenizer=tokenizer,
        sentence=article.text,
        spans=[(a.start, a.end - 1) for a in article.annotations],
        max_length=max_length,
        add_bert_tokens=False,
        allow_truncated_spans=True,
        encode_utf8=False)
    return text_ids, text_mask, mention_spans, span_indices

  def __call__(self, path):
    global tokenizer
    global entities
    filter_stats = AnnotationFilterStats()

    if self.build_entity_vocab_mode:
      annotation_entities = Counter()
    else:
      tf_examples = []
      pad_entity_id = entities['[PAD]']
      assert pad_entity_id == 0
      unk_entity_id = entities['[UNK]']
      assert unk_entity_id == 1
      noun_entity_id = entities['[NOUN]']
      assert noun_entity_id == 2

    with codecs.open(path, 'r', 'utf8') as f:
      for line in f:
        article = json.loads(line[:-1])
        main_entity = strip_wiki_domain(article['url'])

        if (self.min_words_per_doc is not None and
            len(article['text'].split()) < self.min_words_per_doc):
          continue

        human_annotations = article['annotations']
        human_annotations, num_filtered_xao = self.fix_annotations(
            human_annotations,
            len(article['text']),
        )
        filter_stats.num_filtered_xao += num_filtered_xao
        article['annotations'] = human_annotations

        auto_annotations = article['el']
        auto_annotations, num_filtered_xao = self.fix_annotations(
            auto_annotations,
            len(article['text']),
        )
        filter_stats.num_filtered_xao += num_filtered_xao
        text = article['text'].replace('\xa0', ' ').replace('\n', ' ')

        auto_annotations, num_filtered_by_candidate_set = self.filter_by_candidate_set(
            article, auto_annotations)
        filter_stats.num_filtered_by_candidate_set += num_filtered_by_candidate_set

        auto_annotations, num_filtered_by_human_annotations = self.filter_by_human_annotations(
            human_annotations,
            auto_annotations,
        )
        filter_stats.num_filtered_by_human_annotations += num_filtered_by_human_annotations

        auto_annotations, num_filtered_by_self_overlaps = self.filter_by_self_overlaps(
            auto_annotations)
        filter_stats.num_filtered_by_self_overlaps += num_filtered_by_self_overlaps

        if self.target_entity is not None:
          if all([a['uri'] != self.target_entity for a in human_annotations]):
            continue

        noun_annotations = self.annotate_nouns(text)

        noun_annotations, num_filtered_xao = self.fix_annotations(
            noun_annotations,
            len(article['text']),
        )
        filter_stats.num_filtered_xao += num_filtered_xao

        noun_annotations, num_filtered_by_human_annotations = self.filter_by_human_annotations(
            human_annotations + auto_annotations, noun_annotations)
        filter_stats.num_filtered_by_human_annotations += num_filtered_by_human_annotations

        noun_annotations, num_filtered_by_self_overlaps = self.filter_by_self_overlaps(
            noun_annotations)
        filter_stats.num_filtered_by_self_overlaps += num_filtered_by_self_overlaps

        filter_stats.num_human_annotations += len(human_annotations)
        filter_stats.num_el_annotations += len(auto_annotations)
        filter_stats.num_noun_annotations += len(noun_annotations)

        annotations = human_annotations + auto_annotations + noun_annotations
        filter_stats.num_annotations += len(annotations)

        article = AnnotatedArticle.build(
            text=text,
            annotations=[
                Annotation(
                    start=a['offset'],
                    end=a['offset'] + len(a['surface_form']),
                    text=a['surface_form'],
                    name=a['uri'],
                ) for a in annotations
            ],
        )

        if self.build_entity_vocab_mode:
          annotation_entities.update([a.name for a in article.annotations])
        else:
          approximate_max_length = 5 * len(article.text.split(' '))
          text_mask = self.tokenize(article, approximate_max_length)[1]
          max_length = math.ceil(sum(text_mask) / (self.max_length - 2)) * (
              self.max_length - 2)
          text_ids, text_mask, mention_spans, span_indices = self.tokenize(
              article, max_length)

          all_entity_ids = [
              entities[a.name] if entities[a.name] > 2 else 0
              for a in article.annotations
          ]
          all_entity_ids = [all_entity_ids[x] for x in span_indices]

          new_sample = {}
          new_sample['text_ids'] = text_ids
          new_sample['text_mask'] = text_mask
          new_sample.update(
              _get_mention_features(mention_spans, all_entity_ids, max_length,
                                    self.max_length - 2))
          current_tf_examples = _convert_to_tf_examples(new_sample,
                                                        self.max_length - 2)
          tf_examples.extend(current_tf_examples)

    if self.build_entity_vocab_mode:
      output = annotation_entities
    else:
      output = tf_examples

    return (output, filter_stats)


def map_maybe_in_parallel(processor, inputs: List[Any], nworkers: int):
  assert nworkers > 0
  if nworkers == 1:
    processor.initializer()
    for result in map(processor, inputs):
      yield result
  else:
    with multiprocessing.Pool(
        processes=nworkers, initializer=processor.initializer) as pool:
      for result in pool.imap(processor, inputs):
        yield result


def main(args):
  input_files = sorted([
      path for data in FLAGS.data.split(',')
      for path in glob.glob(os.path.expanduser(data.strip("'")))
  ])
  print('-- Found %d files' % len(input_files))

  entity_vocab_path = os.path.join(
      os.path.split(FLAGS.output)[0], 'entity.vocab.txt')
  build_entity_vocab_mode = not os.path.exists(entity_vocab_path)
  print('-- Build entity vocab mode: %s' %
        ('ON' if build_entity_vocab_mode else 'OFF'))

  processor = WikiProcessor(
      vocab_path=FLAGS.vocab_path,
      max_length=FLAGS.max_length,
      entity_vocab=entity_vocab_path if not build_entity_vocab_mode else None,
      build_entity_vocab_mode=build_entity_vocab_mode,
      min_words_per_doc=FLAGS.min_words_per_doc,
      target_entity=FLAGS.target_entity,
  )
  filter_stats = AnnotationFilterStats()
  total_length = 0

  pbar = tqdm.tqdm(
      total=len(input_files),
      desc='Processing Wiki',
      bar_format=TRAINING_TQDM_BAD_FORMAT,
  )
  pbar.set_postfix({
      'tokens': total_length,
      'ann': 0,
      'f_ann': 0,
  })

  tf_example_counter = 0
  if build_entity_vocab_mode:
    entities = Counter()
  else:
    writer = tf.io.TFRecordWriter(FLAGS.output)

  for (
      output,
      _filter_stats,
  ) in map_maybe_in_parallel(
      processor, input_files, nworkers=FLAGS.nworkers):
    if build_entity_vocab_mode:
      _entities = output
      entities.update(_entities)
    else:
      for tf_example in output:
        tf_example_counter += 1
        writer.write(tf_example)

    filter_stats.add(_filter_stats)
    pbar.set_postfix({
        'tf_examples': tf_example_counter,
        'tokens': total_length,
        'ann': filter_stats.get_num_annotations(),
        'f_ann': filter_stats.get_num_filtered_annotations(),
    })
    pbar.update()
  pbar.close()

  if build_entity_vocab_mode:
    counter = 0
    special_entities = ['[PAD]', '[UNK]', '[NOUN]']
    special_entities_set = set(special_entities)
    with codecs.open(entity_vocab_path, 'w', 'utf8') as f:
      for entity in special_entities:
        f.write('%s 0\n' % entity)
      for entity, count in entities.most_common():
        if entity in special_entities_set:
          continue
        counter += 1
        f.write('%s %d\n' % (entity, count))
    print('-- Successfully saved %d entities (out of %d) to %s' % (
        counter,
        len(entities),
        entity_vocab_path,
    ))
  else:
    writer.close()
    print('-- Successfully saved %d tf examples %s' % (
        tf_example_counter,
        FLAGS.output,
    ))


if __name__ == '__main__':
  app.run(main)
