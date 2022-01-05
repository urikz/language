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
"""Utility functions for processing of semantic parsing datasets."""

import re


def process_parse_text(parse_text):

  def process(match):
    return ' '.join(
        [match.group(1),
         match.group(2).lower().replace('_', ' '), '='])

  parse_text = re.sub(r'(\[IN):([A-Z_]*)', process, parse_text)
  parse_text = re.sub(r'(\[SL):([A-Z_]*)', process, parse_text)
  return parse_text


def extract_intents(text):
  return tuple(re.findall('IN:([A-Z_]*)', text))


def extract_template(text):
  stack = []
  tokens_iter = iter(text.split())
  while True:
    try:
      token = next(tokens_iter)
    except StopIteration:
      break

    if token.startswith('[IN') or token.startswith('[SL'):
      stack.extend([token[1:], []])
    elif token == ']':
      top_args = stack.pop(-1)
      top_node = stack.pop(-1)
      if len(top_args) > 0:
        top_args = '[' + ' '.join(sorted(top_args)) + ']'
        top_node = top_node + ' ' + top_args
      else:
        top_node = top_node
      if len(stack) > 0:
        stack[-1].append(top_node)
      else:
        stack = top_node
  return stack


def safe_next(it):
  try:
    return next(it)
  except StopIteration:
    return None


def extract_template_per_depth(text, min_depth, max_depth):
  stack = []
  tokens_iter = iter(text.split())
  depth = 0
  while True:
    token = safe_next(tokens_iter)
    if token is None:
      break
    if token.startswith('[IN') or token.startswith('[SL'):
      stack.extend([token[1:], []])
      depth += 1
    elif token == ']':
      if depth >= min_depth:
        top_args = stack.pop(-1)
        top_node = stack.pop(-1)
        if len(top_args) > 0 and depth <= max_depth:
          top_args = '[' + ' '.join(sorted(top_args)) + ']'
          top_node = top_node + ' ' + top_args
        else:
          top_node = top_node
        if len(stack) > 0:
          stack[-1].append(top_node)
        else:
          stack = top_node
      depth -= 1
  return stack


def extract_top_template(text):
  return extract_template_per_depth(text, 0, 1)


def extract_bottom_template(text):
  templates = extract_template_per_depth(text, 3, 100)
  templates = list(filter(None, templates))
  if len(templates) == 1:
    return None
  for template in templates[1:]:
    if isinstance(template, list) and template[0].startswith('IN'):
      return template[0]
  return None
