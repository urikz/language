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
"""Tests for semantic parsing datasets processing."""

from absl.testing import absltest
from absl.testing import parameterized
from language.mentionmemory.data import semantic_parsing_utils


class SemanticParsingUtilsTest(parameterized.TestCase):
  """Tests for semantic parsing datasets processing."""

  @parameterized.parameters((
      '[IN:GET_CALL_TIME [SL:CONTACT [IN:GET_CONTACT [SL:TYPE_RELATION Mum ] [SL:AAA XXX ] ] ] [SL:DATE_TIME yesterday ] ]',
      'IN:GET_CALL_TIME [SL:CONTACT [IN:GET_CONTACT [SL:AAA SL:TYPE_RELATION]] SL:DATE_TIME]'
  ))
  def test_model_shape(
      self,
      parse_text,
      expected_template,
  ):
    actual_template = semantic_parsing_utils.extract_template(parse_text)
    self.assertMultiLineEqual(actual_template, expected_template)


if __name__ == '__main__':
  absltest.main()
