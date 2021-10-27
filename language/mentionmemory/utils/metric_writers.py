# Copyright 2021
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

from clu.metric_writers import interface
from clu.metric_writers.async_writer import AsyncMultiWriter
from clu.metric_writers.async_writer import AsyncWriter
from clu.metric_writers.logging_writer import LoggingWriter
from clu.metric_writers.multi_writer import MultiWriter
from clu.metric_writers.summary_writer import SummaryWriter
import jax
from language.mentionmemory.utils import checkpoint_utils
import neptune
from typing import Any, Optional, Mapping

Array = interface.Array
Scalar = interface.Scalar


class NeptuneWriter(interface.MetricWriter):
  """MetricWriter that writes to Neptune."""

  def __init__(self):
    super().__init__()
    self.experiment = neptune.get_experiment()

  def write_scalars(self, step: int, scalars: Mapping[str, Scalar]):
    for key, value in scalars.items():
      self.experiment.log_metric(key, step, value)

  def write_images(self, step: int, images: Mapping[str, Array]):
    for key, value in images.items():
      self.experiment.log_image(key, step, value)

  def write_texts(self, step: int, texts: Mapping[str, str]):
    for key, value in texts.items():
      self.experiment.log_text(key, step, value)

  def write_histograms(self,
                       step: int,
                       arrays: Mapping[str, Array],
                       num_buckets: Optional[Mapping[str, int]] = None):
    raise NotImplementedError()

  def write_hparams(self, hparams: Mapping[str, Any]):
    hparams_flattened = checkpoint_utils.flatten_nested_dict(hparams, '__')
    for key, value in hparams_flattened.items():
      self.experiment.set_property(key, value)

  def flush(self):
    pass

  def close(self):
    pass


def create_default_writer(logdir: str,
                          exp_id: Optional[str],
                          *,
                          asynchronous: bool = True) -> interface.MetricWriter:
  just_logging = jax.process_index() > 0
  if just_logging:
    if asynchronous:
      return AsyncWriter(LoggingWriter())
    else:
      return LoggingWriter()
  writers = [LoggingWriter(), SummaryWriter(logdir)]
  if exp_id is not None:
    writers.append(NeptuneWriter())
  if asynchronous:
    return AsyncMultiWriter(writers)
  else:
    return MultiWriter(writers)
