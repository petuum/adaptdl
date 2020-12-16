# Copyright 2020 Petuum, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import math
import logging

from torchtext.data import BPTTIterator
from torchtext.data.dataset import Dataset
from torchtext.data.batch import Batch

import adaptdl.checkpoint
import adaptdl.collective
import adaptdl.env
from adaptdl.torch.data import AdaptiveDataLoaderMixin

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


class AdaptiveBPTTIterator(BPTTIterator, AdaptiveDataLoaderMixin):
    def __init__(self, dataset, batch_size, bptt_len, **kwargs):
        max_batch_size = kwargs.pop("max_batch_size", None)
        local_bsz_bounds = kwargs.pop("local_bsz_bounds", None)

        BPTTIterator.__init__(self, dataset=dataset, batch_size=batch_size,
                              bptt_len=bptt_len, **kwargs)
        AdaptiveDataLoaderMixin.__init__(self, batch_size)

        self.num_replicas = adaptdl.env.num_replicas()
        self.rank = adaptdl.env.replica_rank()

        if max_batch_size and local_bsz_bounds:
            self._elastic.autoscale_batch_size(max_batch_size,
                                               local_bsz_bounds)

    # The start index changes when there is a rescaling. We recompute a new
    # start index based on how much we covered before the restart
    def _recompute_start(self, prev_curr, prev_end, curr_end):
        if prev_end == 0:
            return prev_curr
        return math.ceil(prev_curr * curr_end / prev_end)

    def __iter__(self):
        with self._elastic.context():
            if self._elastic.skipdone():
                return

            self.batch_size = self._elastic._sync_local_bsz()

            text = self.dataset[0].text
            TEXT = self.dataset.fields['text']
            TEXT.eos_token = None
            text = text + ([TEXT.pad_token] *
                           int(math.ceil(len(text) / self.batch_size) *
                               self.batch_size - len(text)))
            data = TEXT.numericalize(
                [text], device=self.device)
            data = data.view(self.batch_size, -1).t().contiguous()
            dataset = Dataset(examples=self.dataset.examples, fields=[
                ('text', TEXT), ('target', TEXT)])
            end = data.size(0)  # current length of dataset

            # Change in current batch size changes the dimensions of dataset
            # which changes the starting position in the reshaped dataset. The
            # local batch size is also a function of number of replicas, so
            # when we rescale we need to recalculate where to start the
            # iterations from for the new batch size.
            self._elastic.current_index = \
                self._recompute_start(self._elastic.current_index,
                                      self._elastic.end_index, end)
            self._elastic.end_index = end

            # Every replica reads data strided by bptt_len
            start = self._elastic.current_index + (self.bptt_len * self.rank)
            step = self.bptt_len * self.num_replicas

            # The starting index of the highest rank
            highest_start = self._elastic.current_index + \
                (self.bptt_len * (self.num_replicas - 1))

            # Number of steps we will take on the highest rank. We limit
            # iterations on all replicas by this number to prevent asymmetric
            # reduce ops which would result in a deadlock.
            min_steps_in_epoch = max(math.ceil((end - highest_start) / step), 0)  # noqa: E501
            self.iterations = 0
            while True:
                for i in range(start, end, step):
                    self.iterations += 1
                    # Make sure that _elastic.profile is called equal number of
                    # times on all replicas
                    if self.iterations > min_steps_in_epoch:
                        break
                    with self._elastic.profile(self.training and i > 0):
                        seq_len = min(self.bptt_len, data.size(0) - i - 1)
                        assert seq_len > 0
                        batch_text = data[i:i + seq_len]
                        batch_target = data[i + 1:i + 1 + seq_len]
                        if TEXT.batch_first:
                            batch_text = batch_text.t().contiguous()
                            batch_target = batch_target.t().contiguous()
                        yield Batch.fromvars(
                            dataset, self.batch_size,
                            text=batch_text,
                            target=batch_target)
                        self._elastic.current_index += step

                if not self.repeat:
                    break
