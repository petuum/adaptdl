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

import collections
import math
import adaptdl.checkpoint
import adaptdl.collective
import adaptdl.env
from adaptdl.torch._metrics import get_goodput_fn
import adaptdl.torch.data
from adaptdl.torch.scaling_rules import ScalingRuleBase

class AdaptiveDLContext(object):
    """
    This class provides context tool to get AdaptDL-suggest parameters,
    such as batch_size, accum_steps and lr_scale.
    """

    def __init__(self, batch_size):
        self._elastic = adaptdl.torch.data.AdaptiveDataLoaderHelper(batch_size)
        # Autoscale batch size fields.
        self._speedup_threshold = 1.05
        self.adapt_batch_size = None
        self.adapt_accum_steps = None
        self.adapt_lr_scale = None

    def autoscale_batch_size(self, max_batch_size, local_bsz_bounds=None,
                             gradient_accumulation=False):
        self._elastic.autoscale_batch_size(max_batch_size, local_bsz_bounds,
                                           gradient_accumulation)

    def get_batch_size(self):
        _, self.adapt_batch_size, _ = self._sync_local_bsz()
        return self.adapt_batch_size

    def get_accum_steps(self):
        _, _, self.adapt_accum_steps = self._sync_local_bsz()
        return self.adapt_accum_steps

    def get_lr_scale(self):
        self.adapt_lr_scale = ScalingRuleBase._get_adapt_lr_scale()
        return float(self.adapt_lr_scale)

    def _sync_local_bsz(self):
        goodput_fn = get_goodput_fn()
        if self._elastic.max_batch_size is None or goodput_fn is None:
            # No autoscale batch size, just divide batch size evenly.
            self._elastic._state.current_local_bsz = math.ceil(
                self._elastic.batch_size / adaptdl.env.num_replicas())
            self._elastic._state.accumulation_steps = 0
        elif not self._elastic._state.current_local_bsz:
            # if init, use the batch size suggested
            _, atomic_bsz, accum_steps = goodput_fn.optimize(
                adaptdl.env.num_nodes(), adaptdl.env.num_replicas(),
                max_batch_size=self._elastic._max_batch_size,
                atomic_bsz_range=self._elastic._local_bsz_bounds,
                accumulation=self._elastic._gradient_accumulation)
            self._elastic._state.current_local_bsz = atomic_bsz
            self._elastic._state.accumulation_steps = accum_steps
        else:
            # if not first time, we check against the relative speedup
            suggest_goodput, atomic_bsz, accum_steps = goodput_fn.optimize(
                adaptdl.env.num_nodes(), adaptdl.env.num_replicas(),
                max_batch_size=self._elastic._max_batch_size,
                atomic_bsz_range=self._elastic._local_bsz_bounds,
                accumulation=self._elastic._gradient_accumulation)
            # get current goodput
            current_goodput = goodput_fn(
                adaptdl.env.num_nodes(), adaptdl.env.num_replicas(),
                self._elastic.current_local_bsz, self._elastic.accumulation_steps)
            # use only if speedup is significant
            speedup = suggest_goodput / max(current_goodput, 1e-8)
            if speedup > self._speedup_threshold:
                self._elastic._state.current_local_bsz = atomic_bsz
                self._elastic._state.accumulation_steps = accum_steps
        self._elastic._state.current_local_bsz, self._elastic._state.accumulation_steps = \
            adaptdl.collective.broadcast((self._elastic._state.current_local_bsz,
                                          self._elastic._state.accumulation_steps))
        return self._elastic.current_local_bsz, self._elastic._state.current_local_bsz, self._elastic._state.accumulation_steps

    @property
    def training(self):
        return self._elastic.training

    def to_tensorboard(self, writer, global_step, tag_prefix=""):
        self._elastic.to_tensorboard(writer, global_step, tag_prefix)
    # to_tensorboard.__doc__ = adaptdl.torch.data.AdaptiveDataLoaderHelper.to_tensorboard.__doc__