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
import adaptdl.torch.data as data
import numpy as np

class Context(object):
    """
    This class provides context tool to get AdaptDL-suggest parameters,
    such as batch_size, accum_steps and lr_scale.
    """

    def __init__(self, batch_size=32):
        # Autoscale batch size fields.
        self._speedup_threshold = 1.05
        self.adapt_batch_size = None
        self.adapt_accum_steps = None
        self.adapt_lr_scale = None

        self._max_batch_size = None
        self._local_bsz_bounds = None
        # Create and load state.
        self._state = data._AdaptiveDataLoaderState()
        adaptdl.checkpoint.load_state(self._state)
        self.batch_size = batch_size
        # self.state_batch_size = 1
        self._gradient_accumulation = False

    def get_batch_size(self):
        self.adapt_batch_size, _ = self._get_local_bsz()
        return self.adapt_batch_size

    def get_accum_steps(self):
        _, self.adapt_accum_steps = self._get_local_bsz()
        return self.adapt_accum_steps

    @staticmethod
    def get_lr_scale(scale_lr, gns, optimizer):
        scale = gns.accum_scale * gns.accum_count
        initial_lr = [pg["lr"] for pg in optimizer.param_groups]
        return scale, np.multiply(scale_lr(scale), initial_lr), initial_lr

    def _get_local_bsz(self):
        goodput_fn = get_goodput_fn()
        if self.max_batch_size is None or goodput_fn is None:
            # No autoscale batch size, just divide batch size evenly.
            self._state.current_local_bsz = math.ceil(
                self.batch_size / adaptdl.env.num_replicas())
            self._state.accumulation_steps = 0
        elif not self._state.current_local_bsz:
            # if init, use the batch size suggested
            _, atomic_bsz, accum_steps = goodput_fn.optimize(
                adaptdl.env.num_nodes(), adaptdl.env.num_replicas(),
                max_batch_size=self._max_batch_size,
                atomic_bsz_range=self._local_bsz_bounds,
                accumulation=self._gradient_accumulation)
            self._state.current_local_bsz = atomic_bsz
            self._state.accumulation_steps = accum_steps
        else:
            # if not first time, we check against the relative speedup
            suggest_goodput, atomic_bsz, accum_steps = goodput_fn.optimize(
                adaptdl.env.num_nodes(), adaptdl.env.num_replicas(),
                max_batch_size=self._max_batch_size,
                atomic_bsz_range=self._local_bsz_bounds,
                accumulation=self._gradient_accumulation)
            # get current goodput
            current_goodput = goodput_fn(
                adaptdl.env.num_nodes(), adaptdl.env.num_replicas(),
                self.current_local_bsz, self.accumulation_steps)
            # use only if speedup is significant
            speedup = suggest_goodput / max(current_goodput, 1e-8)
            if speedup > self._speedup_threshold:
                self._state.current_local_bsz = atomic_bsz
                self._state.accumulation_steps = accum_steps
        return self._state.current_local_bsz, self._state.accumulation_steps

    @property
    def max_batch_size(self):
        """
        The maximum total batch size allowed for adaptive batch size. ``None``
        if adaptive batch size is disabled.
        """
        return self._max_batch_size

    @property
    def local_bsz_bounds(self):
        """
        The local batch size bounds on each replica. A pair of integers,
        (min_local_bsz, max_local_bsz).
        """
        return self._local_bsz_bounds

    @property
    def current_local_bsz(self):
        """
        The current logical local batch size used by the dataloader.
        The batch size returned by the dataloader may be smaller if
        gradient accumulation is used
        """
        return self._state.current_local_bsz

    @property
    def accumulation_steps(self):
        """
        The number of batches returned by the dataloader before a
        step is taken.
        """
        return self._state.accumulation_steps

    def autoscale_batch_size(self, max_batch_size, local_bsz_bounds=None,
                             gradient_accumulation=False):
        """
        Enables adaptive batch size. Should be invoked once after the data
        loader object is created.

        Arguments:
            max_batch_size (int): Maximum total batch size allowed.
            local_bsz_bounds (tuple): A pair of (min_local_bsz, max_local_bsz),
                the min and max local batch sizes allowed on each replica.

        Raises:
            ValueError: If any of the provided batch size bounds are invalid.
        """
        if not isinstance(max_batch_size, int) or \
                max_batch_size < self.batch_size:
            raise ValueError("invalid max_batch_size")
        if local_bsz_bounds is not None and (
                local_bsz_bounds[0] is not None and
                local_bsz_bounds[0] > self.batch_size or
                local_bsz_bounds[1] is not None and
                local_bsz_bounds[1] < self.batch_size):
            raise ValueError("invalid local_bsz_bounds")
        self._max_batch_size = max_batch_size
        self._local_bsz_bounds = local_bsz_bounds
        self._gradient_accumulation = gradient_accumulation

