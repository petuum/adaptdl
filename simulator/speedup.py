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

import numpy as np


class SpeedupFunction(object):

    def __init__(self, goodput_fn, max_batch_size=None, atomic_bsz_range=None,
                 accumulation=False, mem_size=32):
        self._goodput_fn = goodput_fn
        self._max_batch_size = max_batch_size
        self._atomic_bsz_range = atomic_bsz_range
        self._accumulation = accumulation
        self._mem_size = mem_size
        self._base_goodput, _, _ = goodput_fn.optimize(
            num_nodes=1, num_replicas=1, max_batch_size=max_batch_size,
            atomic_bsz_range=atomic_bsz_range, accumulation=accumulation)
        # Memoization for fast repeated queries.
        self._mem_speedup = -np.ones((mem_size, mem_size))
        self._mem_speedup[0, 0] = 0.0

    def __call__(self, num_nodes, num_replicas):
        assert np.all(np.less_equal(0, num_nodes))
        assert np.all(np.less_equal(num_nodes, num_replicas))
        assert np.all((num_nodes > 0) == (num_replicas > 0))
        # Remember what the output shape/format should be and flatten inputs.
        output_scalar = np.isscalar(num_nodes) and np.isscalar(num_replicas)
        output_shape = np.broadcast(num_nodes, num_replicas).shape
        num_nodes = np.broadcast_to(num_nodes, output_shape).flatten()
        num_replicas = np.broadcast_to(num_replicas, output_shape).flatten()
        # Return values which will be filled out.
        speedup = -np.ones(output_shape).flatten()
        # Fill in any previously memoized results first.
        indices = num_replicas < self._mem_size
        mem_idx = (num_nodes[indices], num_replicas[indices])
        speedup[indices] = self._mem_speedup[mem_idx]
        # Find the missing indices which still need to be computed.
        missing = speedup < 0
        if np.count_nonzero(missing) > 0:
            num_nodes, num_replicas = num_nodes[missing], num_replicas[missing]
            # Find unique inputs to reduce compuation.
            (num_nodes, num_replicas), inverse = np.unique(
                    np.stack([num_nodes, num_replicas]),
                    axis=1, return_inverse=True)
            goodput, _, _ = self._goodput_fn.optimize(
                num_nodes, num_replicas,
                max_batch_size=self._max_batch_size,
                atomic_bsz_range=self._atomic_bsz_range,
                accumulation=self._accumulation)
            # Memoize results.
            indices = num_replicas < self._mem_size
            mem_idx = (num_nodes[indices], num_replicas[indices])
            self._mem_speedup[mem_idx] = goodput[indices] / self._base_goodput
            # Fill in computed results.
            speedup[missing] = goodput[inverse] / self._base_goodput
        assert np.all(np.less_equal(0, speedup))
        speedup = speedup.reshape(output_shape)
        return speedup.item() if output_scalar else speedup
