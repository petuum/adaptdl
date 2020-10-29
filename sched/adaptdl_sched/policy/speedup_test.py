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

from unittest.mock import Mock
from adaptdl_sched.policy.speedup import SpeedupFunction


def mock_optimize(num_nodes, num_replicas, *args, **kwargs):
    return 32 * np.sqrt(num_replicas), 32, 0


def test_speedup():
    goodput_fn = Mock()
    goodput_fn.optimize = Mock(side_effect=mock_optimize)
    speedup_fn = SpeedupFunction(goodput_fn)
    num_replicas = np.arange(1, 100)
    expected_ret = np.sqrt(num_replicas)
    assert np.allclose(speedup_fn(1, num_replicas), expected_ret)


def test_memoize():
    goodput_fn = Mock()
    goodput_fn.optimize = Mock(side_effect=mock_optimize)
    speedup_fn = SpeedupFunction(goodput_fn)
    goodput_fn.optimize.reset_mock()
    # Call with never-seen inputs.
    num_nodes = num_replicas = np.arange(1, 10)
    result = speedup_fn(num_nodes, num_replicas)
    assert np.allclose(result, np.sqrt(num_replicas))
    assert goodput_fn.optimize.call_count == 1
    assert np.all(goodput_fn.optimize.call_args[0][0] == num_nodes)
    assert np.all(goodput_fn.optimize.call_args[0][1] == num_replicas)
    # Call with some-seen inputs.
    num_nodes = num_replicas = np.arange(1, 20)
    result = speedup_fn(num_nodes, num_replicas)
    assert np.allclose(result, np.sqrt(num_replicas))
    assert goodput_fn.optimize.call_count == 2
    assert np.all(goodput_fn.optimize.call_args[0][0] == np.arange(10, 20))
    assert np.all(goodput_fn.optimize.call_args[0][1] == np.arange(10, 20))
    # Call with all-seen inputs.
    num_nodes_3 = num_replicas_3 = np.arange(5, 15)
    expect_3 = np.sqrt(num_replicas_3)
    result_3 = speedup_fn(num_nodes_3, num_replicas_3)
    assert np.allclose(result_3, expect_3)
    assert goodput_fn.optimize.call_count == 2  # Shouldn't have increased.
