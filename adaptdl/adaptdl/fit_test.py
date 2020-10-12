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


import adaptdl.speedup as speedup
import numpy as np


def test_fit_1():
    # Tests speedup.fit's ability to fit to data generated
    # by its own model class with random parameters, without
    # gradient accumulation. Serves as a sanity check
    # that the speedup model fitting works in the most
    # optimistic case.
    size = (1000,)
    nodes = np.random.randint(low=1, high=11, size=size)
    replicas = np.random.randint(low=1, high=nodes+1, size=size)
    local_bsz = np.random.randint(32, 1024, size=size)
    accumulation_steps = np.random.randint(0, 1, size=size)
    params = speedup.Params(0.1, 0.01, 0.5, 1.0, 1e-6, 1e-6, 1.2)
    step_time_compute = speedup._predict_compute(params, local_bsz) + \
        np.maximum(np.random.normal(0, 0.001, size=size), 0.0)
    step_time_network = speedup._predict_network(params, nodes, replicas) + \
        np.maximum(np.random.normal(0, 0.001, size=size), 0.0)
    step_time_accumulation = \
        speedup._predict_compute(params, local_bsz) + \
        np.maximum(np.random.normal(0, 0.001, size=size), 0.0)
    gamma = params.gamma
    step_time = (
        (step_time_compute ** gamma
         + step_time_network ** gamma) ** (1 / gamma)
        + step_time_accumulation * accumulation_steps)
    result = speedup.fit(nodes, replicas, local_bsz, accumulation_steps,
                         step_time, step_time_compute, step_time_accumulation)
    loss_result = speedup._obj_fn(
        result, nodes, replicas, local_bsz, accumulation_steps,
        step_time, step_time_compute, step_time_accumulation)
    loss_true = speedup._obj_fn(
        params, nodes, replicas, local_bsz, accumulation_steps,
        step_time, step_time_compute, step_time_accumulation)
    assert(abs(loss_result - loss_true) < 0.1 * loss_true
           or loss_result < loss_true)


def test_fit_2():
    # Tests speedup.fit's ability to fit to data generated
    # by its own model class with random parameters, with
    # gradient accumulation. Serves as a sanity check
    # that the speedup model fitting works in the most
    # optimistic case.
    size = (1000,)
    nodes = np.random.randint(low=1, high=11, size=size)
    replicas = np.random.randint(low=1, high=nodes+1, size=size)
    local_bsz = np.random.randint(32, 1024, size=size)
    accumulation_steps = np.random.randint(0, 10, size=size)
    params = speedup.Params(0.1, 0.01, 0.5, 1.0, 1e-6, 1e-6, 1.2)
    step_time_compute = speedup._predict_compute(params, local_bsz) + \
        np.maximum(np.random.normal(0, 0.01, size=size), 0.0)
    step_time_network = speedup._predict_network(params, nodes, replicas) + \
        np.maximum(np.random.normal(0, 0.01, size=size), 0.0)
    step_time_accumulation = \
        speedup._predict_compute(params, local_bsz) + \
        np.maximum(np.random.normal(0, 0.001, size=size), 0.0)
    gamma = params.gamma
    step_time = (
        (step_time_compute) ** gamma
        + step_time_network ** gamma) ** (1 / gamma) + step_time_accumulation
    result = speedup.fit(nodes, replicas, local_bsz, accumulation_steps,
                         step_time, step_time_compute, step_time_accumulation)
    loss_result = speedup._obj_fn(
        result, nodes, replicas, local_bsz, accumulation_steps,
        step_time, step_time_compute, step_time_accumulation)
    loss_true = speedup._obj_fn(
        params, nodes, replicas, local_bsz, accumulation_steps,
        step_time, step_time_compute, step_time_accumulation)
    assert(abs(loss_result - loss_true) < 0.1 * loss_true
           or loss_result < loss_true)
