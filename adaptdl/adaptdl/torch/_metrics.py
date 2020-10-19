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
import pickle
import time

import numpy as np

import adaptdl.speedup
import adaptdl.checkpoint
import adaptdl.collective
from adaptdl.sched_hints import SCHED_HINTS, PERF_PARAMS, \
        post_sched_hints


def profile_step_start(local_bsz, gradient_accumulation_steps):
    state = _metrics_state()
    state.local_bsz = local_bsz
    state.gradient_accumulation_steps = gradient_accumulation_steps
    state.step_start = time.time()
    state.sync_time = 0.0


def profile_sync_time(sync_time):
    _metrics_state().sync_time += sync_time


_PREV_REPORT = None


def profile_step_commit(accumulation_step=False):
    global _PREV_REPORT
    state = _metrics_state()
    step_time = time.time() - state.step_start
    num_nodes = adaptdl.env.num_nodes()
    num_replicas = adaptdl.env.num_replicas()
    key = (num_nodes, num_replicas,
           state.local_bsz, state.gradient_accumulation_steps)
    if accumulation_step:
        state.profile[key]["accumulation_step_time"] += step_time
        state.profile[key]["accumulation_count"] += 1
    else:
        state.profile[key]["step_time"] += step_time
        state.profile[key]["sync_time"] += state.sync_time
        state.profile[key]["count"] += 1
    del state.local_bsz
    del state.gradient_accumulation_steps
    del state.step_start
    del state.sync_time
    if not accumulation_step:
        if _PREV_REPORT is None:
            _PREV_REPORT = time.time()
        if adaptdl.env.replica_rank() == 0 and time.time() - _PREV_REPORT > 30:
            _fit_perf_params()
            _report_sched_hints()
            _PREV_REPORT = time.time()


_GRAD_PARAM_DICT = {}


def update_grad_params(edp_key, grad_norm_sqr, grad_variance):
    global _GRAD_PARAM_DICT
    _GRAD_PARAM_DICT[edp_key] = np.asarray([grad_norm_sqr, grad_variance])
    grad_params = sum(_GRAD_PARAM_DICT.values())
    _metrics_state().grad_params = (grad_params[0], grad_params[1])


def update_progress(gain):
    _metrics_state().progress += gain


def get_progress():
    return _metrics_state().progress


def set_batch_size(init_batch_size, max_batch_size, local_bsz_bounds,
                   gradient_accumulation):
    state = _metrics_state()
    state.init_batch_size = init_batch_size
    state.max_batch_size = max_batch_size
    state.local_bsz_bounds = local_bsz_bounds
    state.gradient_accumulation = gradient_accumulation


def get_speedup_fn():
    state = _metrics_state()
    grad_params = {
        "norm": state.grad_params and state.grad_params[0],
        "var": state.grad_params and state.grad_params[1],
    }
    return adaptdl.speedup.SpeedupFunction(
        state.perf_params, grad_params,
        init_batch_size=state.init_batch_size,
        max_batch_size=state.max_batch_size,
        local_bsz_bounds=state.local_bsz_bounds,
        gradient_accumulation=state.gradient_accumulation,
        elastic_bsz=(state.max_batch_size is not None),
    )


def _fit_perf_params():
    state = _metrics_state()
    num_nodes = np.array([key[0] for key in state.profile])
    num_replicas = np.array([key[1] for key in state.profile])
    local_bsz = np.array([key[2] for key in state.profile])
    accumulation_steps = np.array([key[3] for key in state.profile])
    values = state.profile.values()
    values = [value for value in values if value["count"] > 0]
    step_time = np.array([val["step_time"] / val["count"] for val in values])
    sync_time = np.array([val["sync_time"] / val["count"] for val in values])
    accumulation_time = np.array(
            [val["accumulation_step_time"] / val["accumulation_count"]
             if val["accumulation_count"] > 0 else 0.0
             for val in values])
    compute_time = step_time - sync_time
    accumulation_time = np.where(
        accumulation_steps > 0, accumulation_time, compute_time)
    state.perf_params = adaptdl.speedup.fit(
        num_nodes, num_replicas, local_bsz, accumulation_steps,
        step_time, compute_time, accumulation_time)


def _report_sched_hints():
    assert adaptdl.env.replica_rank() == 0
    state = _metrics_state()
    # Scheduling hints
    sched_hints = SCHED_HINTS.copy()
    sched_hints["perfParams"] = {k: v for (k, v) in
                                 zip(PERF_PARAMS.keys(),
                                 state.perf_params)}
    sched_hints["maxBatchSize"] = state.max_batch_size
    sched_hints["localBszBounds"] = state.local_bsz_bounds
    sched_hints["initBatchSize"] = state.init_batch_size
    if state.grad_params:
        sched_hints["gradParams"] = {}
        sched_hints["gradParams"]["norm"] = state.grad_params[0]
        sched_hints["gradParams"]["var"] = state.grad_params[1]
    sched_hints["maxProfiledReplicas"] = max(key[1] for key in state.profile)
    sched_hints["gradientAccumulation"] = state.gradient_accumulation
    post_sched_hints(sched_hints, adaptdl.env.job_id())


class _MetricsState(adaptdl.checkpoint.State):
    def __init__(self):
        super().__init__("adaptdl-metrics")
        self.profile = collections.defaultdict(collections.Counter)
        self.perf_params = None
        self.grad_params = None
        self.init_batch_size = None
        self.max_batch_size = None
        self.local_bsz_bounds = None
        self.gradient_accumulation = False
        self.progress = 0.0  # Progress in scale-invariant iterations.

    def save(self, fileobj):
        pickle.dump(self.profile, fileobj)
        pickle.dump(self.perf_params, fileobj)
        pickle.dump(self.grad_params, fileobj)
        pickle.dump(self.init_batch_size, fileobj)
        pickle.dump(self.max_batch_size, fileobj)
        pickle.dump(self.local_bsz_bounds, fileobj)
        pickle.dump(self.gradient_accumulation, fileobj)
        pickle.dump(self.progress, fileobj)

    def load(self, fileobj):
        self.profile = pickle.load(fileobj)
        self.perf_params = pickle.load(fileobj)
        self.grad_params = pickle.load(fileobj)
        self.init_batch_size = pickle.load(fileobj)
        self.max_batch_size = pickle.load(fileobj)
        self.local_bsz_bounds = pickle.load(fileobj)
        self.gradient_accumulation = pickle.load(fileobj)
        self.progress = pickle.load(fileobj)


def _metrics_state():
    global _METRICS_STATE
    if _METRICS_STATE is None:
        _METRICS_STATE = _MetricsState()
        adaptdl.checkpoint.load_state(_METRICS_STATE)
    return _METRICS_STATE


_METRICS_STATE = None
