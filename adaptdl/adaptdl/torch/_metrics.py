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

import adaptdl.checkpoint
import adaptdl.collective
import adaptdl.env
from adaptdl.goodput import GoodputFunction, fit_perf_params
from adaptdl.sched_hints import SCHED_HINTS, PERF_PARAMS, post_sched_hints


def profile_step_start(atomic_bsz):
    state = _metrics_state()
    state.atomic_bsz = atomic_bsz
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
    key = (num_nodes, num_replicas, state.atomic_bsz)
    if accumulation_step:
        state.profile[key]["accum_step_time"] += step_time
        state.profile[key]["accum_count"] += 1
    else:
        state.profile[key]["optim_step_time"] += step_time
        state.profile[key]["optim_sync_time"] += state.sync_time
        state.profile[key]["optim_count"] += 1
    del state.atomic_bsz
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


def update_progress(progress):
    _metrics_state().progress = progress


def get_progress():
    return _metrics_state().progress


def set_batch_size(init_batch_size, max_batch_size, local_bsz_bounds,
                   gradient_accumulation):
    state = _metrics_state()
    state.init_batch_size = init_batch_size
    state.max_batch_size = max_batch_size
    state.local_bsz_bounds = local_bsz_bounds
    state.gradient_accumulation = gradient_accumulation


def get_goodput_fn():
    state = _metrics_state()
    if state.grad_params is None or state.perf_params is None:
        return None
    return GoodputFunction(state.perf_params, state.grad_params,
                           state.init_batch_size)


def _fit_perf_params():
    state = _metrics_state()
    profile = {k: v for k, v in state.profile.items() if v.get("optim_count")}
    # Convert profile into numpy arrays.
    num_nodes, num_replicas, atomic_bsz = (
        np.array(k) for k in zip(*profile.keys()))
    accum_step_time = np.array([v.get("accum_step_time", 0.0)
                                for v in profile.values()])
    accum_count = np.array([v.get("accum_count", 0) for v in profile.values()])
    optim_step_time = np.array([v.get("optim_step_time", 0.0)
                                for v in profile.values()])
    optim_sync_time = np.array([v.get("optim_sync_time", 0.0)
                                for v in profile.values()])
    optim_count = np.array([v.get("optim_count", 0) for v in profile.values()])
    assert np.all(optim_count > 0)
    # Non-sync time during optimization steps should be approximately equal to
    # accumulation step time, combine those data points.
    assert np.all(optim_step_time >= optim_sync_time)
    accum_step_time += optim_step_time - optim_sync_time
    accum_count += optim_count
    accum_step_time /= accum_count
    optim_step_time /= optim_count
    state.perf_params = fit_perf_params(num_nodes, num_replicas, atomic_bsz,
                                        accum_step_time, optim_step_time)


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
