# Copyright 2021 Petuum, Inc. All Rights Reserved.
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

import copy
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
from collections import Counter
import logging
import ray
from ray import tune

from adaptdl.goodput import GoodputFunction, PerfParams, GradParams
from adaptdl_sched.policy.speedup import SpeedupFunction
from adaptdl_sched.policy.utils import JobInfo, NodeInfo
import config

import time

class AdaptDLJobMixin:
    def __init__(self, *args, **kwargs):
        # Be wary of putting large data members here. Tune Experiment checkpointing
        # may try to serialize this.
        self._job_id = kwargs.pop("job_id", 0)
        self.creation_timestamp = datetime.now()
        super().__init__(*args, **kwargs)

    @property
    def job_id(self):
        return self._job_id

    def _fetch_metrics(self):
        """ Returns metrics of this AdaptDLJob."""
        raise NotImplementedError

    def _allocation_in_use(self) -> bool:
        """ Returns True if the allocation is being used by an AdaptDLJob."""
        raise NotImplementedError

    @property
    def job_info(self):
        metrics = self._fetch_metrics()
        if metrics is not None:
            if type(metrics) == dict:
                perf_params = metrics["perfParams"]
                grad_params = metrics["gradParams"]
                init_batch_size = metrics["initBatchSize"]
                atomic_bsz_range = metrics["localBszBounds"]
                max_batch_size = metrics["maxBatchSize"]
                accumulation = metrics["gradientAccumulation"]
            else:
                perf_params = metrics.perf_params
                grad_params = metrics.grad_params
                init_batch_size =  128
                atomic_bsz_range = (32, 256)
                max_batch_size = 1024
                accumulation = False
                min_replicas = config._JOB_MIN_REPLICAS
                max_replicas = config._JOB_MAX_REPLICAS
            goodput_fn = GoodputFunction(perf_params, grad_params, init_batch_size)
            speedup_fn = SpeedupFunction(goodput_fn, max_batch_size=max_batch_size, # TODO
                                         atomic_bsz_range=atomic_bsz_range, accumulation=accumulation)
        else:
            speedup_fn = lambda n, r: r  # noqa: E731

        return JobInfo(config.job_resources(), speedup_fn, self.creation_timestamp,
                       0, 10)

    @property
    def allocation(self):
        # Allocation is in use if the job is using it
        if self._allocation_in_use():
            return AdaptDLJobMixin.pgf_to_allocation(self.placement_group_factory)
        else:
            return []

    @staticmethod
    def pgf_to_allocation(pgf) -> List[str]:
        bundles = pgf._bundles[1:]
        allocs, node_keys, num_devices = [], [], []
        for bundle in bundles:
            node_keys += [k.split(":")[1] for k, v in bundle.items() if k.startswith("node")]
            num_devices += [int(v) for k, v in bundle.items() if k == config.default_device()]

        for node, count in zip(node_keys, num_devices):
            allocs += [node] * count
        return allocs

    @staticmethod
    def allocation_to_pgf(alloc: List[str], resources_per_node):
        time.sleep(5)
        def _construct_bundle(node, device_count):
            resources = copy.deepcopy(resources_per_node)
            #resources = {config.default_device(): device_count}
            if "virtual" not in node:
                resources[f"node:{node}"] = 0.01
            return resources

        assert len(alloc) > 0
        resources = [{"CPU": 0.01}]
        alloc = Counter(alloc)
        for node, res in alloc.items():
            resources.append(_construct_bundle(node, res))
        time.sleep(5)
        return tune.PlacementGroupFactory(resources)
    
    @staticmethod
    def _pgf_to_num_replicas(pgf) -> int:
        return sum(int(bundle.get(config.default_device(), 0)) 
                       for bundle in pgf._bundles[1:])
