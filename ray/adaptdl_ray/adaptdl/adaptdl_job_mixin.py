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


from typing import List
from datetime import datetime

from adaptdl.goodput import GoodputFunction, GradParams
from adaptdl_sched.policy.speedup import SpeedupFunction
from adaptdl_sched.policy.utils import JobInfo
from adaptdl_ray.adaptdl import config
from adaptdl_ray.adaptdl.utils import pgf_to_allocation


class AdaptDLJobMixin:
    def __init__(self, *args, **kwargs):
        # Be wary of putting large data members here. Tune Experiment
        # checkpointing may try to serialize this.
        self._job_id = kwargs.pop("job_id", 0)
        self.creation_timestamp = kwargs.pop("creation_timestamp",
                                             datetime.now())
        super().__init__(*args, **kwargs)

    @property
    def job_id(self):
        """ Unique job ID assigned to this AdaptDL job"""
        return self._job_id

    def _fetch_metrics(self):
        """ Returns perf metrics of this AdaptDLJob. This could return a cached
        copy in case the job is currently not running."""
        raise NotImplementedError

    def _allocation_in_use(self) -> bool:
        """ Returns True if the allocation is being used by an AdaptDLJob."""
        raise NotImplementedError

    @property
    def job_info(self) -> JobInfo:
        metrics = self._fetch_metrics()
        if metrics is not None:
            perf_params = metrics.perf_params
            if metrics.grad_params is not None:
                grad_params = metrics.grad_params
            else:
                grad_params = GradParams(0.0, 1.0)
            goodput_fn = GoodputFunction(perf_params,
                                         grad_params,
                                         metrics.init_batch_size)
            speedup_fn = SpeedupFunction(goodput_fn,
                                         metrics.max_batch_size,
                                         metrics.local_bsz_bounds,
                                         metrics.gradient_accumulation)
        else:
            speedup_fn = lambda n, r: r  # noqa: E731

        return JobInfo(config.job_resources(),
                       speedup_fn,
                       self.creation_timestamp,
                       config._JOB_MIN_REPLICAS,
                       config._JOB_MAX_REPLICAS)

    @property
    def allocation(self) -> List[str]:
        """ Current allocation the job is utilizing"""
        # Allocation is in use if the job is using it
        if self._allocation_in_use():
            assert self.placement_group_factory is not None
            return pgf_to_allocation(self.placement_group_factory)
        else:
            return []
