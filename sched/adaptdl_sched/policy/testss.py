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


import pytest
import time
import sys
sys.path.append("/home/ubuntu/adaptdl/sched")
sys.path.append("/home/ubuntu/adaptdl/adaptdl/adaptdl")
from collections import Counter
from datetime import datetime, timedelta
from adaptdl_sched.policy.pollux import PolluxPolicy
from adaptdl_sched.policy.speedup import SpeedupFunction
from adaptdl_sched.policy.utils import JobInfo, NodeInfo
from goodput import GoodputFunction, PerfParams, GradParams
def test_allocate_job():
    nodes = {
        "0": NodeInfo({"gpu": 1, "cpu": 500, "pods": 32}, preemptible=False),
        "1": NodeInfo({"gpu": 2, "cpu": 2000, "pods": 32}, preemptible=False),
        "2": NodeInfo({"gpu": 2, "cpu": 3000, "pods": 32}, preemptible=True),
    }
    perf_params = PerfParams(0.121, 0.00568, 0.0236, 0.00634,
                             0.0118, 0.00317, 1.14)
    grad_params = GradParams(sqr=0.00136, var=0.000502)
    goodput_fn = GoodputFunction(perf_params, grad_params, 128)
    speedup_fn = SpeedupFunction(goodput_fn, max_batch_size=1280,
                                 atomic_bsz_range=(64, 256))
    now = datetime.now()
    min_replicas = 0
    job_1 = {
        "1": JobInfo({"gpu": 1, "cpu": 500, "pods": 1}, speedup_fn,
                   now + timedelta(minutes=0), min_replicas, max_replicas=1),
    }

    job_2 = {
        "2": JobInfo({"gpu": 1, "cpu": 1000, "pods": 1}, speedup_fn,
                   now + timedelta(minutes=1), min_replicas, max_replicas=1),
    }

    job_3 = {
        "3": JobInfo({"gpu": 1, "cpu": 1000, "pods": 1}, speedup_fn,
                   now + timedelta(minutes=1), 2, max_replicas=2),
    }

    job_4 = {
        "4": JobInfo({"gpu": 1, "cpu": 2000, "pods": 1}, speedup_fn,
                   now + timedelta(minutes=1), 2, max_replicas=2),
    }
    policy = PolluxPolicy()

    assert(policy._allocate_job(job_1, nodes) == ["0"])
    assert(policy._allocate_job(job_2, nodes) == ["1"])
    assert(policy._allocate_job(job_3, nodes) == ["1", "1"])
    assert(policy._allocate_job(job_4, nodes) is None)

test_allocate_job()
