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

from collections import Counter
from datetime import datetime, timedelta
from adaptdl.goodput import GoodputFunction, PerfParams, GradParams
from adaptdl_sched.policy.pollux import PolluxPolicy
from adaptdl_sched.policy.speedup import SpeedupFunction
from adaptdl_sched.policy.utils import JobInfo, NodeInfo


@pytest.mark.parametrize("num_nodes", [1, 2, 4, 8, 16])
def test_optimize(num_nodes, total_devices=16):
    assert total_devices % num_nodes == 0
    num_devices = total_devices // num_nodes
    print("{}x{} nodes:".format(num_nodes, num_devices))
    # Make up a realistic speedup function.
    perf_params = PerfParams(0.121, 0.00568, 0.0236, 0.00634,
                             0.0118, 0.00317, 1.14)
    grad_params = GradParams(sqr=0.00136, var=0.000502)
    goodput_fn = GoodputFunction(perf_params, grad_params, 128)
    speedup_fn = SpeedupFunction(goodput_fn, max_batch_size=1280,
                                 atomic_bsz_range=(64, 256))
    now = datetime.now()
    jobs = {}
    # Add a few jobs.
    job_resources = {"nvidia.com/gpu": 1, "pods": 1}
    for i in range(16):
        creation_timestamp = now + timedelta(minutes=len(jobs)),
        max_replicas = 8
        min_replicas = 0
        key = len(jobs)
        jobs[key] = JobInfo(job_resources, speedup_fn, creation_timestamp,
                            min_replicas, max_replicas)
    # Add a few nodes.
    node_resources = {"nvidia.com/gpu": num_devices, "pods": 32}
    nodes = {i: NodeInfo(node_resources, preemptible=False)
             for i in range(num_nodes)}
    # Add a node template.
    node_template = NodeInfo(node_resources, preemptible=True)
    policy = PolluxPolicy()
    prev_allocs = {}
    for i in range(3):
        start = time.time()
        allocations, desired_nodes = \
            policy.optimize(jobs, nodes, prev_allocs, node_template)
        duration = time.time() - start
        print("optimize {}x ({}s sec):".format(i + 1, duration))
        node_count = Counter()
        for job_key, placement in allocations.items():
            assert len(placement) <= jobs[job_key].max_replicas
            for node_key in placement:
                node_count[node_key] += 1
        for node_key, count in node_count.items():
            assert count <= nodes[node_key].resources["nvidia.com/gpu"]
            assert count <= nodes[node_key].resources["pods"]


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
    job_1 = JobInfo({"gpu": 1, "cpu": 500, "pods": 1}, speedup_fn,
                    now + timedelta(minutes=0), min_replicas, max_replicas=1)
    job_2 = JobInfo({"gpu": 1, "cpu": 1000, "pods": 1}, speedup_fn,
                    now + timedelta(minutes=1), min_replicas, max_replicas=1)
    job_3 = JobInfo({"gpu": 1, "cpu": 1000, "pods": 1}, speedup_fn,
                    now + timedelta(minutes=1), 2, max_replicas=2)
    job_4 = JobInfo({"gpu": 1, "cpu": 2000, "pods": 1}, speedup_fn,
                    now + timedelta(minutes=1), 2, max_replicas=2)
    policy = PolluxPolicy()

    assert(policy.allocate_job(job_1, nodes) == ["0"])
    assert(policy.allocate_job(job_2, nodes) == ["1"])
    assert(policy.allocate_job(job_3, nodes) == ["1", "1"])
    assert(policy.allocate_job(job_4, nodes) == [])


def test_unusable_node():
    # Test where one of the nodes can't be used due to one resource type.
    nodes = {
        0: NodeInfo({"gpu": 1, "cpu": 500, "pods": 32}, preemptible=False),
        1: NodeInfo({"gpu": 1, "cpu": 8000, "pods": 32}, preemptible=False),
        2: NodeInfo({"gpu": 1, "cpu": 8000, "pods": 32}, preemptible=False),
    }
    template = NodeInfo({"gpu": 1, "cpu": 8000, "pods": 32}, preemptible=True)
    perf_params = PerfParams(0.121, 0.00568, 0.0236, 0.00634,
                             0.0118, 0.00317, 1.14)
    grad_params = GradParams(sqr=0.00136, var=0.000502)
    goodput_fn = GoodputFunction(perf_params, grad_params, 128)
    speedup_fn = SpeedupFunction(goodput_fn, max_batch_size=1280,
                                 atomic_bsz_range=(64, 256))
    now = datetime.now()
    min_replicas = 0
    jobs = {
        0: JobInfo({"gpu": 1, "cpu": 1000, "pods": 1}, speedup_fn,
                   now + timedelta(minutes=0), min_replicas, max_replicas=1),
        1: JobInfo({"gpu": 1, "cpu": 1000, "pods": 1}, speedup_fn,
                   now + timedelta(minutes=1), min_replicas, max_replicas=1),
        2: JobInfo({"gpu": 1, "cpu": 1000, "pods": 1}, speedup_fn,
                   now + timedelta(minutes=2), min_replicas, max_replicas=1),
    }
    policy = PolluxPolicy()
    allocations, desired_nodes = policy.optimize(jobs, nodes, {}, template)
    # Check that more nodes are asked for.
    assert desired_nodes > 3
    # Check no job was allocated more than 1 replica.
    assert max(len(alloc) for alloc in allocations.values()) == 1
    # Check two jobs were allocated.
    assert sum(len(alloc) for alloc in allocations.values()) == 2
