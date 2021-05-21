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
import random
import copy

from collections import Counter
from datetime import datetime, timedelta
from adaptdl.goodput import GoodputFunction, PerfParams, GradParams
from adaptdl_sched.policy.pollux import PolluxPolicy
from adaptdl_sched.policy.speedup import SpeedupFunction
from adaptdl_sched.policy.utils import JobInfo, NodeInfo


@pytest.mark.parametrize("num_nodes", [1, 2, 4, 8])
def test_optimize(num_nodes, total_devices=16):
    # Globals
    N_JOBS = 10
    JOBS = list(range(N_JOBS))
    random.shuffle(JOBS)

    PREEMPTIBLE_IDXS = JOBS[:len(JOBS)//2]
    NON_PREEMPTIBLE_IDXS = JOBS[len(JOBS)//2:]

    assert total_devices % num_nodes == 0
    num_devices = total_devices // num_nodes
    print(f"{num_nodes}x{num_devices} nodes:")
    # Make up a realistic speedup function.
    perf_params = PerfParams(0.121, 0.00568, 0.0236, 0.00634,
                             0.0118, 0.00317, 1.14)
    grad_params = GradParams(sqr=0.00136, var=0.000502)
    goodput_fn = GoodputFunction(perf_params, grad_params, 128)
    speedup_fn = SpeedupFunction(goodput_fn, max_batch_size=1280,
                                 atomic_bsz_range=(64, 256))
    now = datetime.now()
    # Add a node template.
    policy = PolluxPolicy()
    job_resources = {"nvidia.com/gpu": 1, "pods": 1}
    # Add a few nodes.
    node_resources = {"nvidia.com/gpu": num_devices, "pods": 32}
    nodes = {i: NodeInfo(node_resources, preemptible=False)
             for i in range(num_nodes)}
    node_template = NodeInfo(node_resources, preemptible=True)

    # Empty allocations
    prev_allocs = {i: [] for i in JOBS}
    for cycle in range(3):
        # Start allocation cycle
        jobs = {}
        for i in PREEMPTIBLE_IDXS:
            creation_timestamp = now + timedelta(minutes=i),
            jobs[i] = JobInfo(job_resources, speedup_fn, creation_timestamp,
                              min_replicas=0, max_replicas=8)
        for i in NON_PREEMPTIBLE_IDXS:
            creation_timestamp = now + timedelta(minutes=i),
            jobs[i] = JobInfo(job_resources, speedup_fn, creation_timestamp,
                              min_replicas=2, max_replicas=4,
                              preemptible=False)
        start = time.time()
        assert len(jobs) > 0
        allocations, desired_nodes = \
            policy.optimize(jobs, nodes, prev_allocs, node_template)
        duration = time.time() - start
        print(f"optimize {cycle + 1}x ({duration}s sec)")
        node_count = Counter()
        for job_key, placement in allocations.items():
            assert len(placement) <= jobs[job_key].max_replicas
            if placement:
                assert len(placement) >= jobs[job_key].min_replicas
            for node_key in placement:
                node_count[node_key] += 1
        for node_key, count in node_count.items():
            assert count <= nodes[node_key].resources["nvidia.com/gpu"]
            assert count <= nodes[node_key].resources["pods"]

        # Check if we are maintaining allocations for non-preemptible jobs
        for i in NON_PREEMPTIBLE_IDXS:
            if (i in allocations) and prev_allocs[i]:
                assert allocations[i] == prev_allocs[i]

        prev_allocs = copy.deepcopy(allocations)
        # Remove one random job
        remove = random.sample(allocations.keys(), 1)[0]
        if remove in NON_PREEMPTIBLE_IDXS:
            NON_PREEMPTIBLE_IDXS.remove(remove)
            print(f"Deleting non-preemptible job {remove}")
        else:
            PREEMPTIBLE_IDXS.remove(remove)
            print(f"Deleting preemptible job {remove}")
        prev_allocs.pop(remove)
