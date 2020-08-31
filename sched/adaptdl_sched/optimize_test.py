import pytest
import time

from collections import Counter
from datetime import datetime, timedelta
from adaptdl_sched.optimize import AdaptDLPolicy, JobInfo, NodeInfo
from adaptdl.speedup import Params, SpeedupFunction


@pytest.mark.parametrize("num_nodes", [1, 2, 4, 8, 16])
def test_optimize(num_nodes, total_devices=16):
    assert total_devices % num_nodes == 0
    num_devices = total_devices // num_nodes
    print("{}x{} nodes:".format(num_nodes, num_devices))
    # Make up a realistic speedup function.
    params = Params(0.121, 0.00568, 0.0236, 0.00634, 0.0118, 0.00317, 1.14)
    grad_params = {"norm": 0.00136, "var": 0.000502}
    speedup_fn = SpeedupFunction(
            params, grad_params, init_batch_size=128, max_batch_size=1280,
            local_bsz_bounds=(64, 256), elastic_bsz=True)
    now = datetime.now()
    jobs = {}
    # Add a few preemptible jobs.
    job_resources = {"nvidia.com/gpu": 1, "pods": 1}
    for i in range(16):
        creation_timestamp = now + timedelta(minutes=len(jobs)),
        progress = 2 ** i
        max_replicas = 8
        preemptible = True
        key = len(jobs)
        jobs[key] = JobInfo(job_resources, speedup_fn, creation_timestamp,
                            progress, max_replicas, preemptible)
    # Add a few non-preemptible jobs.
    for i in range(4):
        creation_timestamp = now + timedelta(minutes=len(jobs)),
        progress = 4 ** i
        max_replicas = 1
        preemptible = False
        key = len(jobs)
        jobs[key] = JobInfo(job_resources, speedup_fn, creation_timestamp,
                            progress, max_replicas, preemptible)
    # Add a few existing nodes.
    node_resources = {"nvidia.com/gpu": num_devices, "pods": 32}
    nodes = {i: NodeInfo(node_resources, 1.0)
             for i in range(num_nodes)}
    # Add a few virtual nodes.
    nodes.update({i: NodeInfo(node_resources, 1.0, eta=600)
                  for i in range(num_nodes, 2 * num_nodes)})
    policy = AdaptDLPolicy()
    prev_allocs = {}
    for i in range(3):
        start = time.time()
        result = policy.optimize(jobs, nodes, prev_allocs, generations=100)
        duration = time.time() - start
        result = sorted(result, key=lambda entry: entry[0])
        print("optimize {}x ({}s sec):".format(i + 1, duration), result[0][0],
              result[len(result) // 2][0], result[-1][0])
        for _, allocs, _ in result:
            node_count = Counter()
            for job_key, placement in allocs.items():
                assert len(placement) <= jobs[job_key].max_replicas
                for node_key in placement:
                    node_count[node_key] += 1
            for node_key, count in node_count.items():
                assert count <= nodes[node_key].resources["nvidia.com/gpu"]
                assert count <= nodes[node_key].resources["pods"]
                assert nodes[node_key].eta == 0
