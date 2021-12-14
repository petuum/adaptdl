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


import numpy as np


def optimize(job, cluster, max_cluster_size):
    hints = job.hints
    if not hints:
        return ["adaptdl_virtual_node_0"]

    job_info = job.job_info

    existing_nodes = cluster.get_nodes()
    virtual_nodes = [
        f"adaptdl_virtual_node_{i}"
        for i in range(max_cluster_size - len(existing_nodes))]

    existing_ips = [node["NodeManagerAddress"] for node in existing_nodes]

    nodes = existing_ips + virtual_nodes

    existing_ips = set(existing_ips)
    node_resources = {
        node["NodeManagerAddress"]: node["Resources"]
        for node in existing_nodes}

    job_resources = cluster.worker_resources

    replicas_per_node = [0 for node in nodes]
    for index, node in enumerate(nodes):
        if node not in node_resources:
            resources = cluster.worker_resources
        else:
            resources = node_resources[node]
        replicas_per_node[index] = int(
            min(resources.get(resource_type, 0.0) / value
                for resource_type, value in job_resources.items()))

    max_workers = sum(replicas_per_node)

    allocation = []

    workers_left = max_workers

    while workers_left > 0:
        count = (min(replicas_per_node[0], workers_left))
        if count:
            allocation += [(nodes[0]) * count]
            if count == replicas_per_node[0]:
                nodes = nodes[1:]
                replicas_per_node = replicas_per_node[1:]
            workers_left -= count
        else:
            nodes = nodes[1:]
            replicas_per_node = replicas_per_node[1:]

    speedup_fn = job_info.speedup_fn
    base_speedup = speedup_fn(1, 1)

    workers_arr = np.asarray(range(1, max_workers+1))
    speedups = speedup_fn(workers_arr, workers_arr)

    best_replicas = 0
    best_speedup = 0.0
    nodes_used = set()
    current_replicas = len(job._worker_tasks)

    current_speedup = speedups[current_replicas - 1]
    for worker, speedup in enumerate(speedups):
        nodes_used.add(allocation[worker])
        num_nodes = len(nodes_used)
        if (speedup / num_nodes >= base_speedup * 0.5):
            best_replicas = worker + 1
            best_speedup = speedup

    if (best_speedup < current_speedup * 1.05 or
            (abs(best_replicas + 1 - current_replicas) <
             0.15 * current_replicas)):
        best_replicas = current_replicas

    return allocation[:best_replicas]
