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


from typing import Dict, List
from collections import defaultdict
import ray

_DEFAULT_DEVICE = None

# Job-level replica bounds
_JOB_MIN_REPLICAS = 0
_JOB_MAX_REPLICAS = 10


def _avail_nodes() -> Dict:
    """ We return all live nodes with their allocatable resources."""
    alive_nodes = {node["NodeID"]: node for node in
                   sorted(ray.nodes(), key=lambda x: x['NodeID'])
                   if node["alive"]}
    for node_id, resources in ray.state.state.\
            _available_resources_per_node().items():
        pruned_resources = defaultdict(float, {k: v for k, v in
                                       resources.items() if "group" not in k})
        alive_nodes[node_id]["Resources"] = pruned_resources
    return alive_nodes


def default_device(refresh=False) -> str:
    """ Default device will be GPU if at least one node has a GPU on it else we
    use CPUs. It is initialized once when the allocator/scheduler is
    instantiated."""
    global _DEFAULT_DEVICE
    if _DEFAULT_DEVICE is None or refresh:
        assert ray.is_initialized()
        if any("GPU" in node['Resources'] for node in _avail_nodes().values()):
            _DEFAULT_DEVICE = "GPU"
        else:
            _DEFAULT_DEVICE = "CPU"
    return _DEFAULT_DEVICE


def job_resources():
    """ Default job resources."""
    return {default_device(): 1, "memory": 1024*1024}


def nodes(consumed_resources: Dict = None) -> List[Dict]:
    """Returns all nodes with the default device on them"""
    avail_nodes = _avail_nodes()
    if consumed_resources is not None:
        # Add resources we are consuming already to the available resources
        node_id_map = {node["NodeID"]: node["NodeManagerAddress"] for node in
                       ray.nodes() if node["alive"]}
        for node_id, node in avail_nodes.items():
            resources = consumed_resources.get(node_id_map[node_id], {})
            for resource, amount in resources.items():
                node["Resources"][resource] += amount
    return [node for node in avail_nodes.values()
            if default_device() in node["Resources"]]
