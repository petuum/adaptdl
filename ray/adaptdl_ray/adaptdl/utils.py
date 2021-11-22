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
from collections import Counter, defaultdict
from copy import deepcopy
from ray import tune
from ray.util.placement_group import get_current_placement_group
from adaptdl_ray.adaptdl import config


def pgf_to_allocation(pgf) -> List[str]:
    """ Convert a Placement Groups Factory to AdaptDL allocation"""
    bundles = pgf._bundles[1:]
    allocs, node_keys, num_devices = [], [], []
    for bundle in bundles:
        node_keys += [k.split(":")[1] for k, v in bundle.items()
                      if k.startswith("node")]
        num_devices += [int(v) for k, v in bundle.items()
                        if k == config.default_device()]

    for node, count in zip(node_keys, num_devices):
        allocs += [node] * count
    return allocs


def allocation_to_pgf(alloc: List[str], resources_per_node=None):
    """ Convert AdaptDL allocation to a Placement Group Factory"""
    if not resources_per_node:
        resources_per_node = {"CPU": 1.0}
        if config.default_device() == "GPU":
            resources_per_node["GPU"] = 1.0

    def _construct_bundle(node, number_of_instances):
        resources = deepcopy(resources_per_node)
        resources["CPU"] *= number_of_instances
        if "GPU" in resources:
            resources["GPU"] *= number_of_instances
        if "adaptdl_virtual" not in node:
            resources[f"node:{node}"] = 0.01
        return resources

    assert len(alloc) > 0
    resources = [{"CPU": 0.001}]
    alloc = Counter(alloc)
    for node, res in alloc.items():
        resources.append(_construct_bundle(node, res))
    return tune.PlacementGroupFactory(resources)


def pgf_to_num_replicas(pgf) -> int:
    """ Extract the number of replicas of the trial from its PGF"""
    return sum(int(bundle.get(config.default_device(), 0))
               for bundle in pgf._bundles[1:])


def pgs_to_resources(pgs: List[Dict]) -> Dict:
    """ Return node-level resource usage by all PGs in pgs."""
    # Note that every bundle is tagged with the node resource
    resources = defaultdict(Counter)
    for pg in pgs:
        for bundle in pg["bundle_cache"][1:]:
            # Every bundle has a node resource
            node_ip = [k.split(":")[1] for k in bundle.keys()
                       if k.startswith("node")][0]
            for k, v in bundle.items():
                resources[node_ip][k] += v
    return resources


def unique_nodes_pg() -> int:
    nodes = []
    if get_current_placement_group() is None:
        return 0
    else:
        for bundle in get_current_placement_group().bundle_specs:
            for resource in bundle:
                if "node" in resource:
                    nodes.append(resource)
        return len(set(nodes))
