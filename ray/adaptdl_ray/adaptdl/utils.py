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
from ray import tune
from adaptdl_ray.adaptdl import config


def pgf_to_allocation(pgf) -> List[str]:
    """ Convert Placement Groups Factory to AdaptDL allocations"""
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


def allocation_to_pgf(alloc: List[str]):
    """ Convert AdaptDL allocations to Placement Group Factory"""
    def _construct_bundle(node, device_count):
        resources = {config.default_device(): device_count,
                     f"node:{node}": 0.01}
        if config.default_device() == "GPU":
            # As per Ray, We need equal amount of CPUs if there are GPUs in
            # this bundle
            resources["CPU"] = device_count
        return resources

    assert len(alloc) > 0
    resources = [{"CPU": 0.001}]
    alloc = Counter(alloc)
    for node, res in alloc.items():
        resources.append(_construct_bundle(node, res))
    return tune.PlacementGroupFactory(resources)


def pgf_to_num_replicas(pgf) -> int:
    """ Extract number of replicas of the trial from its PGF"""
    return sum(int(bundle.get(config.default_device(), 0))
               for bundle in pgf._bundles[1:])


def pgs_to_resources(pgs: List[Dict]) -> Dict:
    """ Returns node-level resource usage by all PGs in pgs."""
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
