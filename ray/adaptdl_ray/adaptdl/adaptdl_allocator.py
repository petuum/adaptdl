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


from typing import Dict, List, Optional, Union
from collections import Counter
from adaptdl.goodput import GoodputFunction, PerfParams, GradParams
from adaptdl_sched.policy.pollux import PolluxPolicy
from adaptdl_sched.policy.utils import JobInfo, NodeInfo
from adaptdl_ray.adaptdl.adaptdl_job_mixin import AdaptDLJobMixin
from adaptdl_ray.adaptdl import config
import ray


class AdaptDLAllocator:
    def __init__(self, nodes=None):
        nodes = nodes if nodes is not None else config.nodes()
        num_nodes = len(nodes)
        self._nodes = {node['NodeManagerAddress']: NodeInfo(node['Resources'], 
                       preemptible=False) for i, node in enumerate(nodes)}
        # Add a node template.
        self._node_template = NodeInfo(list(self._nodes.values())[0].resources, 
                                       preemptible=False)
        self._policy = PolluxPolicy()

    def default_allocation(self, num_devices=1):
        """ Use one device from the first node as default."""
        return [f"{list(self._nodes)[0]}"] * num_devices

    def allocate(self, jobs: List[AdaptDLJobMixin], nodes=None):
        assert len(jobs) > 0
        # gather JobInfos
        job_infos = {job.job_id: job.job_info for job in jobs}
        # gather previous allocations
        prev_allocs = {job.job_id: job.allocation for job in jobs}

        allocations, desired_nodes = \
                self._policy.optimize(job_infos, 
                                      self._nodes, 
                                      prev_allocs, 
                                      self._node_template)
       
        assert all(v == [] for k, v in allocations.items()) == False
        return allocations, desired_nodes
