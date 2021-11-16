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


from itertools import cycle
from typing import Dict, List
from adaptdl_sched.policy.pollux import PolluxPolicy
from adaptdl_sched.policy.utils import NodeInfo
from adaptdl_ray.adaptdl.adaptdl_job_mixin import AdaptDLJobMixin
from adaptdl_ray.adaptdl import config


class AdaptDLAllocator:
    def __init__(self, nodes: List = None):
        nodes = nodes if nodes is not None else config.nodes()
        self._node_infos = {node['NodeManagerAddress']:
                            NodeInfo(node['Resources'], preemptible=False)
                            for node in nodes}
        self._default_node = cycle(list(self._node_infos))
        # Add a node template.
        self._node_template = NodeInfo(list(self._node_infos.values())[0].
                                       resources, preemptible=False)
        self._policy = PolluxPolicy()

    def default_allocation(self, num_devices=1) -> List[str]:
        """ Cycle through nodes for default trial allocation."""
        return [f"{next(self._default_node)}"] * num_devices

    def allocate(self,
                 jobs: List[AdaptDLJobMixin],
                 nodes: List = None) -> (Dict, int):
        """ Use Pollux to distribute available resources between jobs."""
        if nodes is not None:
            node_infos = {node['NodeManagerAddress']:
                          NodeInfo(node['Resources'], preemptible=False)
                          for node in nodes}
        else:
            node_infos = self._node_infos

        assert len(jobs) > 0
        # gather JobInfos
        job_infos = {job.job_id: job.job_info for job in jobs}
        # gather previous allocations
        prev_allocs = {job.job_id: job.allocation for job in jobs}

        allocations, desired_nodes = \
            self._policy.optimize(job_infos,
                                  node_infos,
                                  prev_allocs,
                                  self._node_template)
        # Fill empty allocations for jobs which didn't get any
        for job_id in job_infos:
            allocations[job_id] = allocations.get(job_id, [])

        assert all(v == [] for k, v in allocations.items()) is False
        return allocations, desired_nodes
