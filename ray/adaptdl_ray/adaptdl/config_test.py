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


import unittest
import ray
from ray.cluster_utils import Cluster
from .config import nodes, default_device
from ray.util.placement_group import placement_group


class TrialRunnerTest(unittest.TestCase):
    def setUp(self):
        self.cluster = Cluster(
            initialize_head=True,
            connect=True,
            head_node_args={
                "num_cpus": 4,
                "num_gpus": 1,
            })
        self.cluster.add_node(num_cpus=2, num_gpus=1)
        self.cluster.wait_for_nodes()

    def tearDown(self):
        ray.shutdown()
        self.cluster.shutdown()

    def testAvailableResources(self):
        assert len(nodes()) == 2
        assert default_device(refresh=True) == "GPU"

    def testOthersTakingResources(self):
        # Let someone occupy the head node
        pg = placement_group([{"CPU": 4, "GPU": 1}])
        ray.get(pg.ready())
        # We are left with the second node
        assert len(nodes()) == 1
        assert default_device(refresh=True) == "GPU"

        pg = placement_group([{"GPU": 1}])
        ray.get(pg.ready())
        # Default device should be CPU
        assert default_device(refresh=True) == "CPU"
        assert len(nodes()) == 1
