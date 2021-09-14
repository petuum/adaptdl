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
import sys

from typing import Callable, Dict, Generator, Optional, Type

import ray
from ray import tune
from ray.tune.integration.torch import DistributedTrainableCreator
from adaptdl_ray.tune.adaptdl_trial_sched import AdaptDLScheduler
from adaptdl_ray.adaptdl import AdaptDLAllocator
from adaptdl_ray.tune.adaptdl_trainable import _train_simple


class IncrAllocator(AdaptDLAllocator):
    """Increment allocation by 1 starting with 1"""
    __test__ = False
    def __init__(self):
        super().__init__()
        self._avail_cpus = int(list(self._nodes.values())[0].resources["CPU"])
        self._cur_cpus = 1
    
    def default_allocation(self, num_devices=1):
        """ Use one device from the first node as default."""
        return [f"{list(self._nodes)[0]}"] * num_devices

    def allocate(self, jobs):
        if jobs[0]._num_replicas == self._cur_cpus:
            self._cur_cpus = min(self._cur_cpus + 1, self._avail_cpus)
        return {jobs[0].job_id: self.default_allocation(self._cur_cpus)}, 0


class DecrAllocator(AdaptDLAllocator):
    """Decrement allocation by 1 starting with max"""
    __test__ = False
    def __init__(self):
        super().__init__()
        self._avail_cpus = int(list(self._nodes.values())[0].resources["CPU"])
        self._cur_cpus = self._avail_cpus 
    
    def default_allocation(self, num_devices=None):
        """ Use one device from the first node as default."""
        if num_devices is None:
            num_devices = self._avail_cpus
        return [f"{list(self._nodes)[0]}"] * num_devices

    def allocate(self, jobs):
        if jobs[0]._num_replicas == self._cur_cpus:
            self._cur_cpus = max(self._cur_cpus - 1, 1)
        return {jobs[0].job_id: self.default_allocation(self._cur_cpus)}, 0

EPOCHS = 60
NUM_CPUS_CLUSTER = 4

class MyTest(unittest.TestCase):
    def setUp(self):
        ray.init(num_cpus=NUM_CPUS_CLUSTER, include_dashboard=False)

    def tearDown(self):
        ray.shutdown()

    def testSchedulerDecr(self):
        trainable_cls = DistributedTrainableCreator(_train_simple)
        analysis = tune.run(
            trainable_cls,
            name="Decr",
            num_samples=1,
            scheduler=AdaptDLScheduler(DecrAllocator()),
            config={"epochs": EPOCHS},
            metric="mean_loss",
            mode="min")
        assert len(analysis.results_df) == 1
        assert analysis._checkpoints[0]["last_result"]["training_iteration"] >= EPOCHS
    
    def testSchedulerIncr(self):
        trainable_cls = DistributedTrainableCreator(_train_simple)
        analysis = tune.run(
            trainable_cls,
            name="Incr",
            num_samples=1,
            scheduler=AdaptDLScheduler(IncrAllocator()),
            config={"epochs": EPOCHS},
            metric="mean_loss",
            mode="min")
        assert len(analysis.results_df) == 1
        assert analysis._checkpoints[0]["last_result"]["training_iteration"] >= EPOCHS

if __name__ == "__main__":
    import pytest, sys

    sys.exit(pytest.main(["-v", __file__]))
