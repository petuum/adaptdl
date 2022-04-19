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


import os
import unittest

import ray

from ray import tune
from ray.tune.ray_trial_executor import RayTrialExecutor
from ray.tune.suggest import BasicVariantGenerator
from .adaptdl_trainable import AdaptDLTrainableCreator, _train_simple


class TrialRunnerTest(unittest.TestCase):
    def setUp(self):
        # Wait up to five seconds for placement groups when starting a trial
        os.environ["TUNE_PLACEMENT_GROUP_WAIT_S"] = "5"
        # Block for results even when placement groups are pending
        os.environ["TUNE_TRIAL_STARTUP_GRACE_PERIOD"] = "0"
        os.environ["TUNE_TRIAL_RESULT_WAIT_TIME_S"] = "99999"

    def tearDown(self):
        ray.shutdown()

    def testExperimentTagTruncation(self):
        ray.init(num_cpus=2)
        trainable_cls = AdaptDLTrainableCreator(_train_simple, num_workers=1)
        trial_executor = RayTrialExecutor()
        experiments = {
            "foo": {
                "run": trainable_cls.__name__,
                "config": {
                    "a" * 50: tune.sample_from(lambda spec: 5.0 / 7),
                    "b" * 50: tune.sample_from(lambda spec: "long" * 40)
                },
            }
        }

        for name, spec in experiments.items():
            trial_generator = BasicVariantGenerator()
            trial_generator.add_configurations({name: spec})
            while not trial_generator.is_finished():
                trial = trial_generator.next_trial()
                if not trial:
                    break
                trial_executor.start_trial(trial)
                assert len(os.path.basename(trial.logdir)) <= 200
                trial_executor.stop_trial(trial)
