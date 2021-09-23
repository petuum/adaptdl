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

import ray
from ray.tune.schedulers import TrialScheduler
from ray.tune.trial import Trial

from adaptdl_ray.tune.adaptdl_trial import AdaptDLTrial
from adaptdl_ray.adaptdl import AdaptDLAllocator
from adaptdl_ray.adaptdl import config

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class AdaptDLScheduler(TrialScheduler):
    """AdaptDL TrialScheduler."""

    _ALLOCATOR_INVOKE_FREQ = 30

    @staticmethod
    def _try_realloc(iteration):
        return iteration % AdaptDLScheduler._ALLOCATOR_INVOKE_FREQ == 0

    def __init__(self, allocator=None):
        nodes = config.nodes()
        # Reserve 1 CPU from the first node for the Trainable
        nodes[0]["Resources"]["CPU"] -= 1
        self._allocator = allocator if allocator is not None else AdaptDLAllocator(nodes)

    def on_trial_add(self, trial_runner: "trial_runner.TrialRunner",
                     trial: Trial):
        """Called after SearchAlgorithm.next_trial"""
        trial = AdaptDLTrial.create_from(trial, trial_runner, 
                                         self._allocator.default_allocation())

    def on_trial_error(self, trial_runner: "trial_runner.TrialRunner",
                       trial: Trial):
        pass

    def on_trial_result(self, trial_runner: "trial_runner.TrialRunner",
                        trial: Trial, result: Dict) -> str:
        trials = [trial for trial in trial_runner.get_trials() \
                    if trial.status in (Trial.RUNNING, Trial.PENDING)]
        if AdaptDLScheduler._try_realloc(result.get('training_iteration', 1)):
            allocs, desired = self._allocator.allocate(trials)
        else:
            allocs = {trial.trial_id: trial.allocation for trial in trials}

        if allocs.get(trial.trial_id) == [] and trial.status == Trial.RUNNING:
            # Pause only if the trial is running
            trial.pause(trial_runner)
            return TrialScheduler.PAUSE
        elif allocs.get(trial.trial_id) != trial.allocation:
            trial = AdaptDLTrial.create_from(trial, 
                                             trial_runner, 
                                             allocs.get(trial.trial_id),
                                             copy_state=True)
            # Stop the old trial that's being replaced
            return TrialScheduler.STOP
        return TrialScheduler.CONTINUE

    def on_trial_complete(self, trial_runner: "trial_runner.TrialRunner",
                          trial: Trial, result: Dict):
        pass

    def on_trial_remove(self, trial_runner: "trial_runner.TrialRunner",
                        trial: Trial):
        pass

    def choose_trial_to_run(
            self, trial_runner: "trial_runner.TrialRunner") -> Optional[Trial]:
        for trial in trial_runner.get_trials():
            if (trial.status == Trial.PENDING
                    and trial_runner.has_resources_for_trial(trial)):
                return trial
        for trial in trial_runner.get_trials():
            if (trial.status == Trial.PAUSED
                    and trial_runner.has_resources_for_trial(trial)):
                # Note: this puts the trial back to RUNNING. Need the new trial
                # to have the old chekcpoint
                # TODO: invoke the allocator here?
                return AdaptDLTrial.create_from(trial, 
                                                trial_runner, 
                                                self._allocator.default_allocation(),
                                                copy_state=True)
        return None

    def debug_string(self) -> str:
        return "Using AdaptDL scheduling algorithm."
