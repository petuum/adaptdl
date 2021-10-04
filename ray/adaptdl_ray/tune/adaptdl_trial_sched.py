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
from adaptdl_ray.adaptdl.utils import pgs_to_resources

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class AdaptDLScheduler(TrialScheduler):
    """AdaptDL TrialScheduler."""

    _ALLOCATOR_INVOKE_FREQ = 100

    def _try_realloc(self):
        self._iterations += 1
        return self._iterations % AdaptDLScheduler._ALLOCATOR_INVOKE_FREQ == 0

    def __init__(self, allocator=None):
        # Reserve 1 CPU from the first node for the Trainables
        consumed_resources = {config.nodes()[0]["NodeManagerAddress"]: {"CPU": -1.0}}
        self._allocator = allocator if allocator is not None \
                else AdaptDLAllocator(config.nodes(consumed_resources))
        self._allocs = {}
        self._iterations = 0

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
        trials = [trial for trial in trial_runner.get_trials()
                  if trial.status in (Trial.RUNNING, Trial.PENDING)]
        if self._try_realloc() and len(self._allocs) == 0:
            in_use_pgs = [pg.to_dict() for pg in 
                          trial_runner.trial_executor._pg_manager._in_use_pgs]
            consumed_resources = pgs_to_resources(in_use_pgs)
            nodes = config.nodes(consumed_resources)
            self._allocs, _ = self._allocator.allocate(trials, nodes)

        alloc = self._allocs.pop(trial.trial_id, None)
        if alloc is None:
            return TrialScheduler.CONTINUE

        if alloc == [] and trial.status == Trial.RUNNING:
            # Pause only if the trial is running
            trial.pause(trial_runner)
            return TrialScheduler.PAUSE
        elif alloc != trial.allocation:
            trial = AdaptDLTrial.create_from(trial, trial_runner, alloc, copy_state=True)
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
                and trial_runner.has_resources_for_trial(trial)
                and len(self._allocs) == 0):
                # Note: this puts the trial back to RUNNING
                return AdaptDLTrial.create_from(trial, 
                                                trial_runner, 
                                                self._allocator.default_allocation(),
                                                copy_state=True)
        return None

    def debug_string(self) -> str:
        return "Using AdaptDL scheduling algorithm."
