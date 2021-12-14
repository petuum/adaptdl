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


from datetime import datetime
import logging
from typing import List

import ray
from ray.tune import trial_runner
from ray.tune.trial import Trial
from ray.tune import PlacementGroupFactory
from ray.tune.function_runner import FuncCheckpointUtil
from ray.tune.trainable import TrainableUtil

from adaptdl_ray.adaptdl import AdaptDLJobMixin
from adaptdl_ray.tune.adaptdl_trainable import AdaptDLTrainableCreator
from adaptdl_ray.adaptdl.utils import pgf_to_num_replicas, allocation_to_pgf

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class AdaptDLTrial(AdaptDLJobMixin, Trial):
    """ Tune Trial that brings in AdaptDL functionality. """
    def __init__(self, *args, **kwargs):
        self.rescale_count = kwargs.pop("rescale_count", 0)
        self._cached_metrics = None
        super().__init__(job_id=kwargs["trial_id"], *args, **kwargs)

    @property
    def _num_replicas(self) -> int:
        return self.get_trainable_cls()._num_workers

    def __getstate__(self):
        copy_state = {}
        # Remove problematic members
        for k in ("_trial_in_use", "_cached_metrics"):
            copy_state[k] = self.__dict__[k]
            del self.__dict__[k]
        state = super().__getstate__()
        # Restore members
        for k, v in copy_state.items():
            self.__dict__[k] = v
        return state

    def _requeue(self,
                 old_trial: Trial,
                 trial_runner: "trial_runner.TrialRunner"):
        # Remove the old trial from trial_runner
        trial_runner.trial_executor.stop_trial(old_trial)
        trial_runner._trials.pop(trial_runner._trials.index(old_trial))
        # Important: Add the new trial to the runner
        trial_runner._trials.append(self)
        trial_runner._live_trials.add(self)

    def _fetch_metrics(self):
        if self.runner is not None:
            metrics = ray.get(self.runner.get_sched_hints.remote())
            if metrics is not None:
                # Update cache
                self._cached_metrics = metrics
        return self._cached_metrics

    def _allocation_in_use(self):
        return self._trial_in_use(self)

    @classmethod
    def _clone_from(cls,
                    trial: Trial,
                    allocation,
                    restore_path=None) -> "AdaptDLTrial":
        trainable_cls = trial.get_trainable_cls()
        pgf = allocation_to_pgf(allocation)
        num_workers = pgf_to_num_replicas(pgf)
        assert num_workers > 0
        if isinstance(trial, AdaptDLTrial):
            # Cloning from existing AdaptDLTrial
            rescale_count = trial.rescale_count + 1
            # Carry over the creation_timestamp
            creation_timestamp = trial.creation_timestamp
        else:
            creation_timestamp = datetime.now()
            rescale_count = 0

        adaptdl_trainable_cls = AdaptDLTrainableCreator(trainable_cls.
                                                        _function,
                                                        num_workers,
                                                        group=rescale_count)
        return cls(trainable_name=adaptdl_trainable_cls.__name__,
                   creation_timestamp=creation_timestamp,
                   rescale_count=rescale_count,
                   config=trial.config,
                   experiment_tag=trial.experiment_tag,
                   evaluated_params=trial.evaluated_params,
                   stopping_criterion=trial.stopping_criterion,
                   trial_id=trial.trial_id,
                   restore_path=restore_path,
                   local_dir="/tmp",  # TODO: Decide a proper way
                   placement_group_factory=pgf)

    @classmethod
    def create_from(cls,
                    trial: Trial,
                    trial_runner: "trial_runner.TrialRunner",
                    new_allocation: List[str],
                    copy_state=False) -> "AdaptDLTrial":
        """ Create a new AdaptDLTrial from a Trial or AdaptDLTrial with new
        allocations. This also replaces the existing Trial."""
        checkpoint_path = None
        logger.debug(f"Creating {trial} with {len(new_allocation)} replicas")
        if copy_state:
            if trial.runner is not None:
                # Fetch the state from the other trial
                checkpoint_obj = ray.get(trial.runner.save_all_states.remote(
                                         trial.runner.get_state.remote()))
                # Dump it to disk
                temp_checkpoint_dir = (FuncCheckpointUtil.
                                       mk_temp_checkpoint_dir(trial.logdir))
                checkpoint_path = TrainableUtil. \
                    create_from_pickle(checkpoint_obj, temp_checkpoint_dir)
            else:
                # trial was PAUSED
                checkpoint_path = trial.restore_path

        # Spawn a new trial
        new_trial = cls._clone_from(trial, new_allocation,
                                    restore_path=checkpoint_path)
        # Keep it for later use by the trials
        new_trial._trial_in_use = trial_runner.trial_executor.\
            _pg_manager.trial_in_use
        # Replace with old trial
        new_trial._requeue(trial, trial_runner)
        assert new_trial.restore_path == checkpoint_path
        assert new_trial.status == Trial.PENDING
        return new_trial

    def pause(self, trial_runner):
        """ Pause the AdaptDLTrial with a checkpoint. We try to remove the PG
        attached to this trial"""
        assert self.runner is not None
        checkpoint_obj = ray.get(self.runner.save_all_states.remote(
                                 self.runner.get_state.remote()))
        # Serialize to disk
        temp_checkpoint_dir = (FuncCheckpointUtil.
                               mk_temp_checkpoint_dir(self.logdir))
        checkpoint_path = TrainableUtil.create_from_pickle(checkpoint_obj,
                                                           temp_checkpoint_dir)

        # Trial will be restored from the checkpoint_path when it's resumed
        self.restore_path = checkpoint_path

        # Clear the allocation. This is a hack to clear the PG associated with
        # the trial. We assign a temporary PG which will get replaced with a
        # real PG once we resume the trial. This is needed because Tune likes
        # to keep the PGs around even for PAUSED trials.
        self.placement_group_factory = PlacementGroupFactory([{"CPU": 0.001}])
        # This forces Tune to garbage-collect uneeded PGs which can then be
        # reused
        trial_runner.trial_executor._pg_manager.\
            reconcile_placement_groups([self])
        logger.debug(f"PAUSING {self} w/ checkpoint at {checkpoint_path}")
