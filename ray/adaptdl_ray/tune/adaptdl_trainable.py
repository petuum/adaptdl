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


from typing import Callable, Dict, Generator, Optional, Type
from unittest.mock import patch
from datetime import timedelta

import ray
from ray.tune.resources import Resources
from ray.tune.registry import get_trainable_cls, register_trainable
from ray.tune.integration.torch import _TorchTrainable
from ray.util.sgd.torch.constants import NCCL_TIMEOUT_S

import adaptdl_ray.tune.adaptdl_patch as P 


def AdaptDLTrainableCreator(func: Callable,
                            num_workers: int = 1,
                            group: int = 0,
                            num_cpus_per_worker: int = 1,
                            num_gpus_per_worker: int = 0,
                            num_workers_per_host: Optional[int] = None,
                            backend: str = "gloo",
                            timeout_s: int = NCCL_TIMEOUT_S,
                            use_gpu=None):
    class AdaptDLTrainable(_TorchTrainable):
        """ Similar to DistributedTrainable but for AdaptDLTrials."""
        def setup(self, config: Dict):
            """ Delay-patch methods when the Trainable actors are first initialized"""
            with patch(target="ray.tune.integration.torch.setup_process_group", new=P.setup_process_group), \
                patch(target='ray.tune.integration.torch.wrap_function', new=P.wrap_function_patched):
                    _TorchTrainable.setup(self, config)

        # Override the default resources and use custom PG factory
        @classmethod
        def default_resource_request(cls, config: Dict) -> Resources:
            return None

        def get_sched_hints(self):
            return ray.get(self.workers[0].get_sched_hints.remote())

        def save_all_states(self, trial_state):
            return ray.get(self.workers[0].save_all_states.remote(trial_state))

        @classmethod
        def default_process_group_parameters(self) -> Dict:
            return dict(timeout=timedelta(timeout_s), backend=backend)

    AdaptDLTrainable._function = func
    AdaptDLTrainable._num_workers = num_workers

    # Trainables are named after number of replicas they spawn. This is
    # essential to associate the right Trainable with the right Trial and PG.
    AdaptDLTrainable.__name__ = AdaptDLTrainable.__name__.split("_")[0] \
                                + f"_{num_workers}" + f"_{group}"
    
    register_trainable(AdaptDLTrainable.__name__, AdaptDLTrainable)
    return AdaptDLTrainable
