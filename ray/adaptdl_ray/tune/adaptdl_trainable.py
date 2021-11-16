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


from typing import Callable, Dict, Optional
from unittest.mock import patch
from datetime import timedelta

import ray
from ray.tune.resources import Resources
from ray.tune.registry import register_trainable
from ray.tune.integration.torch import _TorchTrainable
from ray.util.sgd.torch.constants import NCCL_TIMEOUT_S

import adaptdl_ray.tune.adaptdl_patch as P
from adaptdl_ray.adaptdl import config


def AdaptDLTrainableCreator(func: Callable,
                            num_workers: int = 1,
                            group: int = 0,
                            num_cpus_per_worker: int = 1,
                            num_workers_per_host: Optional[int] = None,
                            backend: str = "gloo",
                            timeout_s: int = NCCL_TIMEOUT_S,
                            use_gpu=None):
    """ Trainable creator for AdaptDL's elastic Trials"""
    if config.default_device() == "GPU":
        backend = "nccl"

    class AdaptDLTrainable(_TorchTrainable):
        """ Similar to DistributedTrainable but for AdaptDLTrials."""
        def setup(self, config: Dict):
            """ Delay-patch methods when the Trainable actors are first
            created"""
            with patch(target="ray.tune.integration.torch.setup_process_group",
                       new=P.setup_process_group), \
                 patch(target='ray.tune.integration.torch.wrap_function',
                       new=P.wrap_function_patched):
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
    # Set number of GPUs if we're using them, this is later used when spawning
    # the trial actors
    if config.default_device() == "GPU":
        AdaptDLTrainable._num_gpus_per_worker = 1
    else:
        AdaptDLTrainable._num_gpus_per_worker = 0

    # Trainables are named after number of replicas they spawn. This is
    # essential to associate the right Trainable with the right Trial and PG.
    AdaptDLTrainable.__name__ = AdaptDLTrainable.__name__.split("_")[0] + \
        f"_{num_workers}" + f"_{group}"
    register_trainable(AdaptDLTrainable.__name__, AdaptDLTrainable)
    return AdaptDLTrainable


# To support unit tests and integration tests
def _train_simple(config: Dict, checkpoint_dir: Optional[str] = None):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import adaptdl.torch as adl
    from ray import tune

    class MyDataset:
        def __init__(self, xs, ys):
            self.xs = xs
            self.ys = ys

        def __getitem__(self, i):
            return self.xs[i], self.ys[i]

        def __len__(self):
            return len(self.xs)

    # N is batch size; D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
    N, D_in, H, D_out = 64, 5, 5, 5
    dataset = MyDataset(torch.randn(N, D_in), torch.randn(N, D_out))

    H = config.get("H", 16)
    N = config.get("N", 16)

    # Create random Tensors to hold inputs and outputs
    dataloader = adl.AdaptiveDataLoader(dataset, batch_size=N)
    dataloader.autoscale_batch_size(4096, local_bsz_bounds=(16, 1024))

    loss_fn = nn.MSELoss()

    # Use the nn package to define our model and loss function.
    model = torch.nn.Sequential(
        torch.nn.Linear(D_in, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, D_out),
    )
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    model = adl.AdaptiveDataParallel(model, optimizer)

    loss = torch.Tensor([0.0])
    for epoch in adl.remaining_epochs_until(config.get("epochs", 10)):
        for (x, y) in dataloader:
            optimizer.zero_grad()
            output = model(x)
            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()

        tune.report(mean_loss=loss.item())
