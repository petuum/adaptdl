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


import logging, sys
from typing import Callable, Dict, Generator, Optional, Type

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
import torch.optim as optim
from torch.utils.data import DataLoader

import ray
from ray import tune
from ray.tune.integration.torch import DistributedTrainableCreator

import adaptdl.torch as adl
from adaptdl_ray.tune.adaptdl_trial_sched import AdaptDLScheduler


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
N, D_in, _, D_out = 64, 5, 5, 5

dataset = MyDataset(torch.randn(N, D_in), torch.randn(N, D_out))

def _train_simple(config: Dict, checkpoint_dir: Optional[str] = None):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
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

    model = model.to(device)
    model = adl.AdaptiveDataParallel(model, optimizer)

    loss = torch.Tensor([0.0])
    for epoch in adl.remaining_epochs_until(config.get("epochs", 10)):
        for (x, y) in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = loss_fn(output, y)
            loss.backward()
            optimizer.step()

        tune.report(mean_loss=loss.item())


ray.init(address="auto", _tracing_startup_hook=None)

trainable_cls = DistributedTrainableCreator(_train_simple)

config_0 = {"epochs": 60}
config_1 = {"epochs": 60, "H": tune.choice([8, 12]), "N": tune.grid_search(list(range(32, 64, 8)))}

analysis = tune.run(
    trainable_cls,
    num_samples=1, # total trials will be num_samples x points on the grid
    scheduler=AdaptDLScheduler(),
    config=config_1,
    metric="mean_loss",
    mode="min")

