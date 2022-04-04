# Copyright 2020 Petuum, Inc. All Rights Reserved.
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


import numpy as np
import torch

from torch.utils.data import Dataset
import adaptdl.torch as adl


class LRIterableDataset(Dataset):
    def __init__(self, size, true_values, noise):
        input_values = np.random.uniform(-5.0, 5.0, size)
        bias_input_values = np.stack([np.ones(size), input_values])
        target_values = (
            np.dot(true_values, bias_input_values)
            + np.random.normal(0.0, noise, size=(size,)))
        self._values = list(zip(input_values, target_values))
        self._len = size

    def __getitem__(self, index):
        return self._values[index]

    def __len__(self):
        return self._len


def test_single_replica_parallel():
    adl.init_process_group("gloo")
    true_values = np.asarray([3.0, 4.0])
    dataset = LRIterableDataset(1000, true_values, 1.0)
    dataloader = adl.AdaptiveDataLoader(
        dataset, batch_size=32, shuffle=False, num_workers=1)
    model = torch.nn.Linear(1, 1, bias=True)
    params = [model.bias, model.weight]
    sgd = torch.optim.SGD(
        [{"params": [param]} for param in params],
        lr=0.01)
    schedule = torch.optim.lr_scheduler.MultiStepLR(sgd, [50])
    model = adl.AdaptiveDataParallel(model, sgd, schedule)
    loss = torch.nn.MSELoss()
    for epoch in adl.remaining_epochs_until(100):
        for inputs, targets in dataloader:
            inputs = inputs.float()
            targets = targets.float()
            sgd.zero_grad()
            output = model(torch.reshape(inputs, (-1, 1)))
            targets = torch.reshape(targets, (-1, 1))
            loss_value = loss(output, targets)
            loss_value.backward()
            sgd.step()
        schedule.step()
    params = np.asarray([param.item() for param in params])
    assert(np.all(np.isclose(params, true_values, atol=0.1))), \
        (params, true_values)
