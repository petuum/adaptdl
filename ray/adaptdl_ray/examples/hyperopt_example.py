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


from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torch.nn.functional as F

import ray
from ray import tune
from ray.tune.integration.torch import DistributedTrainableCreator
from ray.tune.suggest.hyperopt import HyperOptSearch

import adaptdl.torch as adl
from adaptdl_ray.tune.adaptdl_trial_sched import AdaptDLScheduler

from hyperopt import hp

# Adapted from https://docs.ray.io/en/latest/tune/tutorials/tune-tutorial.html


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # In this example, we don't change the model architecture
        # due to simplicity.
        self.conv1 = nn.Conv2d(1, 3, kernel_size=3)
        self.fc = nn.Linear(192, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 3))
        x = x.view(-1, 192)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


# Change these values if you want the training to run quicker or slower.
EPOCH_SIZE = 512
TEST_SIZE = 256


def train(model, optimizer, train_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # We set this just for the example to run quickly.
        if batch_idx * len(data) > EPOCH_SIZE:
            return
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()


def test(model, data_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            # We set this just for the example to run quickly.
            if batch_idx * len(data) > TEST_SIZE:
                break
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        else:
            return 0
    return correct / total


def train_mnist(config: Dict, checkpoint_dir: Optional[str] = None):
    # Data Setup
    mnist_transforms = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307, ), (0.3081, ))])

    train_loader = adl.AdaptiveDataLoader(datasets.MNIST("~/data", train=True,
                                          download=True,
                                          transform=mnist_transforms),
                                          batch_size=64,
                                          shuffle=True)

    # Autoscale batch size
    train_loader.autoscale_batch_size(4096, local_bsz_bounds=(16, 1024))

    test_loader = adl.AdaptiveDataLoader(
        datasets.MNIST("~/data", train=False, transform=mnist_transforms),
        batch_size=64,
        shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ConvNet()
    optimizer = optim.SGD(
        model.parameters(), lr=config.get("lr", 0.01),
        momentum=config.get("momentum", 0.79))

    model.to(device)
    model = adl.AdaptiveDataParallel(model, optimizer)

    for epoch in adl.remaining_epochs_until(config.get("epochs", 10)):
        train(model, optimizer, train_loader)
        acc = test(model, test_loader)
        # Send the current training result back to Tune
        tune.report(mean_accuracy=acc)


ray.init(address="auto")

trainable_cls = DistributedTrainableCreator(train_mnist)

space = {
    "lr": hp.uniform("lr", 0.01, 0.1),
    "momentum": hp.uniform("momentum", 0.1, 0.9),
    "epochs": 100
}

hyperopt_search = HyperOptSearch(space, metric="mean_accuracy", mode="max")

analysis = tune.run(
    trainable_cls,
    num_samples=16,  # total trials will be num_samples x points on the grid
    scheduler=AdaptDLScheduler(),
    search_alg=hyperopt_search)

best_trial = analysis.get_best_trial("mean_accuracy", "min")
print(f"Best trial config: {best_trial.config}")
print(f"Best trial mean_accuracy: {best_trial.last_result['mean_accuracy']}")
