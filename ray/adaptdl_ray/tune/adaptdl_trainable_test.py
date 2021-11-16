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


import pytest
import torch.distributed as dist
import ray

from .adaptdl_trainable import AdaptDLTrainableCreator, _train_simple


@pytest.fixture
def ray_start_2_cpus():
    address_info = ray.init(num_cpus=2)
    yield address_info
    # The code after the yield will run as teardown code.
    ray.shutdown()
    # Ensure that tests don't ALL fail
    if dist.is_initialized():
        dist.destroy_process_group()


def test_single_step(ray_start_2_cpus):  # noqa: F811
    trainable_cls = AdaptDLTrainableCreator(_train_simple, num_workers=2)
    trainer = trainable_cls()
    trainer.train()
    trainer.stop()


def test_step_after_completion(ray_start_2_cpus):  # noqa: F811
    trainable_cls = AdaptDLTrainableCreator(_train_simple, num_workers=2)
    trainer = trainable_cls(config={"epochs": 1})
    with pytest.raises(RuntimeError):
        for i in range(10):
            trainer.train()


def test_save_checkpoint_restore(ray_start_2_cpus):  # noqa: F811
    trainable_cls = AdaptDLTrainableCreator(_train_simple, num_workers=2)
    trainer = trainable_cls(config={"epochs": 1})
    trainer.train()
    checkpoint_dir = trainer.save()
    trainer.restore(checkpoint_dir)
    trainer.train()
    trainer.stop()
