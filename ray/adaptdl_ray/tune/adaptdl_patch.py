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


import types
import shutil

from ray.tune.function_runner import wrap_function
from ray.tune.trainable import TrainableUtil

from adaptdl.torch import init_process_group
from adaptdl.checkpoint import save_all_states
from adaptdl.torch._metrics import _get_sched_hints


# Wrap the free functions of AdaptDL with FunctionRunner methods
def save_all_states_remote(self, trial_state):
    """ Save all of AdaptDL's job state and return it as an in-memory
    object."""
    checkpoint = save_all_states()
    parent_dir = TrainableUtil.find_checkpoint_dir(checkpoint)
    checkpoint_path = TrainableUtil.process_checkpoint(checkpoint,
                                                       parent_dir,
                                                       trial_state)
    checkpoint_obj = TrainableUtil.checkpoint_to_object(checkpoint_path)
    # Done with the directory, remove
    shutil.rmtree(checkpoint_path)
    return checkpoint_obj


def get_sched_hints_remote(self):
    """ Return hints for AdaptDL Scheduler."""
    return _get_sched_hints()


def wrap_function_patched(function):
    """ Monkey-patch FunctionRunner remote trainable"""
    func_trainable = wrap_function(function)
    func_trainable.save_all_states = types.MethodType(save_all_states_remote,
                                                      func_trainable)
    func_trainable.get_sched_hints = types.MethodType(get_sched_hints_remote,
                                                      func_trainable)
    return func_trainable


def setup_process_group(*args, **kwargs):
    init_process_group(backend=kwargs["backend"],
                       init_method=kwargs["url"],
                       world_size=kwargs["world_size"],
                       rank=kwargs["world_rank"])
