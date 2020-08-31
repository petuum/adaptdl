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


"""
This module provides functionality to Save and load arbitrary state as part of
checkpoint-restart elasticity. The `State` class can be subclassed to define
how to save/load any state to/from persistent storage, so it can be restored
after the current job restarts and resumed from where it left off.
"""

import os

from adaptdl.env import checkpoint_path, replica_rank

# FIXME: Keeping global state like this will result in memory leaks for
# applications which do not restart too often.
_STATES_TO_NAMES = {}
_NAMES_TO_STATES = {}


class State(object):
    """
    This class implements An arbitrary piece of state which can be saved and
    loaded as part of a checkpoint, and synchronized across all replicas.
    Should be sub-classed to define custom save, load, and sync logic.
    """

    def __init__(self, name):
        """
        Initialize the state object with a unique identifier `name`, which is
        used to refer to the saved object in persistent storage. No two `State`
        objects may share the same `name`.

        Arguments:
            name (str): Unique name of this `State` object.

        Raises:
            ValueError: If a `State` object with the given name already exists.
        """
        if name in _NAMES_TO_STATES:
            raise ValueError("State '{}' already exists".format(name))
        _NAMES_TO_STATES[name] = self
        _STATES_TO_NAMES[self] = name

    def save(self, fileobj):
        """
        This method should be overridden by subclasses to define how the state
        is saved. Is invoked by `save_all_states` and `save_state` to save the
        state into persistent storage.

        Arguments:
            fileobj (BinaryIO): A binary writable file object.
        """
        pass

    def load(self, fileobj):
        """
        This method should be overridden by subclasses to define how the state
        is loaded. Is invoked by `load_state` to load the state from persistent
        storage.

        Arguments:
            fileobj (BinaryIO): A binary readable file object.
        """
        pass

    def sync(self):
        """
        This method should be overridden by subclasses to define how the state
        is synchronized across replicas. This might be necessary to make sure
        the state is consistent before saving it to persistent storage. Is
        invoked by `save_state` before saving the state.
        """
        pass


def save_all_states():
    """
    Invokes `save_state` on all `State` objects for which `State.skip` is True.
    This function can be used to trigger a global checkpoint and save every
    `State` in the current job.
    """
    for state in _STATES_TO_NAMES:
        save_state(state)


def save_state(state, sync=True):
    """
    Saves a `State` object to persistent storage. First invokes `State.sync` on
    all replicas if `sync` is `True` (default), and then invokes `State.save`
    on the replica of rank 0 only.

    Arguments:
        state (State): The `State` object to save to persistent storage.
        sync (bool): Whether `State.sync` should be invoked.
    """
    if sync:
        state.sync()
    if replica_rank() == 0:
        name = _STATES_TO_NAMES[state]
        if checkpoint_path() is not None:
            with open(os.path.join(checkpoint_path(), name), "wb") as f:
                state.save(f)


def load_state(state):
    """
    Load the given `State` object from persistent storage. If the object was
    previously saved, then State.load will be invoked with a readable file
    object to load from.

    Arguments:
        state (State): `State` object to load from persistent storage.

    Returns:
        `True` if state was previously saved and `State.load` was invoked,
        `False` otherwise.
    """
    if checkpoint_path() is None:
        return False
    try:
        name = _STATES_TO_NAMES[state]
        with open(os.path.join(checkpoint_path(), name), "rb") as f:
            state.load(f)
        return True
    except FileNotFoundError:
        return False
