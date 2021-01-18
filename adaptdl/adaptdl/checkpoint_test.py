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


import pytest

from adaptdl.conftest import elastic_multiprocessing


@elastic_multiprocessing
def test_duplicate():
    from adaptdl.env import num_restarts
    from adaptdl.checkpoint import State
    state_1 = State("state_1")  # noqa: F841
    state_2 = State("state_2")  # noqa: F841
    with pytest.raises(ValueError):
        state_dup = State("state_1")  # noqa: F841
    return [2, 0][num_restarts()]


@elastic_multiprocessing
def test_save_load():
    import pickle
    from adaptdl.checkpoint import State, save_all_states, load_state
    from adaptdl.env import replica_rank, num_restarts

    class TestState(State):
        def __init__(self, name):
            super().__init__(name)
            self.synced = False

        def sync(self):
            self.synced = True

        def save(self, fileobj):
            assert replica_rank() == 0  # Should only be called from rank 0.
            pickle.dump(self.value, fileobj)

        def load(self, fileobj):
            # Should load the correct value.
            self.value = pickle.load(fileobj)

    state_1 = TestState("state_1")
    state_2 = TestState("state_2")

    if num_restarts() == 0:
        # Save all state.
        state_1.value = 10
        state_2.value = 20
        save_all_states()
        assert state_1.synced and state_2.synced
        return 2  # Restart with 2 replicas.
    elif num_restarts() == 1:
        load_state(state_1)
        load_state(state_2)
        assert state_1.value == 10
        assert state_2.value == 20
    else:
        assert False
