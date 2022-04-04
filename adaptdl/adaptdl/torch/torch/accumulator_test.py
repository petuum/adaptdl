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


from adaptdl.conftest import elastic_multiprocessing
from adaptdl.torch.accumulator import Accumulator


@elastic_multiprocessing
def test_accumulator_restarts():
    import adaptdl.checkpoint
    import adaptdl.collective
    from adaptdl.env import num_restarts, replica_rank
    adaptdl.collective.initialize("0.0.0.0")
    accum = Accumulator()

    if num_restarts() == 0:
        accum["a"] += 15  # Idempotent.
    assert "a" not in accum
    with accum.synchronized():
        assert "a" in accum
        assert accum["a"] == 15
    assert "a" not in accum
    if num_restarts() == 0:
        accum["a"] -= 5  # Idempotent.
        adaptdl.checkpoint.save_all_states()
        return 4  # Restart with 4 replicas.

    if num_restarts() == 1:  # Idempotent.
        accum.update({"a": replica_rank(), "b": replica_rank()})
    assert len(accum) == 0
    with accum.synchronized():
        assert len(accum) == 2
        assert accum["a"] == 16
        assert accum["b"] == 6
    assert len(accum) == 0
    if num_restarts() == 1:
        adaptdl.checkpoint.save_all_states()
        return 2  # Restart with 2 replicas.

    if num_restarts() == 2:  # Idempotent.
        accum -= {"b": 5, "c": 5}
    with accum.synchronized():
        assert accum["a"] == 16
        assert accum["b"] == -4
        assert accum["c"] == -10
        accum.clear()
    with accum.synchronized():
        assert not accum
