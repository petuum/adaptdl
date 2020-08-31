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


@pytest.mark.parametrize("num_replicas", [1, 2, 3, 4])
@elastic_multiprocessing
def test_profile(num_replicas):
    import adaptdl.checkpoint
    from adaptdl.env import num_restarts
    from adaptdl.torch._metrics import (
            profile_step_start, profile_sync_time,
            profile_step_commit, _metrics_state)
    if num_restarts() == 0:
        profile = _metrics_state().profile
        assert len(profile) == 0
        # Profile local_bsz=1 but don't commit.
        profile_step_start(1)
        profile_sync_time(1.0)
        # Profile local_bsz=2 and commit.
        profile_step_start(2)
        profile_sync_time(1.0)
        profile_sync_time(2.0)
        profile_step_commit()
        # Ensure profile is updated correctly.
        profile = _metrics_state().profile
        assert len(profile) == 1
        assert profile[1, 1, 2]["count"] == 1
        assert profile[1, 1, 2]["sync_time"] == 3.0
        assert profile[1, 1, 2]["step_time"] > 0.0
        # Checkpoint and restart.
        adaptdl.checkpoint.save_all_states()
        return num_replicas
    elif num_restarts() == 1:
        profile = _metrics_state().profile
        # Ensure checkpoint is loaded correctly.
        assert len(profile) == 1
        assert profile[1, 1, 2]["count"] == 1
        assert profile[1, 1, 2]["sync_time"] == 3.0
        assert profile[1, 1, 2]["step_time"] > 0.0
        # Profile local_bsz=3 and commit twice.
        profile_step_start(3)
        profile_sync_time(2.0)
        profile_sync_time(3.0)
        profile_step_commit()
        old_step_time = profile[1, num_replicas, 3]["step_time"]
        profile_step_start(3)
        profile_sync_time(3.0)
        profile_sync_time(4.0)
        profile_step_commit()
        # Ensure profile is updated correctly.
        assert len(profile) == 2
        assert profile[1, num_replicas, 3]["count"] == 2
        assert profile[1, num_replicas, 3]["sync_time"] == 12.0
        assert profile[1, num_replicas, 3]["step_time"] > old_step_time > 0.0
