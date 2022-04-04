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


@elastic_multiprocessing
def test_epoch():
    import adaptdl.checkpoint
    from adaptdl.env import num_restarts
    from adaptdl.torch.epoch import (remaining_epochs_until,
                                     current_epoch, finished_epochs)
    total_epochs = 10
    restart_epoch = 5
    assert current_epoch() is None
    if num_restarts() == 0:
        assert finished_epochs() == 0
        expected_epochs = list(range(restart_epoch + 1))
    elif num_restarts() == 1:
        assert finished_epochs() == restart_epoch
        expected_epochs = list(range(restart_epoch, total_epochs))
    else:
        assert False
    for idx, epoch in enumerate(remaining_epochs_until(10)):
        assert epoch == expected_epochs[idx]
        assert current_epoch() == epoch
        assert finished_epochs() == epoch
        if num_restarts() == 0 and epoch == restart_epoch:
            adaptdl.checkpoint.save_all_states()
            return 5  # Restart with 5 replicas.
