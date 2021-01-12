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


import functools
import logging

import adaptdl.checkpoint
import adaptdl.env
from adaptdl.torch._metrics import _report_sched_hints
from adaptdl.torch.data import current_dataloader

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


GPU_MEM_CUTOFF_PCT = 0.1


def cudaoom(e):
    return "CUDA out of memory" in str(e)


def retry(func):
    @functools.wraps(func)
    def inner(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except RuntimeError as e:
            LOG.info(f"{e}")
            dataloader = current_dataloader()
            if (dataloader is not None and
                dataloader.local_bsz_bounds is not None and
                    cudaoom(e)):
                current_local_bsz = dataloader.current_local_bsz
                low, high = dataloader.local_bsz_bounds
                assert current_local_bsz <= high
                new_high = int((1. - GPU_MEM_CUTOFF_PCT) * current_local_bsz)
                dataloader.local_bsz_bounds = (low, new_high)
                LOG.info(f"Local batch size bounds changed to "
                         f"{dataloader.local_bsz_bounds}")
                if adaptdl.env.replica_rank() == 0:
                    _report_sched_hints()
                adaptdl.checkpoint.save_all_states()
                exit(143)
            else:
                raise e
    return inner
