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


import sys
if "darwin" in sys.platform.lower():
    # To avoid multiple runs of the model code
    # https://pythonspeed.com/articles/python-multiprocessing/
    import multiprocessing
    multiprocessing.set_start_method('fork')

import logging
import portpicker
import requests
import torch.distributed
import pkg_resources

import adaptdl.collective
import adaptdl.env
import semver
from .epoch import current_epoch, finished_epochs, remaining_epochs_until
from .data import current_dataloader, AdaptiveDataLoader, ElasticSampler
from .parallel import AdaptiveDataParallel
from .accumulator import Accumulator

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


def version_check(version):
    if semver.VersionInfo.isvalid(version) and \
            version != "0.0.0":
        return True
    else:
        return False


def init_process_group(backend):
    url = adaptdl.env.supervisor_url()
    if url:
        key = adaptdl.env.job_id()
        group = adaptdl.env.num_restarts()
        while True:
            response = requests.get(url=f"{url}/discover/{key}/{group}")
            if response.status_code != 408:  # Timeout.
                break
        response.raise_for_status()
        master_addr = response.json()[0]
        sched_version = adaptdl.env.adaptdl_sched_version()
        trainer_version = pkg_resources.get_distribution("adaptdl").version
        if version_check(sched_version) and version_check(trainer_version):
            trainer_ver_maj = semver.VersionInfo.parse(trainer_version).major
            sched_ver_maj = semver.VersionInfo.parse(sched_version).major
            if trainer_ver_maj != sched_ver_maj:
                raise Exception('adaptdl version {} is incompatible with'
                                'scheduler version {}'.format(trainer_version,
                                                              sched_version))
    else:
        master_addr = adaptdl.env.master_addr()

    master_port = adaptdl.env.master_port()

    # Initialize collective module.
    adaptdl.collective.initialize(master_addr, master_port)

    # Initialize torch.distributed.
    torch_port = adaptdl.collective.broadcast(portpicker.pick_unused_port())
    init_method = "tcp://{}:{}?rank={}&world_size={}".format(
            master_addr, torch_port, adaptdl.env.replica_rank(),
            adaptdl.env.num_replicas())
    LOG.info("Initializing torch.distributed using %s", init_method)
    torch.distributed.init_process_group(backend, init_method)

    LOG.info("torch.distributed initialized")


__all__ = [
    "init_process_group",
    "current_epoch",
    "finished_epochs",
    "remaining_epochs_until",
    "current_dataloader",
    "AdaptiveDataLoader",
    "ElasticSampler",
    "AdaptiveDataParallel",
    "Accumulator",
]
