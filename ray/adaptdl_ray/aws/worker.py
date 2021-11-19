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


import importlib
import logging
import os
from pathlib import Path
import pkg_resources
import requests
import sys
import time
import traceback

from adaptdl_ray.aws.utils import \
    _checkpoint_obj_to_dir, _serialize_checkpoint, Status

import ray
import ray._private.services as services


@ray.remote(num_cpus=0.1, max_retries=0)
def listen_for_spot_termination(timeout=None):
    MOCK = (os.environ.get("MOCK", "False").lower() == "true")
    logging.basicConfig(level=logging.INFO)

    if MOCK:
        logging.debug("Using mocked spot instance")
        endpoint = f"{services.get_node_ip_address()}:8234"
    else:
        # AWS spot instance termination endpoint
        endpoint = "169.254.169.254"

    start = time.time()

    while True:
        try:
            resp = requests.get(
                f'http://{endpoint}/latest/meta-data/spot/instance-action',
                timeout=0.1)
            if resp.status_code == 404:
                # AWS endpoint responded, no termination detected
                time.sleep(5)
            elif resp.status_code >= 200 and resp.status_code < 300:
                resp_json = resp.json()
                if (resp_json["action"] == "terminate"
                        or resp_json["action"] == "stop"):
                    ip = services.get_node_ip_address()
                    logging.info(f"termination detected on node {ip}")
                    return ip
            else:
                raise RuntimeError(
                    "AWS spot instance interrupt warning "
                    "endpoint not responding")
            if timeout and time.time() - start > timeout:
                return None
        except requests.RequestException as e:
            logging.error(e)
            time.sleep(5)


@ray.remote(max_retries=0)
def run_adaptdl(job_key, job_uid, rank, replicas,
                num_restarts, checkpoint=None, offset=0, path="", argv=None):
    logging.basicConfig(level=logging.INFO)
    logging.info(f"Starting worker {rank}")

    def report_status(status):
        status_obj_ref = ray.put(status.value)
        controller.register_status.remote(status_obj_ref)

    controller = ray.get_actor("AdaptDLController")
    supervisor_url = ray.get(controller.get_url.remote())

    os.environ["ADAPTDL_MASTER_PORT"] = str(47000 + num_restarts + offset)
    os.environ["ADAPTDL_REPLICA_RANK"] = str(rank)
    os.environ["ADAPTDL_NUM_REPLICAS"] = str(replicas)
    os.environ["ADAPTDL_SUPERVISOR_URL"] = supervisor_url
    os.environ["ADAPTDL_JOB_ID"] = job_key
    os.environ["ADAPTDL_NUM_RESTARTS"] = str(num_restarts)
    os.environ["ADAPTDL_SCHED_VERSION"] = str(
        pkg_resources.get_distribution("adaptdl").version)
    suffix = f"{job_uid}-{rank}"
    checkpoint_path = f"/tmp/checkpoint-{suffix}"

    try:
        if os.path.exists(checkpoint_path):
            import shutil
            shutil.rmtree(checkpoint_path)
        os.mkdir(checkpoint_path)
        if checkpoint:
            _checkpoint_obj_to_dir(checkpoint_path, checkpoint)
        num_restarts = int(num_restarts)
        os.environ["ADAPTDL_CHECKPOINT_PATH"] = str(checkpoint_path)
        share_path = f"/tmp/share-{suffix}"
        if not os.path.exists(share_path):
            os.mkdir(share_path)
        os.environ["ADAPTDL_SHARE_PATH"] = str(share_path)

        rank_obj_ref = ray.put(rank)
        ip_obj_ref = ray.put(services.get_node_ip_address())
        controller.register_worker.remote(rank_obj_ref, ip_obj_ref)
    except Exception as e:
        logging.info(traceback.format_exc())
        time.sleep(5)
        report_status(Status.FAILED)
        raise e

        # TODO: replace with block
    try:
        filename = Path(path).name
        sys.argv = [filename]
        if argv:
            # Need to augment the argv to mimic that file being called
            sys.argv += argv
        spec = importlib.util.spec_from_file_location("__main__", path)
        module = importlib.util.module_from_spec(spec)
        # TODO: fix imports when caller module is not in the root path
        spec.loader.exec_module(module)
        time.sleep(5)

    except SystemExit:
        # Received a cancel from the controller -- the job is being rescheduled
        # Worker 0 needs to send the checkpoint back to the controller so the
        # next generation of workers can resume
        logging.info(f"Worker {rank} received system exit")
        if rank == 0:
            checkpoint_obj = _serialize_checkpoint(checkpoint_path)
            logging.info("checkpoint created")
            checkpoint_obj_ref = ray.put(checkpoint_obj)
            logging.info("checkpoint placed")
            result = ray.get(
                controller.register_checkpoint.remote(checkpoint_obj_ref))
            logging.info(f"checkpoint registered: {result}")
        # This sleep is to keep this remote task alive
        # until its worker object can be killed by the controller
        time.sleep(1800)

    except Exception as e:
        logging.error(traceback.format_exc())
        logging.error(e)
        time.sleep(5)
        report_status(Status.FAILED)
        raise e
    else:
        if rank == 0:
            logging.info("Job succeeded, exiting")
            time.sleep(5)
            report_status(Status.SUCCEEDED)
            time.sleep(5)
