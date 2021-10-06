import importlib
import os
from pathlib import Path
import pkg_resources
import requests
import sys
import time
import traceback
import uuid

from utils import _checkpoint_obj_to_dir, _serialize_checkpoint, Status

import ray
import ray.services as services

MOCK = (os.environ.get("MOCK", "False").lower() == "true")

if MOCK:
    print("Using mocked spot instance")

namespace = "foo"
name = "foo"
group = "0"


job_key = str(uuid.uuid4())[:8]


@ray.remote(num_cpus=0.1)
def listen_for_spot_termination():

    if MOCK:
        endpoint = "0.0.0.0:8234"
    else:
        # AWS spot instance termination endpoint
        endpoint = "169.254.169.254"

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
                    print(f"termination detected on node {ip}")
                    return ip
            else:
                raise RuntimeError(
                    "AWS spot instance interrupt warning "
                    "endpoint not responding")
        except requests.RequestException as e:
            print(e)
            time.sleep(5)


@ray.remote(num_cpus=1, num_gpus=1)
def run_adaptdl(job_key, rank, replicas, supervisor_url,
                num_restarts, checkpoint=None, offset=0, path="", argv=None):
    def report_status(status):
        status_obj_ref = ray.put(status.value)
        manager.register_status.remote(status_obj_ref)

    print(f"starting worker: {num_restarts}/{rank}")
    manager = ray.get_actor("AdaptDLManager")
    print(f"manager: {manager}")

    job_id = "foo/foo"

    os.environ["ADAPTDL_MASTER_PORT"] = str(47000 + num_restarts + offset)
    os.environ["ADAPTDL_REPLICA_RANK"] = str(rank)
    os.environ["ADAPTDL_NUM_REPLICAS"] = str(replicas)
    os.environ["ADAPTDL_SUPERVISOR_URL"] = supervisor_url
    print(f"supervisor_url: {supervisor_url}")
    os.environ["ADAPTDL_JOB_ID"] = job_id
    os.environ["ADAPTDL_NUM_RESTARTS"] = str(num_restarts)
    os.environ["ADAPTDL_SCHED_VERSION"] = str(
        pkg_resources.get_distribution("adaptdl").version)
    uid = str(uuid.uuid4())[:8]
    suffix = f"{job_key}-{rank}-{uid}"
    checkpoint_path = f"/tmp/checkpoint-{suffix}"
    print(f"creating checkpoint dir: {checkpoint_path}")
    print(f"master port:{os.environ.get('ADAPTDL_MASTER_PORT')}")

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

        job_key = f"{namespace}/{name}"
        rank_obj_ref = ray.put(rank)
        ip_obj_ref = ray.put(services.get_node_ip_address())
        manager.register_worker.remote(rank_obj_ref, ip_obj_ref)
    except Exception as e:
        print(traceback.format_exc())
        time.sleep(5)
        report_status(Status.FAILED)
        raise e

        # TODO: replace with block

    try:
        print(f"argv: {argv}")
        if argv is not None:
            # Need to augment the argv to mimic that file being called
            filename = Path(path).name
            sys.argv = [filename] + argv
        spec = importlib.util.spec_from_file_location("__main__", path)
        module = importlib.util.module_from_spec(spec)
        # TODO: fix imports when caller module is not in the root path
        spec.loader.exec_module(module)

    except SystemExit:
        # Received a cancel from the manager -- the job is being rescheduled
        # Worker 0 needs to send the checkpoint back to the manager so the
        # next generation of workers can resume
        print(f"Received system exit: {rank}")
        if rank == 0:
            print(f"Checkpoint: {os.listdir(checkpoint_path)}")
            checkpoint_obj = _serialize_checkpoint(checkpoint_path)
            checkpoint_obj_ref = ray.put(checkpoint_obj)
            manager.register_checkpoint.remote(checkpoint_obj_ref)
        # This sleep is to keep this remote task alive
        # until its worker object can be killed by the manager
        time.sleep(1000)

    except Exception as e:
        print(traceback.format_exc())
        print(e)
        time.sleep(5)
        report_status(Status.FAILED)
        raise e
    else:
        if rank == 0:
            print("Job succeeded, exiting")
            report_status(Status.SUCCEEDED)
