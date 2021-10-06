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


import argparse

from manager import Manager
from utils import Status
from pathlib import Path

import ray

import time


def run_as_ray(path, argv, ray_uri, working_dir,
               worker_resources, cluster_size, worker_port_offset):
    if ray.is_initialized():
        return
    if not working_dir:
        path = Path(path)
        working_dir = path.parent.absolute()

    runtime_env = {
        "working_dir": working_dir}
    ray.init(ray_uri, runtime_env=runtime_env, log_to_driver=True)

    manager = Manager.options(name="AdaptDLManager").remote(
        worker_resources, cluster_size, worker_port_offset=worker_port_offset,
        path=path, argv=argv)
    status = ray.get(manager.run_job.remote())
    time.sleep(5)
    print(status)
    if status == Status.SUCCEEDED:
        return 0
    else:
        raise RuntimeError("Job failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adaptdl on Ray")
    parser.add_argument(
        "-f", "--file", type=str,
        help="File to run on the cluster", required=True)
    parser.add_argument(
        "-u", "--uri", type=str,
        help="URI of the ray cluster, e.g. `ray://<ip>:10001", required=True)
    parser.add_argument(
        "-m", "--max-cluster-size", type=int,
        help="Maximum number of workers (with sufficient gpus) in the cluster",
        required=True)
    parser.add_argument(
        "--gpus", type=int, help="number of gpus per worker", default=1)
    parser.add_argument(
        "--port-offset", type=int,
        help="Torch communication worker port offset", default=0)
    parser.add_argument(
        "-d", "--working-dir", type=str,
        help=("Directory to copy to the worker tasks. Should contain the file "
              "specified by -f/--file. Defaults to the parent of -f/--file"),
        required=False)
    parser.add_argument(
        "-a", "--arguments", type=str, nargs="+", action="extend",
        help=("Command line arguments to be passed to the file specified by "
              "-f/--f"))
    args = parser.parse_args()
    run_as_ray(args.file, argv=args.arguments,
               ray_uri=args.uri,
               working_dir=args.working_dir,
               worker_resources={"CPU": 1, "GPU": args.gpus},
               cluster_size=args.max_cluster_size,
               worker_port_offset=args.port_offset)
