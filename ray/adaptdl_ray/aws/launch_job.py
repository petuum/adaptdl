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
import logging
import os

from adaptdl_ray.aws.controller import Controller
from adaptdl_ray.aws.utils import Status

import ray


logging.basicConfig()
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


def run_adaptdl_on_ray_cluster(
        path, argv, ray_uri, working_dir,
        worker_resources, cluster_size, worker_port_offset,
        checkpoint_timeout, rescale_timeout):
    LOG.info("Starting AdaptDLJob")
    if ray.is_initialized():
        return
    if not os.path.exists(working_dir):
        raise RuntimeError(f"Cannot find local directory {working_dir}")
    if not os.path.exists(os.path.join(working_dir, path)):
        raise RuntimeError(
            f"Cannot find local file {os.path.join(working_dir, path)}")
    runtime_env = {
        "working_dir": working_dir}
    ray.init(ray_uri, runtime_env=runtime_env)

    controller = Controller.options(name="AdaptDLController").remote(
        cluster_size, rescale_timeout)

    controller.run_controller.remote()
    try:
        status_obj = controller.create_job.remote(
            worker_resources, worker_port_offset, checkpoint_timeout,
            path=path, argv=argv)
        status = ray.get(status_obj)
        if status.value == Status.SUCCEEDED.value:
            LOG.info("Job succeeded")
            return 0
        else:
            raise RuntimeError("Job failed")
    except Exception as e:
        raise e


def main():
    parser = argparse.ArgumentParser(description="Adaptdl on Ray")
    parser.add_argument(
        "-f", "--file", type=str,
        help=("File to run on the cluster. The path must be a relative path"
              "rooted at the argument of --working-dir"), required=True)
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
        "--cpus", type=int, help="number of cpus per worker", default=1)
    parser.add_argument(
        "--port-offset", type=int, default=0,
        help=("Torch communication worker port offset. Generally to be used "
              "in case the adaptdl communication ports are taken "))
    parser.add_argument(
        "-d", "--working-dir", type=str,
        help=("Directory to copy to the worker tasks. Should contain the file "
              "specified by -f/--file."),
        required=True)
    parser.add_argument(
        "--checkpoint-timeout", type=int, default=120,
        help=("Number of seconds that the controller will wait for the "
              "checkpoint object to be reported back by worker 0. "
              "If the checkpoint is not received by that time, an old "
              "checkpoint will be used, if one exists"))
    parser.add_argument(
        "--cluster-rescale-timeout", type=int, default=60,
        help=("Number of seconds that the controller will wait for the "
              "cluster to rescale to the desired size. If the desired size "
              "is not reached by the end of this timeout, then the controller "
              "will allocate on whatever nodes it finds then"))
    parser.add_argument(
        "arguments", type=str, nargs=argparse.REMAINDER,
        help=("Command line arguments to be passed to the file specified by "
              "-f/--f. You many seperate these from the other command line "
              "arguments with --"))
    args = parser.parse_args()
    if args.arguments and args.arguments[0] == "--":
        arguments = args.arguments[1:]
    else:
        arguments = []
    run_adaptdl_on_ray_cluster(
        args.file,
        argv=arguments,
        ray_uri=args.uri,
        working_dir=args.working_dir,
        worker_resources={"CPU": args.cpus, "GPU": args.gpus},
        cluster_size=args.max_cluster_size,
        worker_port_offset=args.port_offset,
        checkpoint_timeout=args.checkpoint_timeout,
        rescale_timeout=args.cluster_rescale_timeout)


if __name__ == "__main__":
    main()
