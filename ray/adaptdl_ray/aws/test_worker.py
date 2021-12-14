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


import ray
import pytest
from adaptdl_ray.aws.worker import run_adaptdl, listen_for_spot_termination
import asyncio
import os
import time

from aiohttp import web


# see https://stackoverflow.com/a/60334747
@pytest.fixture(scope="module")
def ray_fix():
    if not ray.is_initialized:
        ray.init()
        yield None
        ray.shutdown()
    else:
        yield None


@ray.remote
class MockedController():

    def __init__(self):
        self.checkpoint = None

    def register_status(self, *args, **kwargs):
        print(*args, **kwargs)
        pass

    def register_worker(self, *args, **kwargs):
        print(*args, **kwargs)
        pass

    def register_checkpoint(self, obj):
        self.checkpoint = obj

    def get_checkpoint(self):
        return self.checkpoint

    def get_url(self):
        return ray._private.services.get_node_ip_address()


@ray.remote
class TerminationEndpoint():
    def __init__(self):
        self.terminating = False

    async def handle_termination_request(self, request):
        if self.terminating:
            return web.json_response({"action": "terminate"})
        else:
            return web.HTTPNotFound()

    async def start_server(self):
        app = web.Application()
        app.add_routes([
            web.get('/latest/meta-data/spot/instance-action',
                    self.handle_termination_request)
            ])
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(
            runner, ray._private.services.get_node_ip_address(), 8234)
        await site.start()
        await asyncio.sleep(30)
        await runner.cleanup()

    async def set_to_terminate(self):
        self.terminating = True
        return None


def test_worker(ray_fix):
    controller = MockedController.options(name="AdaptDLController").remote()
    rank = 0
    replicas = 2
    restarts = 3
    checkpoint = None
    offset = 50
    path = "ray/adaptdl_ray/aws/_example_worker.py"
    argv = ["--arg1", "value", "--arg2", "value"]

    worker_task = run_adaptdl.remote(
        "test_key", "test_uid", rank, replicas,
        restarts, checkpoint, offset, path, argv)

    # can't cancel with force=True
    time.sleep(10)
    ray.cancel(worker_task, force=False)
    print("canceling")
    time.sleep(10)
    checkpoint = ray.get(controller.get_checkpoint.remote())
    print(checkpoint)
    assert('file.txt' in checkpoint)
    ray.cancel(worker_task, force=False)

    rank = 1
    replicas = 2
    restarts = 4
    offset = 50

    worker_task = run_adaptdl.remote(
        "test_key_2", "test_uid_2", rank, replicas,
        restarts, checkpoint, offset, path, argv)

    time.sleep(10)
    assert(os.path.exists("/tmp/checkpoint-test_uid_2-1/file.txt"))
    with open("/tmp/checkpoint-test_uid_2-1/file.txt", "rb") as f:
        result = int(f.read())
        assert (result == 5)


async def test_spot_instance_termination(ray_fix):
    endpoint = TerminationEndpoint.remote()
    endpoint.start_server.remote()

    task = listen_for_spot_termination.remote(timeout=5.0)
    ip = ray.get(task)
    assert(ip is None)
    task = listen_for_spot_termination.remote(timeout=15.0)
    time.sleep(5)
    await endpoint.set_to_terminate.remote()
    ip = ray.get(task, timeout=10)
    assert(ip == ray._private.services.get_node_ip_address()), \
        f"found {ip}, expected {ray._private.services.get_node_ip_address()}"
