import pytest
import ray

import json

import worker
from aiohttp import web


@ray.remote
def mocked_listen_for_spot_termination():
    time.sleep(3)
    return "a different ip"


worker.listen_for_spot_termination = mocked_listen_for_spot_termination


from controller import RayAdaptDLJob, Cluster # noqa
from controller import _test_controller as Controller # noqa
from utils import Status # noqa

import time # noqa
import asyncio # noqa
from collections import namedtuple # noqa
import requests # noqa

loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)


# see https://stackoverflow.com/a/60334747
@pytest.fixture(scope="module")
def ray_fix():
    ray.init()
    yield None
    ray.shutdown()


MockedJob = namedtuple("Job", ["workers"])


def test_adaptdl_job_checkpoint(ray_fix):
    job = RayAdaptDLJob(None, 0, 0)

    @ray.remote
    def worker():
        while True:
            time.sleep(1)
    tasks = {i: worker.remote() for i in range(5)}
    job._worker_tasks = tasks
    job._running = True
    loop.run_until_complete(job.force_worker_checkpoint())
    assert not job._running


def test_adaptdl_job_register_hints(ray_fix):
    job = RayAdaptDLJob(None, 0, 0)
    job.register_hints("some hints")
    assert job._last_metrics == "some hints"


def test_adaptdl_job_hints(ray_fix):
    hints = {
        "gradParams": {"norm": 3.0, "var": 4.0},
        "perfParams": {
            'alpha_c': 0, 'beta_c': 0, 'alpha_n': 0,
            'beta_n': 0, 'alpha_r': 0, 'beta_r': 0, 'gamma': 0}}
    job = RayAdaptDLJob(None, 0, 0)
    job._last_metrics = hints
    job.hints


def test_adaptdl_job_register_checkpoint(ray_fix):
    job = RayAdaptDLJob(None, 0, 0)
    checkpoint = "foo"
    assert not job._checkpoint_received
    job.register_checkpoint(checkpoint)
    assert job._checkpoint == "foo"
    assert job._checkpoint_received
    assert ray.get(job._checkpoint_ref) == "foo"


def test_adaptdl_job_register_status(ray_fix):
    job = RayAdaptDLJob(None, 0, 0)
    status = Status.FAILED.value
    job.register_status(status)
    assert job._status == Status.FAILED
    assert job.completed.is_set()

    job = RayAdaptDLJob(None, 0, 0)
    status = Status.SUCCEEDED.value
    job.register_status(status)
    assert job._status == Status.SUCCEEDED
    assert job.completed.is_set()

    job = RayAdaptDLJob(None, 0, 0)
    status = Status.RUNNING.value
    job.register_status(status)
    assert job._status == Status.RUNNING
    assert not job.completed.is_set()


def test_cluster_invalid_nodes(ray_fix):
    cluster = Cluster(None, 0)
    cluster.mark_node_for_termination("some ip")
    cluster.mark_node_for_termination("some other ip")
    assert (cluster._invalid_nodes ==
            {"some ip", "some other ip", ray.services.get_node_ip_address()})


def test_controller_run(ray_fix):
    controller = Controller(100, 5)
    controller.app_ran = False

    class MockedRunner():
        def __init__(self):
            self.cleaned_up = False

        async def cleanup(self):
            self.cleaned_up = True

    async def mocked_run_app():
        controller._completed.set()
        controller._runner = MockedRunner()
        controller.app_ran = True

    controller._run_app = mocked_run_app
    loop.run_until_complete(controller.run_controller())

    assert controller._runner.cleaned_up
    assert controller.app_ran
    controller._runner.cleanup()


def test_controller_create_job(ray_fix):
    controller = Controller(100, 5)
    controller._ready.set()
    controller.rescheduled = False

    async def mocked_reschedule():
        controller.rescheduled = True
        controller._job.completed.set()
        controller._job._status = Status.SUCCEEDED

    controller._reschedule_jobs = mocked_reschedule

    resources = {"CPU": 1, "GPU": 2}
    loop.run_until_complete(
        controller.create_job(
            worker_resources=resources,
            worker_port_offset=0,
            checkpoint_timeout=1))

    assert controller._job._worker_resources == resources
    assert controller._job._worker_port_offset == 0
    assert controller._job._checkpoint_timeout == 1
    assert controller._cluster._worker_resources == resources
    assert controller.rescheduled
    assert controller._job._status == Status.SUCCEEDED


def test_controller_reschedule_jobs(ray_fix):
    controller = Controller(100, 5)
    job = RayAdaptDLJob(None, 0, 0)
    controller._job = job
    job.forced_checkpoint = False
    job.updated = 0

    async def mocked_force_worker_checkpoint():
        job.forced_checkpoint = True

    async def mocked_update_workers(allocation, force_checkpoint):
        await asyncio.sleep(3)
        job.updated += 1

    job.force_worker_checkpoint = mocked_force_worker_checkpoint
    job.update_workers = mocked_update_workers

    controller._cluster = Cluster(None, 0)
    controller._cluster.expanded = None

    async def mocked_expand_cluster(workers, allocation):
        controller._cluster.expanded = (workers, allocation)

    controller._cluster.expand_cluster = mocked_expand_cluster

    async def wrapped_call(duration, force=False):
        await asyncio.sleep(duration)
        await controller._reschedule_jobs(force)

    loop.run_until_complete(asyncio.gather(wrapped_call(0), wrapped_call(1), wrapped_call(2, True)))

    assert job.forced_checkpoint
    assert job.updated == 2
    assert controller._cluster.expanded == ({}, ['virtual_node_0'])


def test_controller_spot_termination_continuation(ray_fix):
    controller = Controller(100, 5)
    job = RayAdaptDLJob(None, 0, 0)
    controller._job = job
    controller.rescheduled = False
    controller.immediate_checkpoint = None

    async def mocked_reschedule(immediate_checkpoint):
        controller.rescheduled = True
        controller.immediate_checkpoint = immediate_checkpoint

    controller._cluster = Cluster(None, 0)
    controller._reschedule_jobs = mocked_reschedule

    controller._cluster.marked = None

    def mocked_mark_node_for_termination(ip):
        controller._cluster.marked = ip

    controller._cluster.mark_node_for_termination = mocked_mark_node_for_termination

    async def task():
        return "some ip"

    async def wrapper():
        awaitable_task = asyncio.create_task(task())
        await controller._spot_termination_continuation(awaitable_task)

    loop.run_until_complete(wrapper())
    assert controller.rescheduled
    assert controller.immediate_checkpoint
    assert controller._cluster.marked == "some ip"


def test_controller_register_worker(ray_fix):
    controller = Controller(100, 5)
    job = RayAdaptDLJob(None, 0, 0)
    controller._job = job
    controller._spot_listener_tasks = {"some-ip": 1}

    controller.task_result = None

    async def mocked_spot_termination_continuation(task):
        controller.task_result = ray.get(task)

    controller._spot_termination_continuation = mocked_spot_termination_continuation

    ip = ray.services.get_node_ip_address()

    loop.run_until_complete(controller.register_worker(0, "some-ip"))
    loop.run_until_complete(controller.register_worker(1, ray.services.get_node_ip_address()))

    assert job._workers[0] == "some-ip"
    assert job._workers[1] == ip
    assert controller.task_result == "a different ip"


def test_controller_register_checkpoint(ray_fix):
    controller = Controller(100, 5)
    job = RayAdaptDLJob(None, 0, 0)
    controller._job = job
    checkpoint = "foo"
    checkpoint_received = loop.run_until_complete(controller.register_checkpoint(checkpoint))
    assert checkpoint_received
    assert job._checkpoint_received
    assert job._checkpoint == "foo"
    assert ray.get(job._checkpoint_ref) == "foo"


def test_controller_register_status():
    controller = Controller(100, 5)
    job = RayAdaptDLJob(None, 0, 0)
    controller._job = job
    status = Status.RUNNING.value
    loop.run_until_complete(controller.register_status(status))
    assert(job._status == Status.RUNNING and not job.completed.is_set())
    status = Status.SUCCEEDED.value
    loop.run_until_complete(controller.register_status(status))
    assert(job._status == Status.SUCCEEDED and job.completed.is_set())


def test_controller_handle_report():
    controller = Controller(100, 5)
    job = RayAdaptDLJob(None, 0, 0)
    controller._job = job
    controller.rescheduled = False

    async def mocked_reschedule():
        controller.rescheduled = True

    controller._reschedule_jobs = mocked_reschedule

    class MockedRequest:
        def __init__(self, body):
            self._body = body

        async def json(self):
            return self._body
    hints = {"some": "hints"}
    hints_json = MockedRequest(json.dumps(hints))
    loop.run_until_complete(controller._handle_report(hints_json))
    assert(
        controller.rescheduled and
        json.loads(job._last_metrics) == hints and
        id(job._last_metrics) != id(hints))


async def test_controller_handle_discover():
    controller = Controller(4, 100)
    controller._job = MockedJob(workers={0: "127.0.0.1", 1: "127.0.0.2", 2: "0.0.0.0"})
    workers = await controller._handle_discover(None)
    assert (json.loads(workers.text) == ["127.0.0.1", "127.0.0.2", "0.0.0.0"])


async def test_controller_run_app(aiohttp_client):
    controller = Controller(4, 100)
    controller.called_report = False
    controller.called_discover = False

    async def mocked_report(request):
        controller.called_report = True
        return web.Response(text="Success Hints")

    async def mocked_discover(request):
        controller.called_discover = True
        return web.Response(text="Success Discover")

    port = 8713

    async def put():
        await asyncio.sleep(5)
        client = await aiohttp_client(controller._runner.app())
        return await client.put("/hints/namespace/name")

    async def get():
        await asyncio.sleep(5)
        client = await aiohttp_client(controller._runner.app())
        return await client.get("/discover/namespace/name/group")

    controller._handle_discover = mocked_discover
    controller._handle_report = mocked_report
    await controller._run_app(port)
    put_response = await put()
    get_response = await get()

    put_text = await(put_response.text())
    assert put_text == "Success Hints"
    get_text = await(get_response.text())
    assert get_text == "Success Discover"

    assert controller.called_report
    assert controller.called_discover
    await controller._runner.cleanup()
