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


from aiohttp import web

import asyncio
import copy
import logging
import uuid
import time

from collections import namedtuple

from adaptdl_ray.adaptdl.adaptdl_job_mixin import AdaptDLJobMixin
from adaptdl_ray.adaptdl.utils import allocation_to_pgf

from adaptdl.goodput import PerfParams, GradParams

import ray
import ray.autoscaler.sdk as sdk
import ray._private.services as services

from .optimizer import optimize
from .utils import Status
from .worker import listen_for_spot_termination, run_adaptdl

from ray.util.placement_group import (
    remove_placement_group
)

# To be used when there are no running tasks
FULL_RESCALE_TIMEOUT = 6000

NAMESPACE = "adaptdl_job"
NAME = "adaptdl_job"

job_uid = str(uuid.uuid4())[:8]


class RayAdaptDLJob(AdaptDLJobMixin):
    def __init__(self,
                 worker_resources, worker_port_offset,
                 checkpoint_timeout, **kwargs):
        super().__init__(job_id=job_uid)
        self._worker_resources = worker_resources
        self._job_params = kwargs

        # TODO: merge workers, worker_tasks
        self._workers = {}  # int: IP
        self._worker_tasks = {}
        self._last_metrics = None

        self._runner = None
        self._pg = None
        self._worker_port_offset = worker_port_offset

        self._running = False
        self._checkpoint = None
        self._checkpoint_ref = None
        self._checkpoint_received = asyncio.Event()
        self._checkpoint_lock = asyncio.Lock()
        self._checkpoint_timeout = checkpoint_timeout

        self._iteration = 0

        self.completed = asyncio.Event()
        self._status = Status.RUNNING

        self._placement_group_factory = None

    async def force_worker_checkpoint(self):
        if self._worker_tasks:
            logging.info("Checkpoint needed: stopping workers")
        async with self._checkpoint_lock:
            if self._running:
                for index, task in self._worker_tasks.items():
                    try:
                        ray.cancel(task, force=False)
                    except Exception as e:
                        raise e
                self._running = False

    def _stop_workers(self):
        if self._worker_tasks:
            logging.info("Terminating workers")
        for index, task in self._worker_tasks.items():
            try:
                ray.cancel(task, force=True)
            except Exception:
                pass
        self._worker_tasks = {}
        self._workers = {}

    @property
    def placement_group_factory(self):
        return self._placement_group_factory

    def register_hints(self, hints):
        self._last_metrics = copy.deepcopy(hints)

    @property
    def status(self):
        return self._status

    @property
    def worker_resources(self):
        return copy.deepcopy(self._worker_resources)

    @property
    def workers(self):
        return copy.deepcopy(self._workers)

    @property
    def hints(self):
        hints = copy.deepcopy(self._last_metrics)
        if hints:
            if hints["gradParams"]:
                hints["gradParams"]["sqr"] = hints["gradParams"]["norm"]
                del hints["gradParams"]["norm"]
            else:
                hints["gradParams"] = {"sqr": 1.0, "var": 1.0}
            hints["gradParams"] = GradParams(**hints["gradParams"])
            hints["perfParams"] = PerfParams(**hints["perfParams"])
        return hints

    def _fetch_metrics(self):
        hints = self.hints
        metrics = {}
        for key, value in hints.items():
            metrics_key = ''.join(
                ['_'+c.lower() if c.isupper() else c for c in key]).lstrip('_')
            metrics[metrics_key] = value
        Metrics = namedtuple("Metrics", metrics.keys())
        return Metrics(**metrics)

    def _allocation_in_use(self):
        return self._workers

    async def _create_workers(self, allocation):
        if self._pg:
            remove_placement_group(self._pg)
        logging.info(f"Creating {len(allocation)} worker tasks")
        self._placement_group_factory = allocation_to_pgf(
            allocation, self._worker_resources)
        self._pg = self._placement_group_factory()
        self._worker_tasks = {
            worker_index:
            run_adaptdl.options(
                num_cpus=self._worker_resources.get("CPU", 1),
                num_gpus=self._worker_resources.get("GPU", 0),
                placement_group=self._pg).remote(
                    f"{NAMESPACE}/{NAME}",
                    job_uid,
                    worker_index,
                    len(allocation),
                    self._iteration,
                    self._checkpoint_ref,
                    self._worker_port_offset,
                    **self._job_params)
            for worker_index, node in enumerate(allocation)}

        self._checkpoint_received.clear()
        self._running = True
        self._iteration += 1

    async def update_workers(self, allocation):
        logging.info(
            f"Updating workers. before: {len(self._workers.values())} workers"
            f", after: {len(allocation)} workers")
        if set(self._workers.values()) != set(allocation):
            await self.force_worker_checkpoint()
            # TODO: use asyncio events
            if self._worker_tasks:
                try:
                    await asyncio.wait_for(
                        self._checkpoint_received.wait(),
                        self._checkpoint_timeout)
                except asyncio.TimeoutError:
                    logging.warning("Waited for checkpoint, not found. "
                                    "Proceeding with previous checkpoint")
            self._stop_workers()
            await self._create_workers(allocation)
            return list(self._worker_tasks.values())
        else:
            logging.info("Allocation unchanged, proceeding")
            return None

    def register_checkpoint(self, obj):
        self._checkpoint = obj
        self._checkpoint_received.set()
        self._checkpoint_ref = ray.put(obj)

    def register_status(self, status):
        if self._status != Status.SUCCEEDED:
            self._status = Status(status)
        if self._status != Status.RUNNING:
            self.completed.set()


class Cluster():
    def __init__(self, worker_resources, rescale_timeout):
        self._worker_resources = worker_resources
        self._rescale_timeout = rescale_timeout
        self._terminating_nodes = set()
        self._force_immediate_allocation = asyncio.Event()

    @property
    def worker_resources(self):
        return copy.deepcopy(self._worker_resources)

    def mark_node_for_termination(self, node):
        self._terminating_nodes.add(node)

    @property
    def _invalid_nodes(self):
        current_ip = services.get_node_ip_address()
        return self._terminating_nodes.union({current_ip})

    def get_nodes(self):
        nodes = ray.nodes()
        return [node for node in nodes
                if (node["NodeManagerAddress"] not in self._invalid_nodes
                    and node["alive"] and "Resources" in node)]

    def _cluster_ready(self, allocation):
        nodes = self.get_nodes()
        nodes = {
            node["NodeManagerAddress"]: node["Resources"] for node in nodes}

        virtual_workers = []

        found_workers_count = 0
        for worker in allocation:
            if ("adaptdl_virtual" in worker or worker not in nodes):
                virtual_workers += [worker]
            else:
                node = nodes[worker]
                for resource, amount in self._worker_resources.items():
                    node[resource] -= amount
                found_workers_count += 1

        for _ in virtual_workers:
            for node in nodes.values():
                if all([node.get(resource, 0.0) >= amount
                        for resource, amount
                        in self._worker_resources.items()]):
                    for resource, amount in self._worker_resources.items():
                        if node[resource]:
                            node[resource] -= amount
                    found_workers_count += 1
                    break
            else:
                return False, found_workers_count
        return found_workers_count >= len(allocation), found_workers_count

    async def expand_cluster(self, current_workers, allocation):
        logging.info("Attempting to expand cluster to "
                     f"{len(allocation)} nodes")
        nodes = self.get_nodes()
        node_ips = {node["NodeManagerAddress"] for node in nodes}
        invalid_workers = {
            worker_id: ip for worker_id, ip in current_workers.items()
            if ip not in node_ips}
        if len(invalid_workers) == len(current_workers):
            rescale_timeout = FULL_RESCALE_TIMEOUT
            logging.info(
                "No live workers found. "
                "Waiting longer than specified for rescaling.")
        else:
            rescale_timeout = self._rescale_timeout

        worker_resources = [
            copy.deepcopy(self._worker_resources)
            for _ in range(len(allocation) + len(invalid_workers))]
        for bundle in worker_resources:
            bundle["CPU"] += 0.1
        sdk.request_resources(bundles=worker_resources)
        waited = 0.0
        logging.info(f"Waiting for up to {rescale_timeout} seconds for "
                     "nodes to be ready")
        while (waited < rescale_timeout and
               not self._cluster_ready(allocation)[0] and
               not self._force_immediate_allocation.is_set()):
            await asyncio.sleep(1.0)
            waited += 1.0
        ready, nodes = self._cluster_ready(allocation)
        logging.info(f"Found {nodes} available nodes")
        if not ready:
            allocation = (
                [node for node in allocation
                 if "adaptdl_virtual" not in node] +
                [node for node in allocation if "adaptdl_virtual" in node])
            return allocation[:nodes]
        else:
            return allocation


# This class is generally used remotely, but for testing purposes
# we also keep a local version. See https://stackoverflow.com/a/62309671
# from Robert Nishihara
class Controller():
    def __init__(self, cluster_size, rescale_timeout):
        super().__init__()
        logging.basicConfig(level=logging.INFO)
        self._job = None
        self._cluster = None
        self._cluster_size = cluster_size
        self._runner = None
        self._url = f"http://{services.get_node_ip_address()}:8080"
        self._spot_listener_tasks = {}
        self._ready = asyncio.Event()
        self._completed = asyncio.Event()
        self._rescale_timeout = rescale_timeout
        self._reschedule_queue = asyncio.Queue(maxsize=1)
        self._last_deployed = time.time()

    def get_url(self):
        return self._url

    async def run_controller(self):
        asyncio.create_task(self._run_app())
        asyncio.create_task(self._reschedule_listener())
        await self._completed.wait()
        await self._runner.cleanup()

    async def create_job(self, *args, **kwargs):
        await self._ready.wait()
        self._job = RayAdaptDLJob(*args, **kwargs)
        self._cluster = Cluster(
            self._job.worker_resources,
            self._rescale_timeout)
        await self._enqueue_reschedule(immediate=True)
        await self._job.completed.wait()
        self._completed.set()
        return self._job.status

    async def _reschedule_listener(self):
        while True:
            immediate = await self._reschedule_queue.get()
            logging.info("waiting for job")
            current_time = time.time()
            if not immediate and (current_time - self._last_deployed <= 300):
                await asyncio.sleep(int(current_time - self._last_deployed))
            await self._reschedule_jobs()
            self._last_deployed = time.time()
            logging.info("done rescheduling")

    async def _enqueue_reschedule(self, immediate=False):
        if not immediate:
            try:
                self._reschedule_queue.put_nowait(False)
                logging.info("enqueued reschedule")
            except asyncio.QueueFull:
                logging.info("enqueued reschedule failed, queue full")
                pass
        else:
            if not self._reschedule_queue.empty():
                self._reschedule_queue.get_nowait()
            await self._reschedule_queue.put(True)

    async def _reschedule_jobs(self):
        allocation = optimize(
            self._job, self._cluster, self._cluster_size)
        allocation = await self._cluster.expand_cluster(
            self._job.workers, allocation)

        worker_tasks = await self._job.update_workers(allocation)
        if worker_tasks:
            asyncio.create_task(
                self._handle_worker_failure(worker_tasks))
        self._cluster._force_immediate_allocation.clear()

    async def _spot_termination_handler(self, task):
        try:
            ip = await task
            self._cluster.mark_node_for_termination(ip)
            self._cluster._force_immediate_allocation.set()
            await self._enqueue_reschedule(immediate=True)
        except ray.exceptions.WorkerCrashedError:
            pass

    async def _handle_worker_failure(self, tasks):
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            if self._job._workers:
                logging.error("worker failure detected")
                logging.error(e)
                # Let the autoscheduler resolve any dead nodes
                await asyncio.sleep(60)
                # Todo: remove
                await self._enqueue_reschedule(immediate=True)

    async def register_worker(self, rank, ip):
        self._job._workers[rank] = ip
        if ip not in self._spot_listener_tasks:
            self._spot_listener_tasks[ip] = \
                listen_for_spot_termination.options(
                     num_cpus=0.1, resources={f"node:{ip}": 0.01}).remote()
            asyncio.create_task(
                self._spot_termination_handler(
                    self._spot_listener_tasks[ip]))

    async def register_checkpoint(self, checkpoint):
        self._job.register_checkpoint(checkpoint)
        return self._job._checkpoint_received.is_set()

    async def register_status(self, status):
        self._job.register_status(status)

    async def _handle_report(self, request):
        self._job.register_hints(await request.json())
        await self._enqueue_reschedule()
        return web.Response(text="metrics added")

    async def _handle_discover(self, request):
        # Long-polling endpoint used for discovering pod IPs for a given job.
        pod_ip_list = [
            ip for rank, ip in list(sorted(self._job.workers.items()))]
        # TODO: error handling
        return web.json_response(pod_ip_list)

    async def _run_app(self):
        port = 8080
        app = web.Application()
        app.add_routes([
            web.get('/discover/{namespace}/{name}/{group}',
                    self._handle_discover),
            web.put('/hints/{namespace}/{name}', self._handle_report),
        ])
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, services.get_node_ip_address(), port)
        await site.start()
        self._ready.set()
        return None


# For testing
_test_controller = Controller

# For general usage
Controller = ray.remote(num_cpus=1)(Controller)
