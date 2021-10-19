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

from adaptdl_job_mixin import AdaptDLJobMixin
from adaptdl_allocator import AdaptDLAllocator

from adaptdl.goodput import PerfParams, GradParams

import ray
import ray.autoscaler.sdk as sdk
import ray.services as services

from optimizer import optimize
from utils import Status
from worker import listen_for_spot_termination, run_adaptdl

from ray.util.placement_group import (
    remove_placement_group
)


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
        self._checkpoint_received = False
        self._checkpoint_lock = asyncio.Lock()
        self._checkpoint_timeout = checkpoint_timeout

        self._iteration = 0

        self.completed = asyncio.Event()
        self.status = Status.RUNNING

        self._placement_group_factory = None

        self._update_lock = asyncio.Lock()

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
        return self.hints

    def _allocation_in_use(self):
        return self._workers

    async def _handle_worker_failure(self, tasks):
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            if self._workers:
                logging.error("worker failure detected")
                logging.error(e)
                # Let the autoscheduler resolve any dead nodes
                await asyncio.sleep(60)
                await self.update_workers(
                    [f"virtual_{i}" for i in self._workers.keys()],
                    force_checkpoint=False, force_update=True)

    async def _create_workers(self, allocation):
        if self._pg:
            remove_placement_group(self._pg)
        logging.info(f"Creating {len(allocation)} worker tasks")
        self._placement_group_factory = AdaptDLJobMixin.allocation_to_pgf(
            allocation, self._worker_resources)
        self._pg = self._placement_group_factory()
        self._worker_tasks = {
            worker_index:
            run_adaptdl.options(
                num_cpus=self._worker_resources["CPU"],
                num_gpus=self._worker_resources["GPU"],
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

        asyncio.create_task(
            self._handle_worker_failure(list(self._worker_tasks.values())))
        self._running = True
        self._iteration += 1

    async def update_workers(self, allocation,
                             force_checkpoint=True, force_update=False):
        logging.info(
            f"Updating workers. before: {len(self._workers.values())} workers"
            f", after: {len(allocation)} workers")
        async with self._update_lock:
            if set(self._workers.values()) != set(allocation) or force_update:
                if force_checkpoint:
                    await self.force_worker_checkpoint()

                waited = 0.0
                while (self._worker_tasks and
                       waited <= self._checkpoint_timeout
                       and not self._checkpoint_received):
                    await asyncio.sleep(1)
                    waited += 1.0

                if waited >= self._checkpoint_timeout:
                    logging.warning("Waited for checkpoint, not found. "
                                    "Proceeding with previous checkpoint")
                self._checkpoint_received = False
                self._stop_workers()
                await self._create_workers(allocation)
            else:
                logging.info("Allocation unchanged, proceeding")

    def register_checkpoint(self, obj):
        self._checkpoint = obj
        self._checkpoint_received = True
        self._checkpoint_ref = ray.put(obj)

    def register_status(self, status):
        self._status = Status(status)
        if self._status != Status.RUNNING:
            self.completed.set()


class Cluster():
    def __init__(self, worker_resources, rescale_timeout):
        self._worker_resources = worker_resources
        self._rescale_timeout = rescale_timeout
        self._terminating_nodes = set()

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
            if ("virtual" in worker or worker not in nodes):
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
        invalid_workers = {
            worker_id: ip for worker_id, ip in current_workers.items()
            if ip not in nodes}
        if len(invalid_workers) == len(current_workers):
            rescale_timeout = 6000
            logging.info(
                "No live workers found. "
                "Waiting longer than specified for rescaling.")
        else:
            rescale_timeout = self._rescale_timeout

        worker_resources = [
            copy.deepcopy(self._worker_resources)
            for _ in range(len(allocation) + len(invalid_workers) + 1)]
        for bundle in worker_resources:
            bundle["CPU"] += 0.1
        sdk.request_resources(bundles=worker_resources)
        waited = 0.0
        logging.info(f"Waiting for up to {rescale_timeout} seconds for "
                     "nodes to be ready")
        while (waited < rescale_timeout and
               not self._cluster_ready(allocation)[0]):
            await asyncio.sleep(1.0)
            waited += 1.0
        ready, nodes = self._cluster_ready(allocation)
        logging.info(f"Found {nodes} available nodes")
        if not ready:
            allocation = (
                [node for node in allocation if "virtual" not in node] +
                [node for node in allocation if "virtual" in node])
            return allocation[:nodes]
        else:
            return allocation


@ray.remote(num_cpus=1)
class Controller(AdaptDLAllocator):
    def __init__(self, cluster_size, rescale_timeout):
        super().__init__()
        logging.basicConfig(level=logging.INFO)
        self._job = None
        self._cluster = None
        self._cluster_size = cluster_size
        self._runner = None
        self._url = f"http://{services.get_node_ip_address()}:8080"
        self._rescale_lock = asyncio.Lock()
        self._spot_listener_tasks = {}
        self._ready = asyncio.Event()
        self._completed = asyncio.Event()
        self._rescale_timeout = rescale_timeout

    def get_url(self):
        return self._url

    async def run_controller(self):
        asyncio.create_task(self._run_app())
        await self._completed.wait()
        await self._runner.cleanup()

    async def create_job(self, *args, **kwargs):
        await self._ready.wait()
        self._job = RayAdaptDLJob(*args, **kwargs)
        self._cluster = Cluster(
            self._job.worker_resources,
            self._rescale_timeout)
        await self._reschedule_jobs()
        await self._job.completed.wait()
        self._completed.set()
        return self._job.status

    async def _reschedule_jobs(self, immediate_checkpoint=False):
        if immediate_checkpoint:
            await self._job.force_worker_checkpoint()
        elif self._rescale_lock.locked():
            return
        async with self._rescale_lock:
            allocation = optimize(
                self._job, self._cluster, self._cluster_size)
            allocation = await self._cluster.expand_cluster(
                self._job.workers, allocation)

            await self._job.update_workers(
                allocation, force_checkpoint=(not immediate_checkpoint))

    async def _spot_termination_continuation(self, task):
        try:
            ip = await task
            self._cluster.mark_node_for_termination(ip)
            await self._reschedule_jobs(immediate_checkpoint=True)
        except ray.exceptions.WorkerCrashedError:
            pass

    async def register_worker(self, rank, ip):
        self._job._workers[rank] = ip
        if ip not in self._spot_listener_tasks:
            self._spot_listener_tasks[ip] = \
                listen_for_spot_termination.options(
                     num_cpus=0.1, resources={f"node:{ip}": 0.01}).remote()
            asyncio.create_task(
                self._spot_termination_continuation(
                    self._spot_listener_tasks[ip]))

    async def register_checkpoint(self, checkpoint):
        self._job.register_checkpoint(checkpoint)
        return self._job._checkpoint_received

    async def register_status(self, status):
        self._job.register_status(status)

    async def _handle_report(self, request):
        self._job.register_hints(await request.json())
        asyncio.create_task(self._reschedule_jobs())
        return web.Response(text="metrics added")

    async def _handle_discover(self, request):
        # Long-polling endpoint used for discovering pod IPs for a given job.
        pod_ip_list = [
            ip for rank, ip in list(sorted(self._job.workers.items()))]
        # TODO: error handling
        return web.json_response(pod_ip_list)

    async def _run_app(self):
        app = web.Application()
        app.add_routes([
            web.get('/discover/{namespace}/{name}/{group}',
                    self._handle_discover),
            web.put('/hints/{namespace}/{name}', self._handle_report),
        ])
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, services.get_node_ip_address(), 8080)
        await site.start()
        self._ready.set()
        return None
