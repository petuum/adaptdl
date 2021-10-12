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


@ray.remote(num_cpus=1)
class Controller(AdaptDLJobMixin):
    def __init__(self, worker_resources, cluster_size,
                 checkpoint_timeout, rescale_timeout,
                 worker_port_offset, **kwargs):
        logging.basicConfig(level=logging.INFO)
        logging.info("launching controller")
        self._worker_resources = worker_resources
        self._max_cluster_size = cluster_size
        self._job_params = kwargs

        self._nodes = {}  # IP: boolean (running/marked for termination)
        # TODO: merge workers, worker_tasks
        self._workers = {}  # int: IP
        self._worker_tasks = {}
        self._last_metrics = None

        self._listener_tasks = {}

        self._runner = None
        self._pg = None
        self._url = f"http://{services.get_node_ip_address()}:8080"
        self._worker_port_offset = worker_port_offset

        self._checkpoint = None
        self._checkpoint_ref = None
        self._checkpoint_recieved = False

        self._iteration = 0

        self._completed = asyncio.Event()
        self._status = Status.RUNNING

        self._update_lock = asyncio.Lock()
        self._rescale_timeout = rescale_timeout
        self._checkpoint_timeout = checkpoint_timeout

    async def run_job(self):
        status, site = await asyncio.gather(
            self._run_controller(), self._run_app())
        return status

    async def _run_controller(self):
        metrics = []
        allocation = optimize(
            metrics, self._nodes,
            self._worker_resources, self._max_cluster_size)
        await asyncio.sleep(10)
        await self._create_workers(allocation)
        await self._completed.wait()
        await self._runner.cleanup()
        return self._status

    def _force_worker_checkpoint(self):
        logging.info("Checkpoint needed: stopping workers")
        for task in self._worker_tasks.values():
            ray.cancel(task, force=False)

    def _stop_workers(self):
        logging.info("Terminating workers")
        tasks = self._worker_tasks.values()
        self._worker_tasks = {}
        self._workers = {}
        for task in tasks:
            try:
                ray.cancel(task, force=True)
            except Exception:
                pass

    def _cluster_ready(self, allocation, terminated_instances):
        nodes = ray.nodes()
        nodes = {
            node["NodeManagerAddress"]: node["Resources"]
            for node in nodes
            if (node["NodeManagerAddress"] not in terminated_instances
                and node["Resources"])}

        virtual_nodes = []

        found_workers_count = 0
        for worker in allocation:
            if ("virtual" in worker or worker in terminated_instances or
                    worker not in nodes):
                virtual_nodes += [worker]
            else:
                node = nodes[worker]
                for resource, amount in self._worker_resources.items():
                    node[resource] -= amount
                found_workers_count += 1

        for _ in virtual_nodes:
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

        return True, found_workers_count

    async def _expand_cluster(self, allocation):
        logging.info("Attempting to expand cluster to "
                     f"{len(allocation)} nodes")
        terminated_nodes = {
            node for node, running in self._nodes.items() if not running}
        terminated_workers = {
            worker_id: ip for worker_id, ip in self._workers.items()
            if ip in terminated_nodes}
        worker_resources = [
            copy.deepcopy(self._worker_resources)
            for _ in range(len(allocation) + len(terminated_workers))]
        for bundle in worker_resources:
            bundle["CPU"] += 0.1
        sdk.request_resources(bundles=worker_resources)
        waited = 0.0
        logging.info(f"Waiting for up to {self._rescale_timeout} seconds for "
                     "nodes to be ready")
        while (waited < self._rescale_timeout and
               not self._cluster_ready(allocation, terminated_nodes)[0]):
            await asyncio.sleep(1.0)
            waited += 1.0
        ready, nodes = self._cluster_ready(allocation, terminated_nodes)
        logging.info(f"Found {nodes} available nodes")
        if not ready:
            allocation = (
                [node for node in allocation if "virtual" not in node] +
                [node for node in allocation if "virtual" in node])
            return allocation[:nodes]
        else:
            return allocation

    async def _handle_spot_instance_termination(self):
        self._force_worker_checkpoint()
        await self._update_workers(self._workers.values(), False)

    async def _handle_worker_failure(self, tasks):
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            if self._workers:
                logging.error("worker failure detected")
                logging.error(e)
                # Let the autoscheduler resolve any dead nodes
                await asyncio.sleep(60)
                await self._update_workers(
                    [f"virtual_{i}" for i in self._workers.keys()],
                    force_checkpoint=False, force_update=True)

    async def _create_workers(self, allocation):
        if self._pg:
            remove_placement_group(self._pg)
            await asyncio.sleep(10)
        self._nodes = {}
        logging.info(f"Creating {len(allocation)} worker tasks")
        self.placement_group_factory = AdaptDLJobMixin.allocation_to_pgf(
            allocation, self._worker_resources)
        self._pg = self.placement_group_factory()
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
                    self._url,
                    self._iteration,
                    self._checkpoint_ref,
                    self._worker_port_offset,
                    **self._job_params)
            for worker_index, node in enumerate(allocation)}

        asyncio.create_task(
            self._handle_worker_failure(list(self._worker_tasks.values())))
        self._iteration += 1

    async def _update_workers(self, allocation,
                              force_checkpoint=True, force_update=False):
        # have worker 0 be on not-spot
        # add way to restore a partially killed job
        async with self._update_lock:
            allocation = await self._expand_cluster(allocation)
            if set(self._workers.values()) != set(allocation) or force_update:
                if force_checkpoint:
                    self._force_worker_checkpoint()

                waited = 0.0
                while (self._worker_tasks and
                       waited <= self._checkpoint_timeout
                       and not self._checkpoint_recieved):
                    await asyncio.sleep(1)
                    waited += 1.0

                if waited >= self._checkpoint_timeout:
                    logging.warning("Waited for checkpoint, not found. "
                                    "Proceeding with previous checkpoint")

                self._checkpoint_recieved = False

                self._stop_workers()
                logging.info(allocation)
                await self._create_workers(allocation)
            else:
                logging.info("allocation unchanged, proceeding")

    async def register_checkpoint(self, obj):
        self._checkpoint = obj
        self._checkpoint_recieved = True
        self._checkpoint_ref = ray.put(obj)

    async def _spot_termination_continuation(self, task):
        try:
            ip = await task
            self._nodes[ip] = False
            await self._handle_spot_instance_termination()
        except ray.exceptions.WorkerCrashedError:
            pass

    async def register_worker(self, rank, ip):
        self._workers[rank] = ip
        self._nodes[ip] = True
        if ip in self._listener_tasks:
            ray.cancel(self._listener_tasks[ip])
            del self._listener_tasks[ip]
        self._listener_tasks[ip] = listen_for_spot_termination.options(
            num_cpus=0.1, resources={f"node:{ip}": 0.01}).remote()
        asyncio.create_task(
            self._spot_termination_continuation(self._listener_tasks[ip]))

    async def register_status(self, status):
        logging.info(f"Received status from worker: {Status(status)}")
        self._completed.set()
        self._status = Status(status)

    async def _handle_report(self, request):
        hints = await request.json()
        if not self._update_lock.locked():
            allocation = optimize(
                hints, self._nodes,
                self._worker_resources, self._max_cluster_size)
            asyncio.create_task(self._update_workers(allocation))
        return web.Response(text="metrics added")

    async def _handle_discover(self, request):
        # Long-polling endpoint used for discovering pod IPs for a given job.
        pod_ip_list = [ip for rank, ip in list(sorted(self._workers.items()))]
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
        return None
