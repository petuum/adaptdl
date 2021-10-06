from aiohttp import web

import asyncio
import copy
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

MOCK = False

namespace = "foo"
name = "foo"
group = "0"

job_key = str(uuid.uuid4())[:8]

RESCALE_TIMEOUT = 60


@ray.remote(num_cpus=1)
class Manager(AdaptDLJobMixin):
    def __init__(self, worker_resources,
                 cluster_size, worker_port_offset, **kwargs):
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

    async def run_job(self):
        status, site = await asyncio.gather(self._run_controller(), self._run_app())
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
        # TODO: replace with logging
        print("Checkpoint needed: stopping workers")
        for task in self._worker_tasks.values():
            ray.cancel(task, force=False)

    def _stop_workers(self):
        print("Terminating workers")
        for worker, task in self._worker_tasks.items():
            ray.cancel(task, force=True)

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
            if "virtual" not in worker:
                node = nodes[worker]
                for resource, amount in self._worker_resources.items():
                    node[resource] -= amount
                found_workers_count += 1
            else:
                virtual_nodes += [worker]

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
        print(f"Attempting to expand cluster to {len(allocation)} nodes")
        terminated_instances = {
            node for node, running in self._nodes.items() if not running}
        node_resources = [
            copy.deepcopy(self._worker_resources)
            for _ in range(len(allocation) + len(terminated_instances))]
        for bundle in node_resources:
            bundle["CPU"] += 0.1
        sdk.request_resources(bundles=node_resources)
        waited = 0.0
        while (waited < RESCALE_TIMEOUT and
               not self._cluster_ready(allocation, terminated_instances)[0]):
            await asyncio.sleep(1.0)
            waited += 1.0
        ready, nodes = self._cluster_ready(allocation, terminated_instances)
        print(f"Found {nodes} available nodes")
        if not ready:
            allocation = (
                [node for node in allocation if "virtual" not in node] +
                [node for node in allocation if "virtual" in node])
            return allocation[:nodes]
        else:
            return allocation

    async def _handle_spot_instance_termination(self):
        self._force_worker_checkpoint()
        await self._update_workers(self._nodes, False)

    async def _create_workers(self, allocation):
        if self._pg:
            remove_placement_group(self._pg)
            await asyncio.sleep(10)
        print(f"Creating {len(allocation)} worker tasks")
        self.placement_group_factory = AdaptDLJobMixin.allocation_to_pgf(
            allocation)
        self._pg = self.placement_group_factory()
        self._worker_tasks = {
            worker_index:
            run_adaptdl.options(placement_group=self._pg).remote(
                job_key,
                worker_index,
                len(allocation),
                self._url,
                self._iteration,
                self._checkpoint_ref,
                self._worker_port_offset,
                **self._job_params)
            for worker_index, node in enumerate(allocation)}

        self._iteration += 1

    async def _update_workers(self, allocation, force_checkpoint=True):
        # have worker 0 be on not-spot
        # add way to restore a partially killed job
        async with self._update_lock:
            allocation = await self._expand_cluster(allocation)
            if set(self._workers.values()) != set(allocation):
                if force_checkpoint:
                    self._force_worker_checkpoint()

                # TODO: fix this
                waited = 0.0
                while (self._worker_tasks and waited <= 120.0 and not self._checkpoint_recieved):
                    await asyncio.sleep(1)
                    waited += 1.0

                if waited >= 120.0:
                    print("Waited for checkpoint, not found. "
                          "Proceeding with previous checkpoint")

                self._checkpoint_recieved = False
                self._stop_workers()
                self._workers = {}
                self._worker_tasks = {}
                await self._create_workers(allocation)

    async def register_checkpoint(self, obj):
        self._checkpoint = obj
        self._checkpoint_recieved = True
        self._checkpoint_ref = ray.put(obj)

    async def _spot_termination_continuation(self, task):
        ip = await task
        self._nodes[ip] = False
        await self._handle_spot_instance_termination()

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
        print(f"Received status from worker: {Status(status)}")
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
            web.put(f'/hints/{namespace}/{name}', self._handle_report),
        ])
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, services.get_node_ip_address(), 8080)
        await site.start()
        return None
