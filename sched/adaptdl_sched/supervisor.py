# Copyright 2020 Petuum, Inc. All Rights Reserved.
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


import kubernetes_asyncio as kubernetes
from aiohttp import web
import logging
from adaptdl.sched_hints import SCHED_HINTS
from adaptdl_sched.config import get_supervisor_port


LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


class Supervisor:
    """
    Supervisor provides a simple REST interface for several functionalities.
    Currently, it has two endpoints:
    1.  /hints for jobs to send scheduling hints.
    2.  /discover for finding the pod IPs of a job.
    """

    def __init__(self, port, host='0.0.0.0'):
        self._host = host
        self._port = port
        self._core_api = kubernetes.client.CoreV1Api()
        self._objs_api = kubernetes.client.CustomObjectsApi()

    async def _handle_healthz(self, request):
        # Health check.
        return web.Response()

    async def _handle_discover(self, request):
        # Long-polling endpoint used for discovering pod IPs for a given job.
        namespace = request.match_info["namespace"]
        name = request.match_info["name"]
        group = request.match_info["group"]
        timeout = int(request.query.get("timeout", "30"))
        pod_ip_list = None
        async with kubernetes.watch.Watch() as w:
            stream = w.stream(self._core_api.list_namespaced_pod, namespace,
                              label_selector="adaptdl/job={}".format(name),
                              field_selector="status.podIP!=",
                              timeout_seconds=timeout)
            async for event in stream:
                pod = event["object"]
                replicas = int(pod.metadata.annotations["adaptdl/replicas"])
                rank = int(pod.metadata.annotations["adaptdl/rank"])
                if pod.metadata.annotations["adaptdl/group"] == group:
                    if pod_ip_list is None:
                        pod_ip_list = [None] * replicas
                    pod_ip_list[rank] = pod.status.pod_ip
                    if all(pod_ip is not None for pod_ip in pod_ip_list):
                        return web.json_response(pod_ip_list)
        return web.json_response(status=408)  # Timeout.

    async def _handle_report(self, request):
        namespace = request.match_info['namespace']
        name = request.match_info['name']
        hints = await request.json()
        # Drop all unrecognized fields. TODO: validate each client-sent field.
        hints = {k: hints[k] for k in SCHED_HINTS if k in hints}
        # Patch only the train field to avoid conflicts with controller.
        patch = {"status": {"train": hints}}
        LOG.info("Patch AdaptDLJob %s/%s: %s", namespace, name, patch)
        await self._objs_api.patch_namespaced_custom_object_status(
            "adaptdl.petuum.com", "v1", namespace, "adaptdljobs", name, patch)
        return web.Response()

    def run(self):
        self.app = web.Application()
        self.app.add_routes([
            web.get('/healthz', self._handle_healthz),
            web.get('/discover/{namespace}/{name}/{group}',
                    self._handle_discover),
            web.put('/hints/{namespace}/{name}', self._handle_report),
        ])
        LOG.info("%s %s", self._host, self._port)
        web.run_app(self.app, host=self._host, port=self._port)


if __name__ == "__main__":
    logging.basicConfig()
    kubernetes.config.load_incluster_config()

    supervisor = Supervisor(get_supervisor_port())
    supervisor.run()
