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
import json
import logging
import ssl


LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


class Validator(object):
    def __init__(self, port, host='0.0.0.0'):
        self._host = host
        self._port = port
        self._core_api = kubernetes.client.CoreV1Api()

    async def _handle_healthz(self, request):
        # Health check.
        return web.Response()

    async def _handle_mutate(self, request):
        review = await request.json()
        LOG.info(review)
        job = review["request"]["object"]
        namespace = review["request"]["namespace"]
        template = {
            "metadata": {"name": "spec.template", "namespace": namespace},
            "template": job["spec"]["template"],
        }
        try:
            await self._core_api.create_namespaced_pod_template(
                namespace, template, dry_run="All")
        except kubernetes.client.rest.ApiException as exc:
            return web.json_response({
                "apiVersion": "admission.k8s.io/v1",
                "kind": "AdmissionReview",
                "response": {
                    "uid": review["request"]["uid"],
                    "allowed": False,
                    "status": {
                        "code": 403,
                        "reason": "Invalid",
                        "message": json.loads(exc.body).get("message"),
                    }
                }
            })
        return web.json_response({
            "apiVersion": "admission.k8s.io/v1",
            "kind": "AdmissionReview",
            "response": {
                "uid": review["request"]["uid"],
                "allowed": True,
            }
        })

    def run(self):
        self.app = web.Application()
        self.app.add_routes([
            web.get('/healthz', self._handle_healthz),
            web.post('/mutate', self._handle_mutate),
        ])
        LOG.info("%s %s", self._host, self._port)
        ssl_ctx = ssl.SSLContext()
        ssl_ctx.load_cert_chain("/etc/webhook/certs/tls.crt",
                                "/etc/webhook/certs/tls.key")
        web.run_app(self.app, host=self._host, port=self._port,
                    ssl_context=ssl_ctx)


if __name__ == "__main__":
    logging.basicConfig()
    kubernetes.config.load_incluster_config()

    validator = Validator(8443)
    validator.run()
