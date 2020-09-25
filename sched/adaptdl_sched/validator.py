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

import argparse
import json
import logging
import ssl

import kubernetes_asyncio as kubernetes

from aiohttp import web

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


class Validator(object):
    def __init__(self):
        self._core_api = kubernetes.client.CoreV1Api()
        self._app = web.Application()
        self._app.add_routes([
            web.get('/healthz', self._handle_healthz),
            web.post('/mutate', self._handle_mutate),
        ])

    def get_app(self):
        return self._app

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
                        "code": 400,
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

    def run(self, host, port, ssl_context=None):
        web.run_app(self.get_app(), host=host, port=port, 
                    ssl_context=ssl_context)


if __name__ == "__main__":
    logging.basicConfig()
    kubernetes.config.load_incluster_config()

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--tls-crt", type=str)
    parser.add_argument("--tls-key", type=str)
    args = parser.parse_args()

    if args.tls_crt and args.tls_key:
        ssl_context = ssl.SSLContext()
        ssl_context.load_cert_chain(args.tls_crt, args.tls_key)
    else:
        ssl_context = None

    validator = Validator()
    validator.run(args.host, args.port, ssl_context=ssl_context)
