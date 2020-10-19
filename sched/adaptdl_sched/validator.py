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
import sys

import kubernetes_asyncio as kubernetes

from aiohttp import web
from http import HTTPStatus

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


class Validator(object):
    def __init__(self):
        self._core_api = kubernetes.client.CoreV1Api()
        self._app = web.Application()
        self._app.add_routes([
            web.get('/healthz', self._handle_healthz),
            web.post('/validate', self._handle_validate),
        ])

    def get_app(self):
        return self._app

    def run(self, host, port, ssl_context=None):
        web.run_app(self.get_app(), host=host, port=port,
                    ssl_context=ssl_context)

    async def _handle_healthz(self, request):
        # Health check.
        return web.Response()

    async def _handle_validate(self, request):
        request_json = await request.json()
        if request_json["request"]["operation"] == "CREATE":
            response = await self._validate_create(request_json["request"])
        elif request_json["request"]["operation"] == "UPDATE":
            response = await self._validate_update(request_json["request"])
        else:
            response = {"allowed": True}
        LOG.info("%s %s/%s: %s",
                 request_json["request"]["operation"],
                 request_json["request"]["namespace"],
                 request_json["request"].get("name", "<none>"),
                 response)
        response["uid"] = request_json["request"]["uid"]
        return web.json_response({
            "apiVersion": "admission.k8s.io/v1",
            "kind": "AdmissionReview",
            "response": response,
        })

    async def _validate_create(self, request):
        job = request["object"]
        namespace = request["namespace"]
        template = {
            "metadata": {"name": "spec.template", "namespace": namespace},
            "template": job["spec"]["template"],
        }
        try:
            await self._core_api.create_namespaced_pod_template(
                namespace, template, dry_run="All")
        except kubernetes.client.rest.ApiException as exc:
            return {
               "allowed": False,
               "status": {
                   "code": HTTPStatus.UNPROCESSABLE_ENTITY,
                   "reason": "Invalid",
                   "message": json.loads(exc.body).get("message"),
               }
            }
        # If maxReplicas is provided, it should be >= minReplicas
        if job["spec"].get("maxReplicas", sys.maxsize) < \
                job["spec"].get("minReplicas", 0):
            return {
               "allowed": False,
               "status": {
                   "code": HTTPStatus.UNPROCESSABLE_ENTITY,
                   "reason": "Invalid",
                   "message": ("spec.maxReplicas must be greater "
                               "than or equal to spec.minReplicas")
               }
            }
        return {"allowed": True}

    async def _validate_update(self, request):
        if request["object"]["spec"] != request["oldObject"]["spec"]:
            return {
                "allowed": False,
                "status": {
                    "code": HTTPStatus.UNPROCESSABLE_ENTITY,
                    "reason": "Forbidden",
                    "message": "updates to job spec are forbidden",
                }
            }
        return {"allowed": True}


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
