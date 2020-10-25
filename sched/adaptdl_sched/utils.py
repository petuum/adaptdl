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


async def patch_job_status(obj_api, namespace, name, patch):
    try:
        return await obj_api.patch_namespaced_custom_object_status(
            "adaptdl.petuum.com", "v1", namespace, "adaptdljobs", name, patch)
    except kubernetes.client.rest.ApiException as exc:
        if exc.status == 404:
            return None
        raise
