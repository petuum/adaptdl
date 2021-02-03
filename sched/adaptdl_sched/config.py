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


import json
import os

ADAPTDL_PH_LABEL = 'adaptdl/placeholder'


def allowed_taints(taints):
    if not taints:
        return True
    return (len(taints) == 1 and taints[0].key == "petuum.com/nodegroup" and
            taints[0].value == "adaptdl")


def get_namespace():
    # for code running outside of AdaptDL
    if not os.path.exists("/var/run/secrets/kubernetes.io/serviceaccount/namespace"):  # noqa: E501
        return "default"
    with open("/var/run/secrets/kubernetes.io/serviceaccount/namespace") as f:
        return f.read()


def get_image():
    return os.environ["ADAPTDL_IMAGE"]


def get_adaptdl_deployment():
    return os.environ["ADAPTDL_SCHED_DEPLOYMENT"]


def get_supervisor_url():
    return os.environ["ADAPTDL_SUPERVISOR_URL"]


def get_supervisor_port():
    return os.getenv("ADAPTDL_SUPERVISOR_SERVICE_PORT", 8080)


def get_storage_subpath():
    return os.environ["ADAPTDL_STORAGE_SUBPATH"]


def get_adaptdl_version():
    return os.environ["ADAPTDL_SCHED_VERSION"]


def get_job_default_resources():
    val = os.getenv("ADAPTDL_JOB_DEFAULT_RESOURCES")
    return json.loads(val) if val is not None else None


def get_job_patch_pods():
    val = os.getenv("ADAPTDL_JOB_PATCH_PODS")
    return json.loads(val) if val is not None else None


def get_job_patch_containers():
    val = os.getenv("ADAPTDL_JOB_PATCH_CONTAINERS")
    return json.loads(val) if val is not None else None
