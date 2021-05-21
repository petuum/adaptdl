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


import copy
import adaptdl_sched.config as config
import kubernetes_asyncio as kubernetes
import math

from typing import List, Dict, Union


def get_node_unrequested(
        node: kubernetes.client.V1Node,
        pods: List[kubernetes.client.V1Pod],
        ) -> Dict[str, int]:
    """
    Get the amount of node resources which are unrequested by a list of pods.

    Args:
        node: The node to get unrequested resources for.
        pods: Pods which may request resources from the node.

    Returns:
        Mapping from resource names (eg. cpu, memory, nvidia.com/gpu) to
        an integer amount of the resource which is unrequested on the node.
        The integer amounts are discretized to the smallest possible unit of
        each resource.
    """
    ret = {key: _discretize_resource(key, val)
           for key, val in node.status.allocatable.items()}
    for pod in pods:
        if pod.spec.node_name != node.metadata.name \
                or pod.status.phase in ["Succeeded", "Failed"]:
            continue
        requests = get_pod_requests(pod.spec)
        for key, val in requests.items():
            if key in ret:
                ret[key] -= val
    # Note that resources can get negative, as Non-terminated Pods include
    # both Running + Pending pods, and there may be Pending pods yet to be
    # scheduled specifically on each node.
    return {key: val for key, val in ret.items() if val > 0}


def get_pod_requests(
        pod_spec: Union[kubernetes.client.V1PodSpec, dict],
        ) -> Dict[str, int]:
    """
    Get the aggregate amount of resources requested by all containers in a pod.

    Args:
        pod_spec: The pod to get requested resources for.

    Returns:
        Mapping from resource names (eg. cpu, memory, nvidia.com/gpu) to
        an integer amount of the resource which is requested by the pod.
        The integer amounts are discretized to the smallest possible unit of
        each resource.
    """
    # Aggregate resource requests of a pod.
    if not isinstance(pod_spec, dict):
        pod_spec = pod_spec.to_dict()
    overcommitable_resources = ["cpu", "memory", "ephemeral-storage"]
    pod_requests = {"pods": 1}
    for container in pod_spec["containers"]:
        if not container.get("resources"):
            continue
        requests = container["resources"].get("requests") or {}
        for key in overcommitable_resources:
            val = requests.get(key)
            if val is not None:
                pod_requests.setdefault(key, 0)
                pod_requests[key] += _discretize_resource(key, val)
        # For non-overcommitable resources (including extended resources),
        # requests are equal to limits.
        limits = container["resources"].get("limits") or {}
        for key, val in limits.items():
            if key not in overcommitable_resources and val is not None:
                pod_requests.setdefault(key, 0)
                pod_requests[key] += _discretize_resource(key, val)
    return {key: val for key, val in pod_requests.items() if val > 0}


def set_default_resources(pod_spec: dict) -> dict:
    """
    Set the default resources for a given AdaptDLJob spec.

    Args:
        pod_spec: The pod spec to set default resources for.

    Returns:
        A new pod spec with default resources set.
    """
    pod_spec = copy.deepcopy(pod_spec)
    default_resources = config.get_job_default_resources()
    if default_resources is not None:
        # Set default resources for main container.
        container = pod_spec["containers"][0]
        container.setdefault("resources", {})
        if default_resources.get("requests") is not None:
            container["resources"].setdefault("requests", {})
            for k, v in default_resources["requests"].items():
                container["resources"]["requests"].setdefault(k, v)
        if default_resources.get("limits") is not None:
            container["resources"].setdefault("limits", {})
            for k, v in default_resources["limits"].items():
                container["resources"]["limits"].setdefault(k, v)
    return pod_spec


def _discretize_resource(name, value):
    # Normalize to the smallest integral units, try to handle all cases in
    # https://kubernetes.io/docs/concepts/configuration/manage-compute-resources-container
    factor = 1000 if name == "cpu" else 1  # Convert CPU to mCPU.
    # https://github.com/kubernetes/apimachinery/blob/41f3f7bd69163010f79a448955beb2260b91811b/pkg/api/resource/quantity.go#L30
    if isinstance(value, str) and value.endswith("m"):
        factor /= 1000
        value = value[:-1]
    elif isinstance(value, str):
        for idx, unit in enumerate(["k", "M", "G", "T", "P", "E"]):
            if value.endswith(unit):
                factor *= 1000 ** (idx + 1)
                value = value[:-1]
        for idx, unit in enumerate(["Ki", "Mi", "Gi", "Ti", "Pi", "Ei"]):
            if value.endswith(unit):
                factor *= 1024 ** (idx + 1)
                value = value[:-2]
    return math.ceil(float(value) * factor)
