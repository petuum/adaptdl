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


import asyncio
import collections
import copy
import jsonpatch
import kubernetes_asyncio as kubernetes
import logging

import adaptdl_sched.k8s_templates as templates
import adaptdl_sched.config as config

from datetime import datetime, timezone
from prometheus_client import Counter, Summary

from adaptdl_sched.resources import set_default_resources
from adaptdl_sched.utils import patch_job_status

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

# Prometheus metrics
METRICS_KWARGS = dict(namespace="adaptdl", subsystem="sched")
JOB_SUBMISSION_COUNT = Counter(
    "job_submission_count", "Number of submitted jobs", **METRICS_KWARGS)
JOB_COMPLETION_TIME = Summary(
    "job_completion_time", "Duration of completed jobs",
    labelnames=["status"], **METRICS_KWARGS)


class AdaptDLController(object):
    """
    The main controller responsible for the overall AdaptDLJob lifecycle.
    Essentially, it keeps a queue of AdaptDLJobs whose states may need to be
    synchronized. It watches for events such as pod status changes and
    allocation changes and enqueues any potentially affects AdaptDLJobs. A
    worker coroutine is responsible for processing AdaptDLJobs from the queue
    and guarantees that a single AdaptDLJob is never processed concurrently.
    """

    def __init__(self):
        self._core_api = kubernetes.client.CoreV1Api()
        self._objs_api = kubernetes.client.CustomObjectsApi()
        self._custom_resource = ("adaptdl.petuum.com", "v1",
                                 "", "adaptdljobs")
        self._queue = asyncio.Queue()

    async def run(self):
        # Create service if it doesn't already exist.
        # FIXME: initialize allocations
        await asyncio.gather(
            self._watch_jobs(),
            self._watch_pods(),
            self._sync_worker()
        )

    async def _watch_jobs(self):
        # Watch for changes to AdaptDLJobs and enqueue them to be synced.
        # Perform a full reconcile after every timeout.
        async with kubernetes.watch.Watch() as watch:
            while True:
                async for event in watch.stream(
                        self._objs_api.list_namespaced_custom_object,
                        *self._custom_resource, timeout_seconds=60):
                    job_name = event["object"]["metadata"]["name"]
                    namespace = event["object"]["metadata"]["namespace"]
                    await self._queue.put((namespace, job_name))

    async def _watch_pods(self):
        # Watch for changes to pods and enqueue their AdaptDLJobs to be synced.
        # Perform a full reconcile after every timeout.
        async with kubernetes.watch.Watch() as watch:
            while True:
                async for event in watch.stream(
                        self._core_api.list_namespaced_pod, "",
                        label_selector="adaptdl/job", timeout_seconds=60):
                    pod = event["object"]
                    job_name = pod.metadata.labels["adaptdl/job"]
                    namespace = pod.metadata.namespace
                    await self._queue.put((namespace, job_name))

    async def _sync_worker(self):
        while True:
            (namespace, name) = await self._queue.get()
            await self._sync_job(namespace, name)
            self._queue.task_done()

    async def _sync_job(self, namespace, job_name):
        current_ts = datetime.now(timezone.utc)
        job, pods = await self._get_job_and_pods(namespace, job_name)
        if job is not None:
            job = await self._validate_pods(job, pods, current_ts)
        if job is None:  # Not Found, presumably was deleted.
            await self._delete_pods(pods)
            return
        # Use ChainMap to record updates to the job status fields.
        job["status"] = collections.ChainMap({}, job.get("status", {}))
        # Get the current phase of the job, None if no phase was set.
        allocation = job["status"].get("allocation", [])
        phase = job["status"].setdefault("phase", "Pending")
        replicas = job["status"].get("replicas", 0)
        preemptible = job["spec"].get("preemptible", True)
        if (completion_status := self._detect_completion(pods, preemptible)):
            # Job is already completed.
            job["status"].update(completion_status)
            job["status"].setdefault("completionTimestamp", current_ts)
            job["status"]["allocation"] = allocation = []
            await self._delete_pods(  # Keep failed pods for debug purposes.
                [pod for pod in pods if pod.status.phase != "Failed"])
        elif phase == "Pending":
            if allocation and not pods:
                # Start the next group of pods.
                job["status"]["phase"] = "Starting"
        elif phase == "Starting":
            # FIXME: In case if a pod experiences indefinite ImagePullBackOff,
            # the job can get stuck in the Starting phase
            if (self._count_scheduled_pods(pods) != replicas and
                    self._detect_restart(pods, allocation)) or not allocation:
                # Allocator changed allocation based on new information about
                # resource availability
                job["status"]["phase"] = "Stopping"
            elif allocation and not pods:
                # Start the next group of pods.
                job["status"]["group"] = job["status"].get("group", -1) + 1
                try:
                    new_pods = []
                    for rank in range(len(allocation)):
                        pod = await self._create_pod(
                            job["metadata"],
                            job["spec"]["template"], allocation,
                            job["status"]["group"], rank)
                        new_pods.append(pod)
                except kubernetes.client.rest.ApiException as e:
                    LOG.warning(f"Failed to create pod for {job_name}. "
                                "Setting job status to failed")
                    job["status"]["phase"] = "Failed"
                    job["status"]["reason"] = "PodCreationError"
                    job["status"]["message"] = str(e)
                    await self._delete_pods(new_pods)
            elif len(pods) != replicas:
                # Controller restarted before we can spawn all pods
                job["status"]["phase"] = "Stopping"
            elif self._count_ready_pods(pods) == replicas:
                # all pods are running
                job["status"]["phase"] = "Running"
        elif phase == "Running":
            if self._detect_restart(pods, allocation) or \
                    not pods:
                # 1. Reallocation OR 2. the controller restarted before
                # we can update phase to Stopping
                job["status"]["phase"] = "Stopping"
        elif phase == "Stopping":
            if pods:
                await self._delete_pods(pods)
            else:
                # all pods successfully deleted
                job["status"]["phase"] = "Pending"
        # Set replicas and ready replicas.
        if allocation:
            job["status"]["replicas"] = len(allocation)
            job["status"]["readyReplicas"] = self._count_ready_pods(pods)
        else:
            job["status"]["allocation"] = None
            job["status"]["replicas"] = None
            job["status"]["readyReplicas"] = None
        # Apply changes to AdaptDLJob status.
        patch = {"status": {k: v for k, v in job["status"].maps[0].items()
                            if v != job["status"].maps[1].get(k)}}
        if patch["status"]:
            LOG.info("Patch AdaptDLJob %s: %s", job_name, patch)
            await patch_job_status(self._objs_api, namespace, job_name, patch)

    def _count_ready_pods(self, pods):
        count = 0
        for pod in pods:
            if pod.status.container_statuses and \
                    all(stat.ready for stat in pod.status.container_statuses):
                count += 1
        return count

    def _count_scheduled_pods(self, pods):
        count = 0
        for pod in pods:
            if pod.status.conditions and \
                    any(cond.type == "PodScheduled" and cond.status == "True"
                        for cond in pod.status.conditions):
                count += 1
        return count

    async def _get_job_and_pods(self, namespace, name):
        # Find all pods owned by this AdaptDLJob.
        pods = await self._core_api.list_namespaced_pod(
            namespace, label_selector=f"adaptdl/job={name}")
        pods = pods.items
        # Read a fresh copy of the AdaptDLJob.
        try:
            job = await self._objs_api.get_namespaced_custom_object(
                "adaptdl.petuum.com", "v1", namespace, "adaptdljobs", name)
        except kubernetes.client.rest.ApiException as exc:
            if exc.status == 404:
                return None, pods
            raise  # Unexpected error.
        return job, pods

    async def _validate_pods(self, job, pods, current_ts):
        namespace = job["metadata"]["namespace"]
        name = job["metadata"]["name"]
        patch_status = {}
        # Validate pods for job.
        group_list = []
        replicas_list = []
        rank_list = []
        for pod in pods:
            # Check the expected annotations exist.
            try:
                group = int(pod.metadata.annotations["adaptdl/group"])
                replicas = int(pod.metadata.annotations["adaptdl/replicas"])
                rank = int(pod.metadata.annotations["adaptdl/rank"])
                node = pod.metadata.annotations["adaptdl/node"]
            except (KeyError, ValueError):
                patch_status["phase"] = "Failed"
                patch_status["reason"] = "Invalid"
                patch_status["message"] = \
                    f"invalid annotations for pod {pod.metadata.name}"
                break
            # Check the pod is running on the correct node.
            if pod.spec.node_name and pod.spec.node_name != node:
                patch_status["phase"] = "Failed"
                patch_status["reason"] = "Invalid"
                patch_status["message"] = \
                    f"incorrect node for pod {pod.metadata.name}"
                break
            group_list.append(group)
            replicas_list.append(replicas)
            rank_list.append(rank)
        else:
            # Check all pods have the same group and replicas values, and no
            # pod has a rank which is out of range.
            if len(set(group_list)) > 1 or len(set(replicas_list)) > 1 or \
                    any(rank >= replicas_list[0] for rank in rank_list):
                patch_status["phase"] = "Failed"
                patch_status["reason"] = "Invalid"
                patch_status["message"] = "inconsistent pods in group"
        if patch_status:
            return await patch_job_status(self._objs_api, namespace, name,
                                          {"status": patch_status})
        return job

    def _detect_completion(self, pods, preemptible):
        if not pods:
            return {}

        # Check if all pods succeeded.
        for pod in pods:
            replicas = int(pod.metadata.annotations["adaptdl/replicas"])
            if len(pods) != replicas or pod.status.phase != "Succeeded":
                break
        else:
            return {"phase": "Succeeded"}

        # Check for fatal failures in a list of pods. Non-fatal failures can be
        # due to evictions or temporary resource unavailability.
        def any143(pod):
            if not pod.status.container_statuses:
                return False
            for status in pod.status.container_statuses:
                if (status.state.terminated and
                        status.state.terminated.exit_code == 143):
                    return True
            return False

        for pod in pods:
            if pod.status.phase == "Unknown":
                # This can happen if there's something wrong with the node
                # or kubelet assigned to this pod.
                LOG.warning("Unknown status for pod %s", pod.metadata.name)
            elif pod.status.phase != "Failed":
                continue
            if pod.status.reason == "UnexpectedAdmissionError":
                # This can happen if a non-AdaptDL pod claims the node's
                # resources before this pod could bind to that node.
                LOG.warning("UnexpectedAdmissionError for pod %s: %s",
                            pod.metadata.name, pod.status.message)
            elif str(pod.status.reason).startswith("Outof"):
                # we might be temporarily out of pods on this node
                LOG.warning(f"Pod {pod.metadata.name} is {pod.status.reason} "
                            f"on {pod.spec.node_name}")
            elif preemptible and (pod.metadata.deletion_timestamp is not None
                                  or any143(pod)):
                # This pod was intentionally terminated.
                LOG.warning(f"Pod {pod.metadata.name} terminated")
            else:
                return {"phase": "Failed", "reason": "PodFailure",
                        "message": f"{pod.metadata.name} {pod.status.phase}"}
        return {}

    def _detect_restart(self, pods, allocation):
        for pod in pods:
            replicas = int(pod.metadata.annotations["adaptdl/replicas"])
            rank = int(pod.metadata.annotations["adaptdl/rank"])
            if replicas != len(allocation) or \
                    (pod.metadata.annotations["adaptdl/node"] !=
                     allocation[rank]):
                return True
        return False

    async def _delete_pods(self, pods):
        if not pods:
            return
        results, names = [], []
        for pod in pods:
            if not pod.metadata.deletion_timestamp:
                results.append(self._core_api.delete_namespaced_pod(
                               pod.metadata.name, pod.metadata.namespace))
                names.append(pod.metadata.name)
        if results:
            LOG.info(f"Deleting {names}")
            await asyncio.gather(*results, return_exceptions=True)

    async def _create_pod(self, job_metadata, pod_template,
                          allocation, group, rank):
        node = await self._core_api.read_node(allocation[rank])
        pod = copy.deepcopy(pod_template)
        pod["apiVersion"] = "v1"
        pod["kind"] = "Pod"
        pod.setdefault("metadata", {})
        pod["metadata"]["name"] = self._get_pod_name(job_metadata, group, rank)
        pod["metadata"].setdefault("labels", {})
        pod["metadata"]["ownerReferences"] = \
            templates.owner_reference_template(
                job_metadata["namespace"], job_metadata["name"],
                job_metadata["uid"], kind="AdaptDLJob")
        pod["metadata"]["labels"]["adaptdl"] = "true"
        pod["metadata"]["labels"]["adaptdl/job"] = job_metadata["name"]
        pod["metadata"]["labels"]["petuum.com/nodegroup"] = "all"
        pod["metadata"].setdefault("annotations", {})
        pod["metadata"]["annotations"]["adaptdl/replicas"] = \
            str(len(allocation))
        pod["metadata"]["annotations"]["adaptdl/group"] = str(group)
        pod["metadata"]["annotations"]["adaptdl/rank"] = str(rank)
        pod["metadata"]["annotations"]["adaptdl/node"] = node.metadata.name
        pod["spec"].setdefault("nodeSelector", {})
        pod["spec"]["hostname"] = f"{job_metadata['name']}-{group}-{rank}"
        pod["spec"]["nodeSelector"]["kubernetes.io/hostname"] = \
            node.metadata.labels["kubernetes.io/hostname"]
        pod["spec"]["restartPolicy"] = "Never"
        pod["spec"].setdefault("volumes", [])
        pod["spec"]["volumes"].append({
            "name": "adaptdl-shm",
            "emptyDir": {
                "medium": "Memory",
            },
        })
        pod["spec"] = set_default_resources(pod["spec"])
        for idx, container in enumerate(pod["spec"]["containers"]):
            container.setdefault("volumeMounts", [])
            container["volumeMounts"].append({
                "name": "adaptdl-shm",
                "mountPath": "/dev/shm",
            })
            container.setdefault("env", [])
            container["env"].append({
                "name": "ADAPTDL_JOB_ID",
                "value": "{}/{}".format(job_metadata["namespace"],
                                        job_metadata["name"]),
            })
            container["env"].append({
                "name": "ADAPTDL_MASTER_PORT",
                "value": str(47000 + group),
            })
            container["env"].append({
                "name": "ADAPTDL_NUM_NODES",
                "value": str(len(set(allocation))),
            })
            container["env"].append({
                "name": "ADAPTDL_NUM_RESTARTS",
                "value": str(group),
            })
            container["env"].append({
                "name": "ADAPTDL_NUM_REPLICAS",
                "value": str(len(allocation)),
            })
            container["env"].append({
                "name": "ADAPTDL_REPLICA_RANK",
                "value": str(rank),
            })
            container["env"].append({
                "name": "ADAPTDL_SUPERVISOR_URL",
                "value": config.get_supervisor_url(),
            })
            container["env"].append({
                "name": "ADAPTDL_SCHED_VERSION",
                "value": config.get_adaptdl_version(),
            })
            resources = container.get("resources", {})
            if not resources.get("limits", {}).get("nvidia.com/gpu"):
                # Apparently if a container doesn't ask for any nvidia.com/gpu,
                # then it will be allocated all of them???
                # https://github.com/NVIDIA/k8s-device-plugin/issues/61
                # https://github.com/NVIDIA/k8s-device-plugin
                # AdaptDL overrides this behavior.
                container["env"].append({
                    "name": "NVIDIA_VISIBLE_DEVICES",
                    "value": "none",
                })
        pod = self._patch_pods_and_containers(pod)
        return await self._core_api.create_namespaced_pod(
            job_metadata["namespace"], pod)

    def _patch_pods_and_containers(self, pod):
        pod_patch = config.get_job_patch_pods()
        container_patch = config.get_job_patch_containers()
        if pod_patch:
            pod = jsonpatch.apply_patch(pod, pod_patch)
        if container_patch:
            for idx, container in enumerate(pod["spec"]["containers"]):
                pod["spec"]["containers"][idx] = (
                    jsonpatch.apply_patch(container, container_patch))
        return pod

    def _get_pod_name(self, job_metadata, group, rank):
        job_name = job_metadata["name"]
        job_uid = job_metadata["uid"]
        return f"{job_name}-{job_uid}-{group}-{rank}"
