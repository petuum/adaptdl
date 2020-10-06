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

import kubernetes
import re

SUPPORTED_PROVISIONERS = (r'microk8s.io/hostpath',
                          r'.*cephfs.csi.ceph.com',
                          r'\befs\b')


def get_storageclass(name=None):
    api = kubernetes.client.StorageV1Api()
    if name is not None:
        return api.read_storage_class(name)
    # Find default storageclass.
    sc_list = api.list_storage_class()
    for sc in sc_list.items:
        for provisioner in SUPPORTED_PROVISIONERS:
            if re.search(provisioner, sc.provisioner):
                return sc
    raise SystemExit("Unsupported storageclass from available storageclasses "
                     f"{[sc.metadata.name for sc in sc_list.items]}")


def create_pvc(name=None, storage_class=None, size="100Gi",
               owner_metadata=None):
    context = kubernetes.config.list_kube_config_contexts()[1]
    namespace = context["context"].get("namespace", "default")
    core_api = kubernetes.client.CoreV1Api()
    if storage_class is None:
        storage_class_name = get_storageclass().metadata.name
    else:
        storage_class_name = storage_class.metadata.name
    if owner_metadata is None:
        owner_references = []
    else:
        owner_references = [
            kubernetes.client.V1OwnerReference(
                api_version="adaptdl.petuum.com/v1",
                kind="AdaptDLJob",
                name=owner_metadata["name"],
                uid=owner_metadata["uid"])]
    if name is None:
        metadata = kubernetes.client.V1ObjectMeta(
            namespace=namespace,
            generate_name="adaptdl-pvc-",
            owner_references=owner_references
        )
    else:
        metadata = kubernetes.client.V1ObjectMeta(
            namespace=namespace,
            name=name,
            owner_references=owner_references
        )
    claim = kubernetes.client.V1PersistentVolumeClaim(
        metadata=metadata,
        spec=kubernetes.client.V1PersistentVolumeClaimSpec(
            storage_class_name=storage_class_name,
            access_modes=["ReadWriteMany"],
            volume_mode="Filesystem",
            resources=kubernetes.client.V1ResourceRequirements(
                requests={"storage": size})
        ),
    )
    return core_api.create_namespaced_persistent_volume_claim(
        namespace, claim)


def create_copy_pod(pvc_name, cp_job_uid):
    context = kubernetes.config.list_kube_config_contexts()[1]
    namespace = context["context"].get("namespace", "default")
    core_api = kubernetes.client.CoreV1Api()
    volume_name = "adaptdl-pvc"

    labels = {"adaptdl/cli-copy": cp_job_uid}

    pvc = core_api.read_namespaced_persistent_volume_claim(pvc_name, namespace)
    pvc_uid = pvc.metadata.uid

    metadata = {"namespace": namespace,
                "generateName": "copy-{}".format(pvc_name),
                "labels": labels,
                "owner_references": [
                    kubernetes.client.V1OwnerReference(
                        api_version="v1",
                        kind="PersistentVolumeClaim",
                        name=pvc_name,
                        uid=pvc_uid)]
                }

    container = kubernetes.client.V1Container(
        name="copy-container",
        image="alpine",
        command=["sleep"],
        args=["1000000"],
        volume_mounts=[
            kubernetes.client.V1VolumeMount(
                name=volume_name,
                mount_path="adaptdl_pvc",
            ),
        ],
    )

    body = kubernetes.client.V1Pod(
        metadata=metadata,
        spec=kubernetes.client.V1PodSpec(
            volumes=[{
                "name": volume_name,
                "persistentVolumeClaim": {
                    "claimName": pvc_name
                    }
                }],
            containers=[container]
        )
    )
    return core_api.create_namespaced_pod(namespace, body)
