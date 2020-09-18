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

import time
import kubernetes
from .pvc import get_storageclass
from .proxy import service_proxy


TENSORBOARD_PREFIX = "adaptdl-tensorboard-"


def tensorboard_create(args, remaining):
    context = kubernetes.config.list_kube_config_contexts()[1]
    namespace = context["context"].get("namespace", "default")
    name = "{}{}".format(TENSORBOARD_PREFIX, args.name)
    labels = {"adaptdl/tensorboard": args.name}
    storageclass = get_storageclass(name=args.storageclass)
    # Create Deployment.
    container = kubernetes.client.V1Container(
        name="tensorboard",
        image="tensorflow/tensorflow",
        command=["tensorboard"],
        args=["--host=0.0.0.0", "--logdir=/adaptdl/tensorboard"],
        ports=[kubernetes.client.V1ContainerPort(container_port=6006)],
        volume_mounts=[
            kubernetes.client.V1VolumeMount(
                name="logdir",
                mount_path="/adaptdl/tensorboard",
            ),
        ],
    )
    template = kubernetes.client.V1PodTemplateSpec(
        metadata=kubernetes.client.V1ObjectMeta(labels=labels),
        spec=kubernetes.client.V1PodSpec(
            volumes=[
                kubernetes.client.V1Volume(
                    name="logdir",
                    persistent_volume_claim=kubernetes.
                    client.V1PersistentVolumeClaimVolumeSource(
                       claim_name=name),
                )
            ],
            containers=[container],
        )
    )
    deployment = kubernetes.client.V1Deployment(
        metadata=kubernetes.client.V1ObjectMeta(
            namespace=namespace,
            name=name,
            labels=labels,
        ),
        spec=kubernetes.client.V1DeploymentSpec(
            selector=kubernetes.client.V1LabelSelector(match_labels=labels),
            template=template,
        ),
    )
    apps_api = kubernetes.client.AppsV1Api()
    deployment = apps_api.create_namespaced_deployment(namespace, deployment)
    # Construct OwnerReference.
    owner_reference = kubernetes.client.V1OwnerReference(
        api_version="v1",
        kind="Deployment",
        name=deployment.metadata.name,
        uid=deployment.metadata.uid,
    )
    # Create PersistentVolumeClaim.
    claim = kubernetes.client.V1PersistentVolumeClaim(
        metadata=kubernetes.client.V1ObjectMeta(
            name=name,
            labels=labels,
            owner_references=[owner_reference],
        ),
        spec=kubernetes.client.V1PersistentVolumeClaimSpec(
            storage_class_name=storageclass.metadata.name,
            access_modes=["ReadWriteMany"],
            volume_mode="Filesystem",
        ),
    )
    if args.size is not None:
        claim.spec.resources = kubernetes.client.V1ResourceRequirements(
            requests={"storage": args.size})
    core_api = kubernetes.client.CoreV1Api()
    core_api.create_namespaced_persistent_volume_claim(namespace, claim)
    # Create Service.
    service = kubernetes.client.V1Service(
        metadata=kubernetes.client.V1ObjectMeta(
            name=name,
            labels=labels,
            owner_references=[owner_reference],
        ),
        spec=kubernetes.client.V1ServiceSpec(
            selector=labels,
            type=("NodePort" if args.nodeport else "ClusterIP"),
            ports=[
                kubernetes.client.V1ServicePort(
                    port=6006,
                    target_port=6006,
                ),
            ],
        )
    )
    core_api.create_namespaced_service(namespace, service)
    print("Successfully created TensorBoard instance {}".format(args.name))


def tensorboard_delete(args, remaining):
    context = kubernetes.config.list_kube_config_contexts()[1]
    namespace = context["context"].get("namespace", "default")
    name = "{}{}".format(TENSORBOARD_PREFIX, args.name)
    apps_api = kubernetes.client.AppsV1Api()
    apps_api.delete_namespaced_deployment(name, namespace)
    print("Successfully deleted TensorBoard instance {}".format(args.name))


def tensorboard_list(args, remaining):
    context = kubernetes.config.list_kube_config_contexts()[1]
    namespace = context["context"].get("namespace", "default")
    apps_api = kubernetes.client.AppsV1Api()
    core_api = kubernetes.client.CoreV1Api()
    deployment_list = apps_api.list_namespaced_deployment(
        namespace, label_selector="adaptdl/tensorboard")
    pvc_list = core_api.list_namespaced_persistent_volume_claim(
        namespace, label_selector="adaptdl/tensorboard")
    svc_list = core_api.list_namespaced_service(
        namespace, label_selector="adaptdl/tensorboard")
    records = []
    for deployment in deployment_list.items:
        pvc = next((pvc for pvc in pvc_list.items
                    if pvc.metadata.name == deployment.metadata.name), None)
        svc = next((svc for svc in svc_list.items
                    if svc.metadata.name == deployment.metadata.name), None)
        records.append({
            "NAME": deployment.metadata.labels["adaptdl/tensorboard"],
            "NODEPORT": svc and svc.spec.ports[0].node_port,
            "SIZE": pvc and pvc.spec.resources.requests.get("storage"),
            "READY": "{}/{}".format(deployment.status.ready_replicas or 0,
                                    deployment.status.replicas or 0),
        })
    # Format and print table.
    header = ["NAME", "NODEPORT", "SIZE", "READY"]
    widths = [len(key) + 3 for key in header]
    for rec in records:
        for idx, key in enumerate(header):
            rec[key] = "<none>" if rec[key] is None else str(rec[key])
            widths[idx] = max(widths[idx], len(rec[key]) + 3)
    line = "".join("{:<" + str(w) + "}" for w in widths)
    print(line.format(*header))
    for rec in records:
        print(line.format(*[rec[key] for key in header]))


def tensorboard_proxy(args, remaining):
    context = kubernetes.config.list_kube_config_contexts()[1]
    namespace = context["context"].get("namespace", "default")
    service = f"adaptdl-tensorboard-{args.name}"
    with service_proxy(namespace, service, args.address, args.port) as addr:
        print(f"Proxying to TensorBoard instance {args.name} at http://{addr}")
        try:
            while True:
                time.sleep(1000000)
        except KeyboardInterrupt:
            pass


def add_tensorboard_commands(parser):
    parser.set_defaults(handler=lambda args, remaining: parser.print_help())
    subparsers = parser.add_subparsers(help="sub-command help")
    # create
    parser_create = subparsers.add_parser(
        "create", help="create tensorboard deployment")
    parser_create.add_argument(
        "name", type=str, help="tensorboard deployment name")
    parser_create.add_argument("--nodeport", action="store_true",
                               help="also create NodePort service")
    parser_create.add_argument(
        "--storageclass", type=str, help="name of StorageClass")
    parser_create.add_argument(
        "--size", type=str, default="1Gi", help="storage size")
    parser_create.set_defaults(handler=tensorboard_create)
    # delete
    parser_delete = subparsers.add_parser(
        "delete", help="delete tensorboard deployment")
    parser_delete.add_argument(
        "name", type=str, help="tensorboard deployment name")
    parser_delete.set_defaults(handler=tensorboard_delete)
    # list
    parser_list = subparsers.add_parser(
        "list", help="list tensorboard deployments")
    parser_list.set_defaults(handler=tensorboard_list)
    # proxy
    parser_proxy = subparsers.add_parser(
       "proxy", help="proxy to a tensorboard deployment")
    parser_proxy.add_argument(
       "name", type=str, help="tensorboard deployment name")
    parser_proxy.add_argument("--address", type=str, default="127.0.0.1",
                              help="local address to bind for the proxy")
    parser_proxy.add_argument("-p", "--port", type=int,
                              help="local port to bind for the proxy")
    parser_proxy.set_defaults(handler=tensorboard_proxy)
