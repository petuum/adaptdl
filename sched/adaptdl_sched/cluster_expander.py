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
import kubernetes_asyncio as kubernetes
from kubernetes_asyncio.client.rest import ApiException
import logging
import adaptdl_sched.config as config
from collections import namedtuple
import adaptdl_sched.k8s_templates as templates

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


class ClusterExpander(object):
    """ ClusterExpander tries to keep expected node count available to the
    allocator. It does that by spawning equal number of placeholder pods (one
    of each node). The pods have anti-affinity which prevents them from getting
    scheduled on the same node. This pushes the cluster autoscaler to provision
    one node for each Pending placeholder. The job of ClusterExpander then is
    really to maintain the requested count of placeholders. This also holds
    when the jobs are scaling down or finshing and we no longer need specific
    nodes"""

    def __init__(self):
        self._v1 = kubernetes.client.CoreV1Api()
        self._apps_api = kubernetes.client.AppsV1Api()
        self._allocations = set()
        self._active_nodes = set()

    def _gen_placeholder_pod(self):
        return {
                'apiVersion': 'v1',
                'kind': 'Pod',
                'metadata': {
                    'generateName': 'adaptdl-placeholder-',
                    'labels': {
                        config.ADAPTDL_PH_LABEL: 'true',
                        'petuum.com/nodegroup': 'adaptdl'
                        },
                    'ownerReferences': self._owner_reference
                    },
                'spec': {
                    'affinity': {
                        'podAntiAffinity': {
                            'requiredDuringSchedulingIgnoredDuringExecution': [
                                {
                                    'labelSelector': {
                                        'matchExpressions': [
                                            {
                                                'key': config.ADAPTDL_PH_LABEL,
                                                'operator': 'In',
                                                'values': ['true']
                                            }]
                                        },
                                    'topologyKey': 'kubernetes.io/hostname'
                                    }
                                ]
                            }
                    },
                    'containers': [{
                        'image': 'busybox',
                        'name': 'placeholder',
                        'command': ["/bin/sh", "-ec",
                                    "while :; do echo '.'; sleep 5 ; done"],
                        "resources": {
                            "requests": {
                                "memory": "5Mi",
                                "cpu": "1m"
                            }
                        }
                    }],
                    'restartPolicy': 'Never',
                    }
                }

    async def _reconcile(self, expected):
        Pod = namedtuple('Pod', ['name', 'node', 'phase'])

        def key_fn(x):
            if x.node in self._allocations and x.phase == 'Running':
                return 2
            elif x.node not in self._allocations and x.phase == 'Running':
                return 1
            else:
                return 0

        try:
            ret = await self._v1.list_namespaced_pod(config.get_namespace(),
                                                     label_selector=f"{config.ADAPTDL_PH_LABEL}=true")  # noqa: E501

            pods = [Pod(pod.metadata.name, pod.spec.node_name,
                        pod.status.phase) for pod in ret.items]

            # sort pods based on 1. in allocations 2. Running 3. Pending
            pods = sorted(pods, key=key_fn)

            running = sum(1 if pod.phase == 'Running' else 0
                          for pod in pods)
            spawned = sum(1 if pod.phase in ('Running', 'Pending') else 0
                          for pod in pods)

            succeeded = sum(1 if pod.phase == 'Succeeded' else 0
                            for pod in pods)
            assert succeeded == 0

            LOG.info(f"Received {expected} expected nodes. There are "
                     f"{running} Running, {spawned - running} Pending "
                     f"placeholder pods.")

            if expected == spawned:
                return
            elif expected > spawned:  # spawn the difference
                r = []
                for _ in range(expected - spawned):
                    r.append(self._v1.create_namespaced_pod(
                        body=self._gen_placeholder_pod(),
                        namespace=config.get_namespace()))
                ret = await asyncio.gather(*r)
                for pod in ret:
                    LOG.info(f"Spawned {pod.metadata.name}")
            else:   # expected < spawned, delete from the left
                r = []
                for name in [x.name for x in pods][:spawned - expected]:
                    assert name not in self._allocations
                    r.append(self._v1.delete_namespaced_pod(
                        name=name, namespace=config.get_namespace()))
                await asyncio.gather(*r)
                for name in [x.name for x in pods][:spawned - expected]:
                    LOG.info(f"Deleted {name}")
        except ApiException as e:
            if e.status == 401:  # Unauthorized
                raise  # Fatal
            else:
                LOG.warning(f"{e}")

    async def run(self):
        adaptdl_sched = \
            await self._apps_api.read_namespaced_deployment(
                namespace=config.get_namespace(),
                name=config.get_adaptdl_deployment())
        self._owner_reference = templates.owner_reference_template(
            config.get_namespace(),
            adaptdl_sched.metadata.name,
            adaptdl_sched.metadata.uid,
            "Deployment",
            "apps/v1")
        while True:
            await self._reconcile(len(self._active_nodes))
            await asyncio.sleep(30)

    def fit(self, active_nodes):
        """ active_nodes contain allocations + virtual nodes, our job at the
        expander is to 1. maintain the allocations, and 2. provision nodes in
        active nodes and not in allocations."""
        self._active_nodes = set(active_nodes)
        self._allocations = set()
        for node in self._active_nodes:
            if not node.startswith("~"):  # real nodes only
                self._allocations.add(node)


if __name__ == '__main__':
    # unit test to check basic sanity
    loop = asyncio.get_event_loop()
    loop.run_until_complete(kubernetes.config.load_kube_config())

    expander = ClusterExpander()

    async def run():
        expander.fit(['n0', '~n1'])
        await expander.run()

    loop.run_until_complete(run())
    loop.close()
