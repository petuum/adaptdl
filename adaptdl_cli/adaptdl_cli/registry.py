#!/usr/bin/env python3
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
from pathlib import Path
import sys
import socket
import kubernetes


ADAPTDL_REGISTRY_URL = "adaptdl-registry.remote:32000"


if "linux" in sys.platform.lower():
    _DAEMON_FILE = "/etc/docker/daemon.json"
else:
    _DAEMON_FILE = f"{str(Path.home())}/.docker/daemon.json"
_REG_KEY = "insecure-registries"


# https://forums.docker.com/t/restart-docker-from-command-line/9420/9
MACOS_DOCKER_RESTART_SCRIPT = r"""osascript -e 'quit app "Docker"'; \
                               open -a Docker ; \
                               while ! docker system info > /dev/null 2>&1; do sleep 1; done """  # noqa: E501

LINUX_DOCKER_RESTART_SCRIPT = r"""sudo systemctl restart docker \
                               while ! sudo docker system info > /dev/null 2>&1; do sleep 1; done """  # noqa: E501


def _get_node_ip():
    def ready(conditions):
        return any(cond.type == "Ready" and
                   cond.status == "True" for cond in conditions)

    # Collect Ready nodes sorted by descending age
    v1 = kubernetes.client.CoreV1Api()
    nodes = v1.list_node()
    nodes = [node for node in nodes.items if ready(node.status.conditions)]
    nodes = sorted(nodes, key=lambda x: x.metadata.creation_timestamp)
    if len(nodes) == 0:
        return None

    for addr in nodes[0].status.addresses:
        if addr.type == 'InternalIP':
            ip = addr.address
        if addr.type == 'ExternalIP':
            ip = addr.address
            break
    assert ip
    return ip


def fix_etc_hosts():
    # Correct /etc/hosts entry with the current node IP kubectl is pointing to
    node_ip = _get_node_ip()
    host_entry = None
    if not node_ip:
        raise SystemExit("Didn't find any nodes.")
    try:
        host_entry = socket.gethostbyname(ADAPTDL_REGISTRY_URL.split(':')[0])
    except socket.gaierror:
        pass

    if node_ip != host_entry:
        assert os.system(f"sudo `which hostman` add -f {node_ip} \
                {ADAPTDL_REGISTRY_URL.split(':')[0]}") == 0


def _find_entry():
    if not os.path.exists(_DAEMON_FILE):
        return False, {}
    with open(_DAEMON_FILE, 'r') as f:
        data = json.load(f)
        if _REG_KEY in data and ADAPTDL_REGISTRY_URL in data[_REG_KEY]:
            return True, None
        else:
            return False, data


def _fix_daemon_json():
    found, data = _find_entry()
    if not found:
        with open(_DAEMON_FILE, 'w') as f:
            if _REG_KEY in data:
                data[_REG_KEY].append(ADAPTDL_REGISTRY_URL)
            else:
                data[_REG_KEY] = [ADAPTDL_REGISTRY_URL]
            json.dump(data, f)
            print(f"Wrote {data}")
        return True  # file updated
    return False


def registry_running():
    v1 = kubernetes.client.CoreV1Api()
    svc = v1.list_service_for_all_namespaces(
        field_selector="metadata.name=adaptdl-registry")
    return len(svc.items) > 0


def fix_local_docker():
    if not registry_running():
        raise SystemExit("Registry service not installed or running.")

    if _fix_daemon_json():
        # restart docker daemon
        if "darwin" in sys.platform.lower():
            assert os.system(MACOS_DOCKER_RESTART_SCRIPT) == 0
        elif "linux" in sys.platform.lower():
            assert os.system(LINUX_DOCKER_RESTART_SCRIPT) == 0
        else:
            print("Restart your docker daemon for changes to take effect.")

    assert os.system(f"docker login -u user -p password \
                      {ADAPTDL_REGISTRY_URL}") == 0


if __name__ == "__main__":
    fix_local_docker()
