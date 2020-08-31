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
import subprocess
from pathlib import Path
import sys

ADAPTDL_REGISTRY_URL = "adaptdl-registry.default.svc.cluster.local:31001"
ADAPTDL_REGISTRY_CREDS = "adaptdl-registry-creds"

_DAEMON_FILE = f"{str(Path.home())}/.docker/daemon.json"
_REG_KEY = "insecure-registries"

# https://forums.docker.com/t/restart-docker-from-command-line/9420/9
MACOS_DOCKER_RESTART_SCRIPT = r"""osascript -e 'quit app "Docker"'; \
                               open -a Docker ; \
                               while ! docker system info > /dev/null 2>&1; do sleep 1; done """  # noqa: E501

LINUX_DOCKER_RESTART_SCRIPT = r"""sudo systemctl restart docker \
                               while ! sudo docker system info > /dev/null 2>&1; do sleep 1; done """  # noqa: E501


def _get_node_ip():
    nodes = json.loads(subprocess.check_output(['kubectl', 'get', 'no',
                                                '-ojson']))
    ip = None
    if len(nodes['items']) > 0:
        for addr in nodes['items'][0]['status']['addresses']:
            if addr['type'] == 'InternalIP':
                ip = addr['address']
            if addr['type'] == 'ExternalIP':
                ip = addr['address']
                break
    return ip


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
    svc = json.loads(subprocess.check_output(['kubectl', 'get', 'svc', '-l',
                                             'app=docker-registry', '-ojson']))
    return len(svc['items']) > 0


def fix_local_docker():
    node_ip = _get_node_ip()
    if not node_ip:
        raise SystemExit("Didn't find any nodes.")
    if not registry_running():
        raise SystemExit("Registry service not installed or running.")

    if _fix_daemon_json():
        # restart docker daemon
        if "darwin" in sys.platform.lower():
            os.system(MACOS_DOCKER_RESTART_SCRIPT) == 0
        elif "linux" in sys.platform.lower():
            os.system(LINUX_DOCKER_RESTART_SCRIPT) == 0
        else:
            print("Restart your docker daemon for changes to take effect.")

    assert os.system(f"sudo `which hostman` add -f {node_ip} \
                      {ADAPTDL_REGISTRY_URL.split(':')[0]}") == 0
    assert os.system(f"docker login -u user -p password \
                      {ADAPTDL_REGISTRY_URL}") == 0


if __name__ == "__main__":
    fix_local_docker()
