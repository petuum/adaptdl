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
import logging
import prometheus_client

from adaptdl_sched.controller import AdaptDLController

logging.basicConfig()
kubernetes.config.load_incluster_config()
prometheus_client.start_http_server(9091)

controller = AdaptDLController()

loop = asyncio.get_event_loop()
loop.run_until_complete(
    controller.run(),
)
loop.close()
