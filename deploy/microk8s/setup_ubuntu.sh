#!/usr/bin/env bash
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


sudo snap install microk8s --classic --channel=1.18/stable

sudo microk8s enable dns  # Init dns first.
sudo microk8s enable gpu helm storage dashboard

sudo microk8s helm init

sudo usermod -a -G microk8s $USER
sudo chown -f -R $USER ~/.kube

sudo microk8s status --wait-ready
sudo microk8s kubectl config view --raw > $HOME/.kube/config
