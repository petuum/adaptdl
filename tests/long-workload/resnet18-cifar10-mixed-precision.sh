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


if ! python3 -m pip show adaptdl-cli > /dev/null 2>&1
then python3 -m pip install adaptdl-cli
fi

ROOT=$(dirname $0)/../..
adaptdl tensorboard create tensorboard --nodeport || true
cat << EOF | adaptdl submit --checkpoint-storage-size 1Gi $ROOT --tensorboard tensorboard -d $ROOT/examples/Dockerfile -f -
apiVersion: adaptdl.petuum.com/v1
kind: AdaptDLJob
metadata:
  generateName: mixed-precision-resnet50-cifar10-v2-elastic-
spec:
  template:
    spec:
      containers:
      - name: main
        command:
        - python3
        - /root/examples/pytorch-cifar/main.py
        - --model=ResNet18
        - --bs=256
        - --lr=0.1
        - --epochs=60
        - --autoscale-bsz
        - --mixed-precision
        env:
        - name: PYTHONUNBUFFERED
          value: "true"
        resources:
          limits:
            nvidia.com/gpu: 1
EOF
