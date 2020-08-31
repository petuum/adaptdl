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


FROM python:3.7-slim
WORKDIR /root

FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-runtime
WORKDIR /root

# Install apps
COPY adaptdl adaptdl
COPY examples/requirements.txt .

RUN cd adaptdl && python3 setup.py bdist_wheel

ARG ADAPTDL_VERSION=0.0.0
RUN ADAPTDL_VERSION=${ADAPTDL_VERSION} pip install adaptdl/dist/*.whl
RUN pip install -r requirements.txt

RUN rm -rf adaptdl/dist

COPY tutorial tutorial

ENV PYTHONUNBUFFERED=true
