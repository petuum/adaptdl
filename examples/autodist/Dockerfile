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


FROM python:3.6.12-buster
WORKDIR /root

FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-runtime

FROM tensorflow/tensorflow:2.2.0-gpu
# Install apps
COPY adaptdl adaptdl
COPY examples/requirements.txt .

RUN cd adaptdl && python3 setup.py bdist_wheel

ARG ADAPTDL_VERSION=0.0.0
RUN ADAPTDL_VERSION=${ADAPTDL_VERSION} pip install adaptdl/dist/*.whl
RUN pip install -r requirements.txt

RUN rm -rf adaptdl/dist

# autodist env
SHELL ["/bin/bash", "-cu"]

RUN rm -rf /etc/bash.bashrc

RUN apt-get update && apt-get install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
        build-essential \
        git \
        curl \
        vim \
        wget \
        unzip
WORKDIR /root
RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

COPY bert_config.json bert_config.json
COPY tf_examples.tfrecord tf_examples.tfrecord
RUN git clone https://github.com/petuum/autodist.git
#RUN cd autodist
#WORKDIR autodist
#RUN git checkout integration
COPY autodist autodist
RUN wget https://github.com/protocolbuffers/protobuf/releases/download/v3.11.0/protoc-3.11.0-linux-x86_64.zip
RUN unzip protoc-3.11.0-linux-x86_64.zip
RUN PROTOC=$(pwd)/bin/protoc python setup.py build
RUN pip install -e .[dev]
RUN pip install tensorflow_hub
WORKDIR autodist
#RUN rm ./examples/resource_spec.yml

# setup ssh
# Install OpenSSH to communicate between containers
RUN apt-get install -y --no-install-recommends openssh-client openssh-server && \
    mkdir -p /var/run/sshd

WORKDIR /root
RUN mkdir /root/.ssh
RUN ssh-keygen -f /root/.ssh/id_rsa && cat /root/.ssh/id_rsa.pub | cat >> /root/.ssh/authorized_keys
RUN chown -R root /root/.ssh
RUN chmod 700 /root/.ssh
RUN chmod 600 /root/.ssh/authorized_keys

RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
RUN sed -i 's/#PubkeyAuthentication yes/PubkeyAuthentication yes/' /etc/ssh/sshd_config

# Allow OpenSSH to talk to containers without asking for confirmation
RUN cat /etc/ssh/ssh_config | grep -v StrictHostKeyChecking > /etc/ssh/ssh_config.new && \
    echo "    StrictHostKeyChecking no" >> /etc/ssh/ssh_config.new && \
    mv /etc/ssh/ssh_config.new /etc/ssh/ssh_config

ENV PYTHONUNBUFFERED=true
