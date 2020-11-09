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
WORKDIR /root

# Set default shell to /bin/bash
# SHELL ["/bin/bash", "-cu"]

# RUN rm -rf /etc/bash.bashrc

# Install apps
COPY adaptdl adaptdl
COPY examples/requirements.txt .

RUN cd adaptdl && python3 setup.py bdist_wheel

ARG ADAPTDL_VERSION=0.0.0
RUN ADAPTDL_VERSION=${ADAPTDL_VERSION} pip install adaptdl/dist/*.whl
RUN pip install -r requirements.txt

RUN rm -rf adaptdl/dist

#COPY examples examples
#RUN apt-get update && apt-get install -y --no-install-recommends apt-utils
RUN apt-get update && apt-get --assume-yes install systemd && apt-get install sudo

RUN curl https://get.docker.com | sh \
#    && sudo systemctl start docker\
    && sudo service docker start \ 
#    && sudo systemctl enable docker
    && sudo update-rc.d docker enable

RUN distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

RUN sudo apt-get update && sudo apt-get install -y nvidia-docker2
#RUN sudo systemclt restart docker
RUN sudo service docker restart
RUN git clone https://github.com/petuum/autodist.git &&\
    cd autodist && sudo docker build -t autodist:latest -f docker/Dockerfile.gpu .

# install nvidia docker


#COPY autodist/examples examples

# autodist env
#RUN pip install tensorflow-gpu==2.2.0
#COPY autodist autodist
#RUN cd autodist
#RUN pip install wget
#RUN wget https://github.com/protocolbuffers/protobuf/releases/download/v3.11.0/protoc-3.11.0-linux-x86_64.zip
#COPY autodist/protoc-3.11.0-linux-x86_64.zip protoc-3.11.0-linux-x86_64.zip
#RUN unzip protoc-3.11.0-linux-x86_64.zip
#RUN PROTOC=autodist/bin/protoc python autodist/setup.py build
#WORKDIR autodist
#RUN pip install -e .[dev]

ENV PYTHONUNBUFFERED=true
