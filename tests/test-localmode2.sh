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


set -ex

REMOTE=${1?Remote standalone node IP has to be specified}

cd ../


echo "Setting up remote environment"
rm /tmp/adaptdl* || true
ssh -t ubuntu@$REMOTE << EOF
rm -rf adaptdl*
rm -rf examples/
EOF

export ADAPTDL_VERSION=${ADAPTDL_VERSION:-0.0.0} 
python3 setup_adaptdl.py sdist bdist_wheel --dist-dir=/tmp 1>/dev/null
tar cfz /tmp/adaptdl-examples.tar.gz examples 1>/dev/null
scp /tmp/adaptdl-examples.tar.gz ubuntu@$REMOTE:~/ 1>/dev/null
scp examples/requirements.txt ubuntu@$REMOTE:~/ 1>/dev/null
scp /tmp/adaptdl-${ADAPTDL_VERSION}-*.whl ubuntu@$REMOTE:~/ 1>/dev/null
#rm /tmp/adaptdl.tar.gz
#rm /tmp/adaptdl-${ADAPTDL_VERSION}-*.whl

ssh ubuntu@$REMOTE << EOF
pip install -r requirements.txt --no-cache-dir 1>/dev/null
pip install adaptdl-${ADAPTDL_VERSION}-*.whl 1>/dev/null
tar xfz adaptdl-examples.tar.gz 1>/dev/null

echo "Starting job 1"
nohup python3 examples/pytorch-cifar/main.py > job1 &
sleep 30

echo "Starting job 2"
nohup python3 examples/pytorch-cifar/main.py > job2 &
sleep 30

echo "Starting job 3"
nohup python3 examples/pytorch-cifar/main.py > job3 &

# clean up all these processes in 60 seconds
sleep 30
ps -aef | grep pytorch-cifar | grep -v grep | tr -s ' ' | cut -d ' ' -f2 | xargs kill -9

EOF
