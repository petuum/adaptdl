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


set -e

NAMESPACE=${1:?Namespace has to be specified}
TARGET=${2:-2}

kubectl get pods -w --no-headers &

i=0
while true; do
    active=$(kubectl --namespace $NAMESPACE get adaptdljobs -o jsonpath=$'{range .items[*]}{.status.completionTimestamp}\n' | grep -cv '\S')
    if [ $active -lt $TARGET ]; then
       NOF=$(ls ./long-workload/*.sh | wc -l)
       file=$(ls ./long-workload/*.sh | awk -v r="$RANDOM" -v f="$NOF" '{ if ((r%f+1) == NR) print }')
       $file | kubectl --namespace $NAMESPACE create -f -
       echo $file
       i=$[$i+1]
    fi
    sleep 10
done
