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


def owner_reference_template(namespace, name, uid, kind="AdaptDLContext",
                             api="adaptdl.petuum.com/v1"):
    return [{"apiVersion": api,
             "controller": True,
             "blockOwnerDeletion": True,
             "kind": kind,
             "name": name,
             "uid": uid}]
