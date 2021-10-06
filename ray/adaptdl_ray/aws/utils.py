# Copyright 2021 Petuum, Inc. All Rights Reserved.
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


from enum import Enum
import os


# Adapted from Ray Tune
def _checkpoint_obj_to_dir(checkpoint_dir, checkpoint_obj):
    for (path, data) in checkpoint_obj.items():
        file_path = os.path.join(checkpoint_dir, path)
        directory = os.path.dirname(file_path)
        os.makedirs(directory, exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(data)
    return


def _serialize_checkpoint(checkpoint_dir):
    data = {}
    for basedir, _, file_names in os.walk(checkpoint_dir):
        for file_name in file_names:
            path = os.path.join(basedir, file_name)
            with open(path, "rb") as f:
                data[os.path.relpath(path, checkpoint_dir)] = f.read()
    return data


class Status(Enum):
    FAILED = 0
    SUCCEEDED = 1
    RUNNING = 2
