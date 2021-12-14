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


import os
import time
import adaptdl._signal as signal


def foo(x, y):
    return x + y


if __name__ == "__main__":
    x = foo(3, 2)
    if os.environ.get("ADAPTDL_REPLICA_RANK", None) == "0":
        with open(
                f"{os.environ['ADAPTDL_CHECKPOINT_PATH']}/file.txt", "w") as f:
            f.write(f"{x}\n")
        for _ in range(30):
            time.sleep(1)
            if signal.get_exit_flag():
                exit(143)

else:
    raise RuntimeError("Not running as main")
