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

import pathlib
import pkg_resources
import sys


if __name__ == "__main__":
    errors = []
    path = pathlib.Path(__file__).with_name("requirements.txt")
    with open(path) as f:
        # https://stackoverflow.com/a/45474387
        for requirement in pkg_resources.parse_requirements(f):
            try:
                pkg_resources.require(str(requirement))
            except pkg_resources.DistributionNotFound as exc:
                errors.append(str(exc))
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        raise SystemExit("To install requirements, run the command:\n"
                         f"{sys.executable} -m pip install -r {path}")
