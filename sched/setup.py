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


import os
import setuptools


def read_requirements(file_path):
    requirements = []
    with open(file_path) as f:
        for line in f:
            if "#" in line:
                line = line[:line.index("#")]
            line = line.strip()
            if line and not line.startswith("-"):
                requirements.append(line)
    return requirements


if __name__ == "__main__":
    setuptools.setup(
        name="adaptdl-sched",
        version=os.getenv("ADAPTDL_VERSION", "0.0.0"),
        author="Petuum Inc. & The AdaptDL Authors",
        author_email="aurick.qiao@petuum.com",
        description="Dynamic-resource trainer and scheduler for deep learning",
        url="https://github.com/petuum/adaptdl",
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: Other/Proprietary License",
            "Operating System :: POSIX :: Linux",
        ],
        packages=setuptools.find_packages(include=["adaptdl_sched",
                                                   "adaptdl_sched.*"]),
        python_requires='>=3.6',
        install_requires=read_requirements("requirements.txt")
    )
