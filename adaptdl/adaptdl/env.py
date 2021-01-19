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

"""
This module contains functions for retrieving the values of AdaptDL
environment variables, or their defaults if unset.
"""

import os


def checkpoint_path():
    """
    Path to the directory used for saving and loading checkpoints. Determined
    by the environment variable ``ADAPTDL_CHECKPOINT_PATH``, or ``None`` if
    unset. Setting this environment variable is required for checkpointing, and
    is automatically set in AdaptDL-scheduled clusters.

    Returns:
        str: checkpoint path or ``None``.
    """
    return os.getenv("ADAPTDL_CHECKPOINT_PATH")


def share_path():
    """
    Path to a directory shared by all AdaptDL job replicas, which can be used
    by the application, e.g. for storing downloaded datasets or artifacts.
    Determined by the environment variable ``ADAPTDL_SHARE_PATH``, or ``None``
    if unset. Automatically set in AdaptDL-scheduled clusters.

    Returns:
        str: shared directory path or ``None``.
    """
    return os.getenv("ADAPTDL_SHARE_PATH")


def job_id():
    """
    A string which uniquely identifies the current job in an AdaptDL-scheduled
    cluster. ``None`` if running standalone.

    Returns:
        str: unique job identifier or ``None``.
    """
    return os.getenv("ADAPTDL_JOB_ID")


def master_addr():
    """
    Network address of the rank 0 replica, required for distributed training.
    Determined by the environment variable ``ADAPTDL_MASTER_ADDR``, or
    `0.0.0.0` if unset.

    In AdaptDL-scheduled clusters, this environment variable is unset. The rank
    0 replica is discovered dynamically by querying the supervisor
    (:func:`supervisor_url`).


    Returns:
        str: address of the rank 0 replica, or `0.0.0.0`.
    """
    return os.getenv("ADAPTDL_MASTER_ADDR", "0.0.0.0")


def master_port():
    """
    Available port for the rank 0 replica, required for distributed training.
    Determined by the environment variable ``ADAPTDL_MASTER_PORT``, or 0 if
    unset. Automatically set in AdaptDL-scheduled clusters.

    Returns:
        int: available port for the rank 0 replica, or 0.
    """
    return int(os.getenv("ADAPTDL_MASTER_PORT", "0"))


def replica_rank():
    """
    Rank of the current replica, required for distributed training. Each
    replica is assigned a unique rank from 0 to K-1, where K is the total
    number of replicas. Determined by the environment variable
    ``ADAPTDL_REPLICA_RANK``, or 0 if unset. Automatically set in
    AdaptDL-scheduled clusters.

    Returns:
        int: rank of the current replica, or 0.
    """
    return int(os.getenv("ADAPTDL_REPLICA_RANK", "0"))


def num_nodes():
    """
    Number of unique nodes being used for the current job. For example, if
    there are 4 nodes, each running 2 replicas, then this function returns 4.
    Determined by the environment variable ``ADAPTDL_NUM_NODES``, or is equal
    to :func:`num_replicas` if unset. Thus, this environment variable only
    needs to be set if some node runs multiple replicas. Automatically set in
    AdaptDL-scheduled clusters.

    Returns:
        int: number of unique nodes, or the value of :func:`num_replicas`.
    """
    return int(os.getenv("ADAPTDL_NUM_NODES", num_replicas()))


def num_replicas():
    """
    Total number of replicas, required for distributed training. For example,
    if there are 4 nodes, each running 2 replicas, then this function returns
    8. Determined by the environment variable ``ADAPTDL_NUM_REPLICAS``, or 1 if
    unset. Automatically set in AdaptDL-scheduled clusters.

    Returns:
        int: total number of replicas, or 1.
    """
    return int(os.getenv("ADAPTDL_NUM_REPLICAS", "1"))


def num_restarts():
    """
    Number of times the current job was restarted. Determined by the
    environment variable ``ADAPTDL_NUM_RESTARTS``, or 0 if unset. This value is
    mainly informational, and is automatically set in AdaptDL-scheduled
    clusters.

    Returns:
        int: number of restarts, or 0.
    """
    return int(os.getenv("ADAPTDL_NUM_RESTARTS", "0"))


def adaptdl_sched_version():
    """
    A string which gives the AdaptDL version of scheduler. Determined
    by the environment variable ``ADAPTDL_SCHED_VERSION`` or ``None``

    Returns:
        str: AdaptDL version of scheduler, or ``None``.
    """
    return os.environ.get("ADAPTDL_SCHED_VERSION")


def supervisor_url():
    """
    URL of the supervisor in an AdaptDL-scheduled cluster. The address of the
    rank 0 replica is dynamically discovered via the supervisor, instead of via
    the ``ADAPTDL_MASTER_ADDR`` environment variable.

    Returns:
        str: URL of the supervisor, or ``None``.
    """
    return os.getenv("ADAPTDL_SUPERVISOR_URL")
