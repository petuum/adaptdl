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
This module contains simple collective communications primitives which operate
on arbitrary python objects. It is meant to be general but *non-performant*.
Only use these primitives if you are synchronizing small objects which can be
efficiently pickled and operated on. For larger objects, use framework-specific
functions, such as those provided by `torch.distributed`.

The functions in this module should be invoked *in the same order* across all
replicas in the current job. Otherwise, their behavior is undefined and you may
encounter unexpected bugs and errors.
"""

# TODO: Merge the reducer into this module once the previous trainer APIs
# are removed.

import adaptdl.env
from .reducer import Reducer, default_reduce_fn

_REDUCER = None


def initialize(master_addr=None, master_port=None):
    """
    Initialize this module, must be invoked before calling any other functions.
    This function will block until it has been invoked from all replicas.

    Arguments:
        master_addr: address of the replica with rank 0.
        master_port: free port of the replica with rank 0.

    Raises:
        RuntimeError: If this module had already been initialized.
    """
    global _REDUCER
    if _REDUCER is not None:
        raise RuntimeError("{} is already initialized".format(__name__))
    if master_addr is None:
        master_addr = adaptdl.env.master_addr()
    if master_port is None:
        master_port = adaptdl.env.master_port()
    _REDUCER = Reducer(adaptdl.env.replica_rank(),
                       adaptdl.env.num_replicas(),
                       master_addr, master_port)


def teardown():
    """
    Teardown this module, will block until this function has been invoked from
    all replicas.

    Raises:
        RuntimeError: If this module has not been initialized.
    """
    raise NotImplementedError  # TODO


def allreduce(value, reduce_fn=default_reduce_fn):
    """
    Reduces a value across all replicas in such a way that they all get the
    final result. Blocks until this function is invoked by all replicas.

    Arguments:
        value (object): The object which will be reduced together with all
            other replicas.
        reduce_fn (Function): A reduction function which two objects as
            arguments, and returns the resulting reduced object.

    Returns:
        object: Resulting value after being reduced across all replicas.

    Raises:
        RuntimeError: If this module has not been initialized.
    """
    if _REDUCER is None:
        raise RuntimeError("{} has not been initialized".format(__name__))
    return _REDUCER.allreduce(value, reduce_fn)


def allreduce_async(value, reduce_fn=default_reduce_fn):
    """
    Asynchronous version of the `allreduce` function. Does not block, instead
    returns a future which can be used to obtain the result later.

    Arguments:
        value (object): The object which will be reduced together with all
            other replicas.
        reduce_fn (Function): A reduction function which two objects as
            arguments, and returns the resulting reduced object.

    Returns:
        Future: Object from which the result can be obtained later.

    Raises:
        RuntimeError: If this module has not been initialized.
    """
    if _REDUCER is None:
        raise RuntimeError("{} has not been initialized".format(__name__))
    return _REDUCER.allreduce_async(value, reduce_fn)


def broadcast(value):
    """
    Broadcasts a value from the replica of rank 0 to all replicas. Blocks until
    this function is invoked by all replicas.

    Arguments:
        value (object): The object which will be broadcasted from replica 0.
            Ignored on all other replicas.

    Returns:
        object: The value broadcasted from replica 0.

    Raises:
        RuntimeError: If this module has not been initialized.
    """
    if _REDUCER is None:
        raise RuntimeError("{} has not been initialized".format(__name__))
    return _REDUCER.broadcast(value)
