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


from multiprocessing import Process
import numpy as np
import collections
from adaptdl.reducer import Reducer
import portpicker
import signal
import faulthandler

root_host = "127.0.0.1"
DEFAULT_REDUCER_PORT = portpicker.pick_unused_port()


def main(rank, size):
    faulthandler.enable(all_threads=True)
    faulthandler.register(signal.SIGUSR1, all_threads=True, chain=False)

    reducer = Reducer(rank, size, root_host, DEFAULT_REDUCER_PORT)

    if rank == 0:
        batch_size = 28
        x = {"foo": 1}
    else:
        x = {"bar": 1}
        batch_size = 0

    # start a async reducer
    ax = reducer.allreduce_async(np.asarray([1, 1, 1]))

    # do a bunch of bcasts
    batch_size = reducer.broadcast(batch_size)
    batch_size = reducer.broadcast(batch_size)
    batch_size = reducer.broadcast(batch_size)
    assert batch_size == 28

    # do allreduce on Counter
    x = reducer.allreduce(collections.Counter(x))
    assert x["foo"] == 1
    assert x["bar"] == size - 1

    # collect the allreduce_async result
    ax = ax.result()
    assert np.allclose(ax, size * np.asarray([1, 1, 1]))

    # try to simulate a training loop
    x = None
    for _ in range(10):
        if x:
            x = x.result()
            assert np.allclose(x, size * np.asarray([1, 1, 1]))
        x = reducer.allreduce_async(np.asarray([1, 1, 1]))


def test_reducer():
    size = 3  # number of replicas
    processes = []
    for rank in range(size):
        p = Process(target=main, args=(rank, size), daemon=True)
        p.start()
        processes.append(p)

    for p in processes[1:]:
        p.join()

    processes[0].join()

    # check exceptions raised by the processes
    for p in processes:
        assert not p.exitcode
