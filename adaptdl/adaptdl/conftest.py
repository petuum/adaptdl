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


import functools
import multiprocessing as mp
import os
import signal
import tempfile

import portpicker


def elastic_multiprocessing(func):
    """
    Decorator which runs a function inside a temporary local environment
    which mimics a real AdaptDLJob. Runs replicas of the decorated function
    in their own processes, and sets up the shared environment, including
    environment variables and shared directories. The decorated function is
    always started with a single replica, but can optionally return an integer
    number of replicas to trigger a restart using that many replicas.

    ```python
    @elastic_multiprocessing
    def test_my_stuff():
        from adaptdl.env import num_replicas, num_restarts
        if num_restarts() == 0:
            print(num_replicas)  # Outputs '1'.
            return 5  # Restart using 5 replicas.
        if num_restarts() == 1:
            print(num_replicas)  # Outputs '5'.
        return 0  # No more restarts, this line can be omitted.
    ```

    .. warning::
       The replica processes are forked from the current main process. This
       means that mutations to global variables in the main process prior to
       calling the decorated function may be observed by the child processes!
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        num_restarts = 0
        num_replicas = 1
        with tempfile.TemporaryDirectory() as tmpdir:
            while num_replicas:
                assert isinstance(num_replicas, int)
                master_port = portpicker.pick_unused_port()
                queue = mp.Queue()  # For passing return values back.

                def run(rank):  # Invoked in each child process.
                    os.environ["ADAPTDL_CHECKPOINT_PATH"] = str(tmpdir)
                    os.environ["ADAPTDL_JOB_ID"] = "tmpjob"
                    os.environ["ADAPTDL_MASTER_PORT"] = str(master_port)
                    os.environ["ADAPTDL_REPLICA_RANK"] = str(rank)
                    os.environ["ADAPTDL_NUM_REPLICAS"] = str(num_replicas)
                    os.environ["ADAPTDL_NUM_NODES"] = "1"
                    os.environ["ADAPTDL_NUM_RESTARTS"] = str(num_restarts)
                    ret = None
                    try:
                        ret = func(*args, **kwargs)
                    finally:
                        queue.put((rank, ret))

                # Start each replica in a separate child process.
                procs = [mp.Process(target=run, args=(rank,))
                         for rank in range(num_replicas)]
                for proc in procs:
                    proc.start()

                try:  # Wait for results from child processes.
                    for i in range(num_replicas):
                        rank, ret = queue.get()
                        procs[rank].join()
                        assert procs[rank].exitcode == 0
                        if i == 0:  # All return values should be the same.
                            num_replicas = ret
                        assert num_replicas == ret
                finally:
                    # Clean up any remaining child processes.
                    for proc in procs:
                        if proc.is_alive():
                            os.kill(proc.pid, signal.SIGKILL)
                        proc.join()
                    # Clean up the queue.
                    queue.close()
                num_restarts += 1

    return wrapper
