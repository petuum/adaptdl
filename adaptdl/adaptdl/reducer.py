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


import logging
import pickle
import socket
import threading
import time
import traceback
import sys


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Future(object):
    def __init__(self, reducer, key):
        self._reducer = reducer
        self._key = key

    def result(self):
        try:
            return self._result
        except AttributeError:
            while self._key not in self._reducer._result_map:
                try:
                    key, result = pickle.load(self._reducer._sockfile)
                    self._reducer._result_map[key] = result
                except Exception as e:
                    logger.error(f"reducer._rank = {self._reducer._rank}"
                                 f" is exiting unexpectedly because of {e}")
                    raise
            self._result = self._reducer._result_map.pop(self._key)
            return self._result


def default_reduce_fn(a, b):
    a += b
    return a


class Reducer(object):
    """
    Simple asynchronous (all)reduce operations on python objects. Assumes all
    invokations to allreduce, allreduce_async, and Future.result happen in the
    same order across all processes.
    """

    def __init__(self, rank, replicas, root_host, root_port):
        self._root_port = root_port
        self._result_map = {}
        self._next_key = 0
        self._rank = rank

        if rank == 0:
            self._reduce_fn_map = {}
            threading.Thread(target=self._run_server,
                             args=(self._root_port, replicas),
                             daemon=True).start()
        # Keep retrying connection, because (1) the root pod might not have
        # a registered domain name yet, and (2) the root server socket might
        # not be bound yet.
        exception_cnt = 0
        while True:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            if (exception_cnt > 25):
                logger.error("Could not connect to root after max "
                             "retries, exiting...")
                break
            try:
                if (self._root_port == 0):
                    # waiting for server to get a valid port in local mode
                    raise ConnectionRefusedError
                logger.info(f"rank {rank} connecting to {root_host} "
                            f"on port {self._root_port}")
                sock.connect((root_host, self._root_port))
            except ConnectionRefusedError:
                logger.warning("Could not connect to root, trying again...")
                exception_cnt += 1
                time.sleep(5)
            else:
                break
        self._sockfile = sock.makefile("rwb")
        pickle.dump(rank, self._sockfile)
        self._sockfile.flush()

    def broadcast(self, obj):
        """
        Broadcast a value from replica 0 to all other replicas. Currently uses
        allreduce with left-projection.
        """
        return self.allreduce(obj, lambda x, y: x)

    def allreduce(self, obj, reduce_fn=default_reduce_fn):
        future = self.allreduce_async(obj, reduce_fn)
        return future.result()

    def allreduce_async(self, obj, reduce_fn=default_reduce_fn):
        key = self._next_key
        self._next_key += 1
        try:
            self._reduce_fn_map[key] = reduce_fn
        except AttributeError:
            pass
        pickle.dump(obj, self._sockfile)
        # TODO - flush throws an unhandled exception
        self._sockfile.flush()
        return Future(self, key)

    def _run_server(self, port, replicas):
        try:
            listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            listener.bind(("0.0.0.0", port))
            if port == 0:
                # local mode
                self._root_port = listener.getsockname()[1]
            listener.listen(replicas)
            # wait for connections from all clients
            logger.info(f"Master waiting for connections on {port}")
            clients = [None] * replicas
            while None in clients:
                client = listener.accept()[0].makefile("rwb")
                rank = pickle.load(client)
                assert clients[rank] is None
                clients[rank] = client
            # main server loop
            key = 0
            while True:
                for rank, client in enumerate(clients):
                    obj = pickle.load(client)
                    if rank == 0:
                        result = obj
                        reduce_fn = self._reduce_fn_map.pop(key)
                    else:
                        result = reduce_fn(result, obj)
                # Respond to clients in reverse order, with rank 0 last.
                # Prevents deadlocks where the rank 0 client gets unblocked
                # first and grabs the GIL in a later operation, blocking this
                # server from responding to the remaining replicas.
                for client in reversed(clients):
                    pickle.dump((key, result), client)
                    client.flush()
                key += 1
        except Exception:
            traceback.print_exception(*sys.exc_info())
            exit(1)
