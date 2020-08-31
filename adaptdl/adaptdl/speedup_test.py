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


from adaptdl.speedup import SpeedupFunction
import numpy as np


def test_no_bounds():
    np.random.seed(0)
    for i in range(100):
        params = np.random.gamma(2.0, 2.0, (7,))
        grad_params = np.random.gamma(2.0, 2.0, (2,))
        grad_params = {"norm": grad_params[0], "var": grad_params[1]}
        fun = SpeedupFunction(params, grad_params, 128, None, None, False)
        speedup, bsz = fun(1, 3, return_local_bsz=True)
        assert(bsz == 128//3 + 1), "expected bsz = 43, got {}".format(bsz)
        assert(isinstance(speedup, float))

        replicas = np.asarray([1, 2, 3, 4, 5])
        # single-node
        speedup, bsz = fun(np.ones_like(replicas), replicas,
                           return_local_bsz=True)
        assert(bsz.shape == (5,))
        assert(np.all(bsz == np.ceil(128 / replicas).astype(int)))
        assert(speedup.shape == (5,))
        assert(bsz[0] == 128)
        # multi-node
        speedup, bsz = fun(replicas, replicas, return_local_bsz=True)
        assert(bsz.shape == (5,))
        assert(np.all(bsz == np.ceil(128 / replicas).astype(int)))
        assert(speedup.shape == (5,))
        assert(bsz[0] == 128)


def test_local_bounds():
    np.random.seed(0)
    for i in range(100):
        params = np.random.gamma(2.0, 2.0, (7,))
        grad_params = np.random.gamma(2.0, 2.0, (2,))
        grad_params = {"norm": grad_params[0], "var": grad_params[1]}
        fun = SpeedupFunction(params, grad_params, 128, None, (64, 256), True)
        speedup, bsz = fun(1, 1, return_local_bsz=True)
        assert(bsz == 128), "expected bsz = 128, got {}".format(bsz)
        assert(isinstance(speedup, float))

        replicas = np.asarray(range(1, 100))
        # single-node
        speedup, bsz = fun(np.ones_like(replicas), replicas,
                           return_local_bsz=True)
        assert(np.all(bsz >= np.ceil(128 / replicas).astype(int)))
        assert(np.all(np.logical_or(bsz >= (64), speedup == 0.0)))
        assert(np.all(bsz <= (256)))
        assert(np.all(bsz * replicas <= 100 * 128))
        assert(bsz[0] == 128)
        # multi-node
        speedup, bsz = fun(replicas, replicas, return_local_bsz=True)
        assert(np.all(bsz >= np.ceil(128 / replicas).astype(int)))
        assert(np.all(np.logical_or(bsz >= (64), speedup == 0.0)))
        assert(np.all(bsz <= (256)))
        assert(np.all(bsz * replicas <= 100 * 128))
        assert(bsz[0] == 128)


def test_max_bounds():
    np.random.seed(0)
    for i in range(100):
        params = np.random.gamma(2.0, 2.0, (7,))
        grad_params = np.random.gamma(2.0, 2.0, (2,))
        grad_params = {"norm": grad_params[0], "var": grad_params[1]}
        fun = SpeedupFunction(params, grad_params, 128, 1280, None, True)
        speedup, bsz = fun(1, 1, return_local_bsz=True)
        assert(bsz == 128), "expected bsz = 128, got {}".format(bsz)
        assert(isinstance(speedup, float))

        replicas = np.asarray(range(1, 100))
        # single-node
        speedup, bsz = fun(np.ones_like(replicas), replicas,
                           return_local_bsz=True)
        assert(np.all(bsz >= np.ceil(128 / replicas).astype(int)))
        assert(np.all(bsz * replicas <= 1280))
        assert(bsz[0] == 128)
        # multi-node
        speedup, bsz = fun(replicas, replicas, return_local_bsz=True)
        assert(np.all(bsz >= np.ceil(128 / replicas).astype(int)))
        assert(np.all(bsz * replicas <= 1280))
        assert(bsz[0] == 128)


def test_all_bounds():
    np.random.seed(0)
    for i in range(100):
        params = np.random.gamma(2.0, 2.0, (7,))
        grad_params = np.random.gamma(2.0, 2.0, (2,))
        grad_params = {"norm": grad_params[0], "var": grad_params[1]}
        fun = SpeedupFunction(params, grad_params, 128, 1280, (64, 256), True)
        speedup, bsz = fun(1, 1, return_local_bsz=True)
        assert(bsz == 128), "expected bsz = 128, got {}".format(bsz)
        assert(isinstance(speedup, float))

        replicas = np.asarray(range(1, 20))
        # single-node
        speedup, bsz = fun(np.ones_like(replicas), replicas,
                           return_local_bsz=True)
        assert(np.all(np.logical_or(bsz >= np.ceil(128 / replicas).astype(int),
                                    speedup == 0.0)))
        assert(np.all(np.logical_or(bsz >= (64),
                                    speedup == 0.0)))
        assert(np.all(bsz <= (256)))
        assert(np.all(np.logical_or(bsz * replicas <= 1280,
                                    speedup == 0.0)))
        assert(bsz[0] == 128)
        # multi-node
        speedup, bsz = fun(replicas, replicas, return_local_bsz=True)
        assert(np.all(np.logical_or(bsz >= np.ceil(128 / replicas).astype(int),
                                    speedup == 0.0)))
        assert(np.all(np.logical_or(bsz >= (64),
                                    speedup == 0.0)))
        assert(np.all(bsz <= (256)))
        assert(np.all(np.logical_or(bsz * replicas <= 1280,
                                    speedup == 0.0)))
        assert(bsz[0] == 128)
