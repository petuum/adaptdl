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
        fun = SpeedupFunction(params, grad_params, 128,
                              None, None, False, False)
        speedup, (bsz, steps) = fun(1, 3, return_config=True)
        assert(bsz == 128//3 + 1), "expected bsz = 43, got {}".format(bsz)
        assert(isinstance(speedup, float))

        replicas = np.asarray([1, 2, 3, 4, 5])
        # single-node
        speedup, (bsz, steps) = fun(np.ones_like(replicas), replicas,
                                    return_config=True)
        assert(bsz.shape == (5,))
        assert(np.all(bsz == np.ceil(128 / replicas).astype(int)))
        assert(speedup.shape == (5,))
        assert(bsz[0] == 128)
        assert(np.all(steps == 0))
        # multi-node
        speedup, (bsz, steps) = fun(replicas, replicas, return_config=True)
        assert(bsz.shape == (5,))
        assert(np.all(bsz == np.ceil(128 / replicas).astype(int)))
        assert(speedup.shape == (5,))
        assert(bsz[0] == 128)
        assert(np.all(steps == 0))


def test_local_bounds():
    np.random.seed(0)
    for i in range(100):
        params = np.random.gamma(2.0, 2.0, (7,))
        grad_params = np.random.gamma(2.0, 2.0, (2,))
        grad_params = {"norm": grad_params[0], "var": grad_params[1]}
        fun = SpeedupFunction(params, grad_params, 128, None, (64, 256),
                              False, True)
        speedup, (bsz, steps) = fun(1, 1, return_config=True)
        assert(bsz == 128), "expected bsz = 128, got {}".format(bsz)
        assert(isinstance(speedup, float))

        replicas = np.asarray(range(1, 100))
        # single-node
        speedup, (bsz, steps) = fun(np.ones_like(replicas), replicas,
                                    return_config=True)
        assert(np.all(bsz >= np.ceil(128 / replicas).astype(int)))
        assert(np.all(np.logical_or(bsz >= (64), speedup == 0.0)))
        assert(np.all(bsz <= (256)))
        assert(np.all(bsz * replicas <= 100 * 128))
        assert(bsz[0] == 128)
        assert(np.all(steps == 0))
        # multi-node
        speedup, (bsz, steps) = fun(replicas, replicas, return_config=True)
        assert(np.all(bsz >= np.ceil(128 / replicas).astype(int)))
        assert(np.all(np.logical_or(bsz >= (64), speedup == 0.0)))
        assert(np.all(bsz <= (256)))
        assert(np.all(bsz * replicas <= 100 * 128))
        assert(bsz[0] == 128)
        assert(np.all(steps == 0))


def test_max_bounds():
    np.random.seed(0)
    for i in range(100):
        params = np.random.gamma(2.0, 2.0, (7,))
        grad_params = np.random.gamma(2.0, 2.0, (2,))
        grad_params = {"norm": grad_params[0], "var": grad_params[1]}
        fun = SpeedupFunction(params, grad_params, 128, 1280,
                              None, False, True)
        speedup, (bsz, steps) = fun(1, 1, return_config=True)
        assert(bsz == 128), "expected bsz = 128, got {}".format(bsz)
        assert(isinstance(speedup, float))

        replicas = np.asarray(range(1, 100))
        # single-node
        speedup, (bsz, steps) = fun(np.ones_like(replicas), replicas,
                                    return_config=True)
        assert(np.all(bsz >= np.ceil(128 / replicas).astype(int)))
        assert(np.all(bsz * replicas <= 1280))
        assert(bsz[0] == 128)
        assert(np.all(steps == 0))
        # multi-node
        speedup, (bsz, steps) = fun(replicas, replicas, return_config=True)
        assert(np.all(bsz >= np.ceil(128 / replicas).astype(int)))
        assert(np.all(bsz * replicas <= 1280))
        assert(bsz[0] == 128)
        assert(np.all(steps == 0))


def test_all_bounds():
    np.random.seed(0)
    for i in range(100):
        params = np.random.gamma(2.0, 2.0, (7,))
        grad_params = np.random.gamma(2.0, 2.0, (2,))
        grad_params = {"norm": grad_params[0], "var": grad_params[1]}
        fun = SpeedupFunction(params, grad_params, 128, 1280, (64, 256),
                              False, True)
        speedup, (bsz, steps) = fun(1, 1, return_config=True)
        assert(bsz == 128), "expected bsz = 128, got {}".format(bsz)
        assert(isinstance(speedup, float))

        replicas = np.asarray(range(1, 20))
        # single-node
        speedup, (bsz, steps) = fun(np.ones_like(replicas), replicas,
                                    return_config=True)
        assert(np.all(np.logical_or(bsz >= np.ceil(128 / replicas).astype(int),
                                    speedup == 0.0)))
        assert(np.all(np.logical_or(bsz >= (64),
                                    speedup == 0.0)))
        assert(np.all(bsz <= (256)))
        assert(np.all(np.logical_or(bsz * replicas <= 1280,
                                    speedup == 0.0)))
        assert(bsz[0] == 128)
        assert(np.all(steps == 0))
        # multi-node
        speedup, (bsz, steps) = fun(replicas, replicas, return_config=True)
        assert(np.all(np.logical_or(bsz >= np.ceil(128 / replicas).astype(int),
                                    speedup == 0.0)))
        assert(np.all(np.logical_or(bsz >= (64),
                                    speedup == 0.0)))
        assert(np.all(bsz <= (256)))
        assert(np.all(np.logical_or(bsz * replicas <= 1280,
                                    speedup == 0.0)))
        assert(bsz[0] == 128)
        assert(np.all(steps == 0))


def test_gradient_accumulation():
    np.random.seed(0)
    for i in range(100):
        params = np.random.gamma(2.0, 2.0, (7,))
        grad_params = np.random.gamma(2.0, 2.0, (2,))
        grad_params = {"norm": grad_params[0], "var": grad_params[1]}
        fun = SpeedupFunction(params, grad_params, 128, 1280, (64, 256),
                              True, True)
        speedup, (bsz, steps) = fun(1, 1, return_config=True)
        assert(isinstance(speedup, float))

        replicas = np.asarray(range(1, 20))
        # single-node
        speedup, (bsz, steps) = fun(np.ones_like(replicas), replicas,
                                    return_config=True)
        assert(np.all(np.logical_or(bsz >= np.ceil(128 / replicas).astype(int),
                                    speedup == 0.0)))
        assert(np.all(np.logical_or(bsz >= (64),
                                    speedup == 0.0)))
        assert(np.all(bsz <= (256)))
        assert(np.all(np.logical_or(bsz * replicas <= 1280,
                                    speedup == 0.0)))
        assert(np.all(steps <= 15))
        assert(np.all(steps >= 0))
        assert(np.all(np.logical_or(np.multiply(steps, bsz) >= 256,
                                    steps == 0)))
        # multi-node
        speedup, (bsz, steps) = fun(replicas, replicas, return_config=True)
        assert(np.all(np.logical_or(bsz >= np.ceil(128 / replicas).astype(int),
                                    speedup == 0.0)))
        assert(np.all(np.logical_or(bsz >= (64),
                                    speedup == 0.0)))
        assert(np.all(bsz <= (256)))
        assert(np.all(np.logical_or(bsz * replicas <= 1280,
                                    speedup == 0.0)))
        assert(np.all(steps <= 15))
        assert(np.all(steps >= 0))
        assert(np.all(np.logical_or(np.multiply(steps, bsz) >= 256,
                                    steps == 0)))


def test_partition_local_bsz():
    # Note: params values don't matter for this test
    params = np.random.gamma(2.0, 2.0, (7,))
    grad_params = np.random.gamma(2.0, 2.0, (2,))
    grad_params = {"norm": grad_params[0], "var": grad_params[1]}
    max_batch_sizes = np.random.randint(100, 10000, 1000)
    local_bsz_bounds = np.minimum(
        max_batch_sizes, np.random.randint(32, 1000, 1000))
    init_batch_sizes = np.minimum(128, local_bsz_bounds)
    for max_batch_size, local_bsz_bound, init_batch_size in zip(
            max_batch_sizes, local_bsz_bounds, init_batch_sizes):
        samples = 1000
        fun = SpeedupFunction(
            params, grad_params, init_batch_size, max_batch_size,
            (16, local_bsz_bound), True, True)
        replicas = np.random.randint(1, 11, samples)
        batch_sizes = np.random.randint(init_batch_size,
                                        np.maximum(max_batch_size / replicas,
                                                   init_batch_size + 1),
                                        samples)
        atomic_bsz, steps = fun._partition_local_bsz(batch_sizes, replicas)
        # Ensure that single replicas can't scale the batchsize without
        # using gradient accumulation
        assert(np.all(np.logical_or(
                        np.logical_or(replicas != 1,
                                      atomic_bsz == init_batch_size),
                        steps > 0)))
        # Ensure that the total batch size doesn't exceed the user limit
        assert(np.all(steps * atomic_bsz <= max_batch_size))
        # Ensure that the atomic batch size is greater than the minimum
        assert(np.all(atomic_bsz > 16))
        # Ensure that when replicas > 1, the partitioned batch size
        # is close to the original value
        assert(np.all(np.logical_or(
            replicas == 1,
            np.abs(atomic_bsz * (steps + 1) - batch_sizes) < (steps + 1))))
