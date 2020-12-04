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


from adaptdl.goodput import GoodputFunction, PerfParams, GradParams
import itertools
import numpy as np
import pytest

RNG = np.random.RandomState(0)
PERF_PARAMS = [PerfParams(*RNG.gamma(2.0, 2.0, [7])) for i in range(10)]
GRAD_PARAMS = [GradParams(*RNG.gamma(2.0, 2.0, [2])) for i in range(10)]


def groupby_indices(*args):
    _, indices = np.unique(np.stack(args), axis=1, return_inverse=True)
    groups = {}
    for i, g in enumerate(indices):
        groups.setdefault(g, []).append(i)
    return list(groups.values())


@pytest.mark.parametrize("perf_params", PERF_PARAMS)
@pytest.mark.parametrize("grad_params", GRAD_PARAMS)
def test_evaluate(perf_params, grad_params):
    init_batch_size = 16
    goodput_fn = GoodputFunction(perf_params, grad_params, init_batch_size)
    # Generate a range of different goodput function arguments.
    num_nodes = np.array([1, 2, 3, 4])
    num_replicas = np.array([1, 2, 4, 8])
    atomic_bsz = np.array([8, 12, 16, 20, 24])
    accum_steps = np.array([0, 1, 2, 3, 4])
    # Cartesian product.
    num_nodes, num_replicas, atomic_bsz, accum_steps = \
        map(np.array, zip(*itertools.product(num_nodes, num_replicas,
                                             atomic_bsz, accum_steps)))
    # Only keep valid arguments.
    valid = np.logical_and(num_nodes <= num_replicas, init_batch_size
                           <= num_replicas * atomic_bsz * accum_steps)
    num_nodes = num_nodes[valid]
    num_replicas = num_replicas[valid]
    atomic_bsz = atomic_bsz[valid]
    accum_steps = accum_steps[valid]
    # Evaluate goodput.
    goodput = goodput_fn(num_nodes, num_replicas, atomic_bsz, accum_steps)
    throughput = goodput_fn.throughput(num_nodes, num_replicas,
                                       atomic_bsz, accum_steps)
    efficiency = goodput_fn.efficiency(num_replicas * atomic_bsz
                                       * (accum_steps + 1))
    # Check basic invariants.
    assert np.all(0 <= throughput)
    assert np.all(0 <= efficiency) and np.all(efficiency <= 1)
    assert np.allclose(goodput, throughput * efficiency)
    # Increasing batch size should decrease efficiency.
    batch_size = num_replicas * atomic_bsz * (accum_steps + 1)
    sort = np.argsort(batch_size)
    assert np.all(np.diff(efficiency[sort]) <= 0)
    # All else equal, increasing atomic_bsz should increase throughput.
    for indices in groupby_indices(num_nodes, num_replicas, accum_steps):
        sort = np.argsort(atomic_bsz[indices])
        assert np.all(np.diff(throughput[indices][sort]) >= 0)
        # Increasing throughput should experience diminishing returns.
        if len(indices) > 1:
            diffx = np.diff(atomic_bsz[indices][sort])
            diffy = np.diff(throughput[indices][sort])
            assert np.all(diffx[:-1] * diffy[1:] - diffx[1:] * diffy[:-1] <= 0)
    # All else equal, scalability is sublinear with respect to num_replicas.
    for indices in groupby_indices(num_nodes, atomic_bsz, accum_steps):
        scalability = throughput / num_replicas
        sort = np.argsort(num_replicas[indices])
        assert np.all(np.diff(scalability[indices][sort]) <= 0)


@pytest.mark.parametrize("perf_params", PERF_PARAMS)
@pytest.mark.parametrize("grad_params", GRAD_PARAMS)
def test_optimize_no_bounds(perf_params, grad_params):
    goodput_fn = GoodputFunction(perf_params, grad_params, 128)
    goodput, bsz, steps = goodput_fn.optimize(1, 3)
    assert(bsz == 128//3 + 1), "expected bsz = 43, got {}".format(bsz)
    assert(isinstance(goodput, float))

    replicas = np.asarray([1, 2, 3, 4, 5])
    # single-node
    goodput, bsz, steps = goodput_fn.optimize(np.ones_like(replicas), replicas)
    assert(bsz.shape == (5,))
    assert(np.all(bsz == np.ceil(128 / replicas).astype(int)))
    assert(goodput.shape == (5,))
    assert(bsz[0] == 128)
    assert(np.all(steps == 0))
    # multi-node
    goodput, bsz, steps = goodput_fn.optimize(replicas, replicas)
    assert(bsz.shape == (5,))
    assert(np.all(bsz == np.ceil(128 / replicas).astype(int)))
    assert(goodput.shape == (5,))
    assert(bsz[0] == 128)
    assert(np.all(steps == 0))


@pytest.mark.parametrize("perf_params", PERF_PARAMS)
@pytest.mark.parametrize("grad_params", GRAD_PARAMS)
def test_optimize_local_bounds(perf_params, grad_params):
    fun = GoodputFunction(perf_params, grad_params, 128)
    goodput, bsz, steps = fun.optimize(1, 1, atomic_bsz_range=(64, 256))
    assert(bsz == 128), "expected bsz = 128, got {}".format(bsz)
    assert(isinstance(goodput, float))

    replicas = np.asarray(range(1, 100))
    # single-node
    goodput, bsz, steps = fun.optimize(np.ones_like(replicas), replicas,
                                       atomic_bsz_range=(64, 256))
    assert(np.all(bsz >= np.ceil(128 / replicas).astype(int)))
    assert(np.all(np.logical_or(bsz >= (64), goodput == 0.0)))
    assert(np.all(bsz <= (256)))
    assert(np.all(bsz * replicas <= 100 * 128))
    assert(bsz[0] == 128)
    assert(np.all(steps == 0))
    # multi-node
    goodput, bsz, steps = fun.optimize(replicas, replicas,
                                       atomic_bsz_range=(64, 256))
    assert(np.all(bsz >= np.ceil(128 / replicas).astype(int)))
    assert(np.all(np.logical_or(bsz >= (64), goodput == 0.0)))
    assert(np.all(bsz <= (256)))
    assert(np.all(bsz * replicas <= 100 * 128))
    assert(bsz[0] == 128)
    assert(np.all(steps == 0))


@pytest.mark.parametrize("perf_params", PERF_PARAMS)
@pytest.mark.parametrize("grad_params", GRAD_PARAMS)
def test_optimize_max_bounds(perf_params, grad_params):
    fun = GoodputFunction(perf_params, grad_params, 128)
    goodput, bsz, steps = fun.optimize(1, 1, max_batch_size=1280)
    assert(bsz == 128), "expected bsz = 128, got {}".format(bsz)
    assert(isinstance(goodput, float))

    replicas = np.asarray(range(1, 100))
    # single-node
    goodput, bsz, steps = fun.optimize(np.ones_like(replicas), replicas,
                                       max_batch_size=1280)
    assert(np.all(bsz >= np.ceil(128 / replicas).astype(int)))
    assert(np.all(bsz * replicas <= 1280 + replicas))
    assert(bsz[0] == 128)
    assert(np.all(steps == 0))
    # multi-node
    goodput, bsz, steps = fun.optimize(replicas, replicas, max_batch_size=1280)
    assert(np.all(bsz >= np.ceil(128 / replicas).astype(int)))
    assert(np.all(bsz * replicas <= 1280 + replicas))
    assert(bsz[0] == 128)
    assert(np.all(steps == 0))


@pytest.mark.parametrize("perf_params", PERF_PARAMS)
@pytest.mark.parametrize("grad_params", GRAD_PARAMS)
def test_optimize_all_bounds(perf_params, grad_params):
    fun = GoodputFunction(perf_params, grad_params, 128)
    goodput, bsz, steps = fun.optimize(1, 1, max_batch_size=1280,
                                       atomic_bsz_range=(64, 256))
    assert(bsz == 128), "expected bsz = 128, got {}".format(bsz)
    assert(isinstance(goodput, float))

    replicas = np.asarray(range(1, 20))
    # single-node
    goodput, bsz, steps = fun.optimize(np.ones_like(replicas), replicas,
                                       max_batch_size=1280,
                                       atomic_bsz_range=(64, 256))
    assert(np.all(np.logical_or(bsz >= np.ceil(128 / replicas).astype(int),
                                goodput == 0.0)))
    assert(np.all(np.logical_or(bsz >= (64),
                                goodput == 0.0)))
    assert(np.all(bsz <= (256)))
    assert(np.all(np.logical_or(bsz * replicas <= 1280 + replicas,
                                goodput == 0.0)))
    assert(bsz[0] == 128)
    assert(np.all(steps == 0))
    # multi-node
    goodput, bsz, steps = fun.optimize(replicas, replicas,
                                       max_batch_size=1280,
                                       atomic_bsz_range=(64, 256))
    assert(np.all(np.logical_or(bsz >= np.ceil(128 / replicas).astype(int),
                                goodput == 0.0)))
    assert(np.all(np.logical_or(bsz >= (64),
                                goodput == 0.0)))
    assert(np.all(bsz <= (256)))
    assert(np.all(np.logical_or(bsz * replicas <= 1280 + replicas,
                                goodput == 0.0)))
    assert(bsz[0] == 128)
    assert(np.all(steps == 0))
    # multi-node edge case
    replicas = 4
    goodput, bsz, steps = fun.optimize(4, 4, max_batch_size=1024,
                                       atomic_bsz_range=(128, 128))
    assert goodput > 0.0
    assert bsz == 128
    assert steps == 0


@pytest.mark.parametrize("perf_params", PERF_PARAMS)
@pytest.mark.parametrize("grad_params", GRAD_PARAMS)
def test_optimize_accumulation(perf_params, grad_params):
    fun = GoodputFunction(perf_params, grad_params, 128)
    goodput, bsz, steps = fun.optimize(1, 1, max_batch_size=1280,
                                       atomic_bsz_range=(64, 256),
                                       accumulation=True)
    assert(isinstance(goodput, float))

    replicas = np.asarray(range(1, 20))
    # single-node
    goodput, bsz, steps = fun.optimize(np.ones_like(replicas), replicas,
                                       max_batch_size=1280,
                                       atomic_bsz_range=(64, 256),
                                       accumulation=True)
    assert(np.all(np.logical_or(bsz >= np.ceil(128 / replicas).astype(int),
                                goodput == 0.0)))
    assert(np.all(np.logical_or(bsz >= (64),
                                goodput == 0.0)))
    assert(np.all(bsz <= (256)))
    assert(np.all(np.logical_or(bsz * replicas * (steps + 1) <
                                1280 + replicas * (steps + 1),
                                goodput == 0.0)))
    assert(np.all(steps <= 15))
    assert(np.all(steps >= 0))
    assert(np.all(np.logical_or(replicas > 1,
                                np.logical_or(bsz == 128, steps > 0))))
    # multi-node
    goodput, bsz, steps = fun.optimize(replicas, replicas,
                                       max_batch_size=1280,
                                       atomic_bsz_range=(64, 256),
                                       accumulation=True)
    assert(np.all(np.logical_or(bsz >= np.ceil(128 / replicas).astype(int),
                                goodput == 0.0)))
    assert(np.all(np.logical_or(bsz >= (64),
                                goodput == 0.0)))
    assert(np.all(bsz <= (256)))
    assert(np.all(np.logical_or(bsz * replicas * (steps + 1) <
                                1280 + replicas * (steps + 1),
                                goodput == 0.0)))
    assert(np.all(steps <= 15))
    assert(np.all(steps >= 0))
    assert(np.all(np.logical_or(np.multiply(steps, bsz) >= 256,
                                steps == 0)))


@pytest.mark.parametrize("perf_params", PERF_PARAMS)
@pytest.mark.parametrize("grad_params", GRAD_PARAMS)
def test_one_replica_accumulation(perf_params, grad_params):
    fun = GoodputFunction(perf_params, grad_params, 128)

    replicas = np.asarray([1])
    max_batch_sizes = np.asarray(range(128, 128 * 20, 128))
    # single-node
    for max_batch_size in max_batch_sizes:
        goodput, bsz, steps = fun.optimize(np.ones_like(replicas), replicas,
                                           max_batch_size=1280,
                                           atomic_bsz_range=(64, 256),
                                           accumulation=True)
        assert(np.all(np.logical_or(bsz >= (64),
                                    goodput == 0.0)))
        assert(np.all(bsz <= (256)))
        assert(np.all(np.logical_or(bsz * (steps + 1) <=
                                    max_batch_size,
                                    goodput == 0.0)))
        assert(np.all(np.logical_or(bsz >= np.ceil(128 / replicas).astype(int),
                                    goodput == 0.0)))
        assert(np.all(np.logical_or(bsz * (steps + 1) != 128,
                                    steps == 0)))
