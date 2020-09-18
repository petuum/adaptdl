import adaptdl.speedup as speedup
import numpy as np


def test_fit_1():
    size = (1000,)
    nodes = np.random.randint(low=1, high=11, size=size)
    replicas = np.random.randint(low=1, high=nodes+1, size=size)
    local_bsz = np.random.randint(32, 1024, size=size)
    grad_acc_steps = np.random.randint(0, 1, size=size)
    params = speedup.Params(0.1, 0.01, 0.08, 0.008, 0.5, 1.0, 1e-6, 1e-6, 1.2)
    step_time_compute = speedup._predict_compute(params, local_bsz) + \
        np.maximum(np.random.normal(0, 0.001, size=size), 0.0)
    step_time_network = speedup._predict_network(params, nodes, replicas) + \
        np.maximum(np.random.normal(0, 0.001, size=size), 0.0)
    step_time_accumulation = \
        speedup._predict_accumulation(params, local_bsz, grad_acc_steps) + \
        np.maximum(np.random.normal(0, 0.001, size=size), 0.0)
    gamma = params.gamma
    step_time = (
        (step_time_compute + step_time_accumulation * grad_acc_steps) ** gamma
        + step_time_network ** gamma) ** (1 / gamma)
    result = speedup.fit(nodes, replicas, local_bsz, grad_acc_steps,
                         step_time, step_time_compute, step_time_accumulation)
    loss_result = speedup._obj_fn(
        result, nodes, replicas, local_bsz, grad_acc_steps,
        step_time, step_time_compute, step_time_accumulation)
    loss_true = speedup._obj_fn(
        params, nodes, replicas, local_bsz, grad_acc_steps,
        step_time, step_time_compute, step_time_accumulation)
    assert(abs(loss_result - loss_true) < 0.01 * loss_true)


def test_fit_2():
    size = (1000,)
    nodes = np.random.randint(low=1, high=11, size=size)
    replicas = np.random.randint(low=1, high=nodes+1, size=size)
    local_bsz = np.random.randint(32, 1024, size=size)
    grad_acc_steps = np.random.randint(0, 10, size=size)
    params = speedup.Params(0.1, 0.01, 0.08, 0.008, 0.5, 1.0, 1e-6, 1e-6, 1.2)
    step_time_compute = speedup._predict_compute(params, local_bsz) + \
        np.maximum(np.random.normal(0, 0.01, size=size), 0.0)
    step_time_network = speedup._predict_network(params, nodes, replicas) + \
        np.maximum(np.random.normal(0, 0.01, size=size), 0.0)
    step_time_accumulation = \
        speedup._predict_accumulation(params, local_bsz, grad_acc_steps) + \
        np.maximum(np.random.normal(0, 0.01, size=size), 0.0)
    gamma = params.gamma
    step_time = (
        (step_time_compute + step_time_accumulation * grad_acc_steps) ** gamma
        + step_time_network ** gamma) ** (1 / gamma)
    speedup.fit(nodes, replicas, local_bsz, grad_acc_steps,
                step_time, step_time_compute, step_time_accumulation)
    # TODO: Complete this
