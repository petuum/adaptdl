import numpy as np
import pytest
import torch
import random

from unittest.mock import Mock

from adaptdl.torch.gradient_noise_scale import GradientNoiseScale


def test_object():
    params = [torch.tensor([[1., -1.], [2., 3.]], requires_grad=True),
              torch.tensor([[2., 3.]], requires_grad=True)]
    sgd = torch.optim.SGD(params, lr=0.1)
    adp = Mock(require_backward_grad_sync=True)
    obj = GradientNoiseScale(adp, sgd, accum_scale=1.0, num_replicas=1)
    assert(obj._accum_scale == 1.0)
    obj._num_replicas = 8
    obj.set_accum_scale(3.0)
    assert(obj.accum_scale == 3.0)
    obj._num_replicas = 4
    obj.set_accum_scale(3.0)
    assert(obj.accum_scale == 3.0)
    assert(np.isclose(obj.gain(2.0), 1.0))
    obj._state['var_avg'] = 3.0
    obj._state['norm_avg'] = 1.0
    assert(np.isclose(obj.gain(3.0), 2.0))


ATOL = 0.01


def test_nan():
    def nan_objective(tensor):
        if random.random() > 0.5:
            target = float("Nan")
        else:
            target = 4.0
        return (tensor - target)**2

    params_t = torch.Tensor([1.0])
    params = torch.autograd.Variable(params_t, requires_grad=True)
    sgd = torch.optim.SGD([params], lr=0.1)
    adp = Mock(require_backward_grad_sync=True)
    gns = GradientNoiseScale(adp, sgd, accum_scale=1.0, num_replicas=1)
    adp.gns = gns
    for i in range(100):
        gns.reset_accumulation()
        loss = nan_objective(params)
        loss.backward()
        if np.all(np.isfinite(loss.detach().numpy())):
            sgd.step()
        if params.allclose(torch.tensor([4.0]), atol=ATOL):
            break
    else:
        pytest.fail(f"Did not converge: {params}")
    if not (np.all(np.isfinite(gns.sqr_avg())) and
            np.all(np.isfinite(gns.var_avg()))):
        pytest.fail(f"non-finite adascale parameters:"
                    f"{gns.sqr_avg()}, {gns.var_avg()}")
