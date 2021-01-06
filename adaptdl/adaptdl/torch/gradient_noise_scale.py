import functools
import logging
import numpy as np
import torch.distributed
import torch.optim

from torch.autograd import Variable

__all__ = ["GradientNoiseScale"]

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


def _average_groups(grads1, grads2):
    ret = []
    for group1, group2 in zip(grads1, grads2):
        ret.append([])
        for g1, g2 in zip(group1, group2):
            if g1 is None:
                ret[-1].append(g2)
            elif g2 is None:
                ret[-1].append(g1)
            else:
                ret[-1].append((g1 + g2) / 2)
    return ret


def _normsqr_groups(grads):
    ret = []
    for group in grads:
        normsqr = [g.pow(2).sum(dtype=torch.float64)
                   for g in group if g is not None]
        ret.append(sum(normsqr).item() if normsqr else 0.0)
    return np.array(ret)


class GradientNoiseScale(object):
    """This class tracks gradient related stats and takes care of gradient
    accumulation."""
    def __init__(self, adp, optimizer,
                 mp_scaler=None,
                 num_replicas=None,
                 accum_scale=None):
        self._adp = adp
        self._optimizer = optimizer
        self._orig_optimizer_zero_grad = optimizer.zero_grad
        self._should_zero_grad = True
        self._mp_scaler = mp_scaler
        self._local_sqr = None
        self._num_replicas = (num_replicas if num_replicas is not None
                              else torch.distributed.get_world_size())
        self._accum_scale = accum_scale or self._num_replicas
        self._prev_grads = None

        self.reset_accumulation()

        self._optimizer.state.setdefault("gns", {
            "progress": 0.0,
            "prev_scale": 0.0,
            # Averages of n and v
            "sqr_avg": np.ones(len(optimizer.param_groups)),
            "var_avg": np.zeros(len(optimizer.param_groups)),
            # Whether estimates are biased (using differenced estimator).
            "biased": False,
        })

        for idx, param_group in enumerate(self._optimizer.param_groups):
            for param in param_group["params"]:
                param.register_hook(
                    functools.partial(self._backward_hook, idx, param))
        self._callback_queued = False
        self._smoothing = 0.999

    @property
    def _state(self):
        return self._optimizer.state["gns"]

    def reset_accumulation(self):
        """reset accumulation calculations and gradients."""
        self._orig_optimizer_zero_grad()
        self._local_sqr = None
        self._accum_count = 0

    @property
    def should_zero_grad(self):
        return self._should_zero_grad

    @property
    def accum_scale(self):
        return self._accum_scale

    @property
    def accum_count(self):
        return self._accum_count

    def set_accum_scale(self, accum_scale):
        if not np.isclose(self._accum_scale, accum_scale):
            self.reset_accumulation()
            self._accum_scale = accum_scale

    @property
    def raw_sqr_avg(self):
        view = self._state["sqr_avg"].view()
        view.flags.writeable = False
        return view

    def sqr_avg(self):
        """
        Current estimate of the squared l2-norm of the true gradient (sigma
        squared).

        Returns (float): Estimate of squared l2-norm.
        """
        return float(np.sum(np.maximum(self._state["sqr_avg"], 0.0)))

    @property
    def raw_var_avg(self):
        view = self._state["var_avg"].view()
        view.flags.writeable = False
        return view

    def var_avg(self):
        """
        Current estimate of the trace of the covariance of the true gradient
        (mu squared).

        Returns (float): Estimate of trace of the covariance.
        """
        return float(np.sum(np.maximum(self._state["var_avg"], 1e-6)))

    def get_progress(self):
        return self._state["progress"]

    def set_progress(self, progress):
        self._state["progress"] = progress

    def gain(self, scale):
        """
        Current estimate of the GradientNoiseScale gain ratio.

        Arguments:
            scale (float): The total scale to estimate the gain ratio for.

        Returns (float): Estimate of gain ratio.
        """
        var = self.var_avg()
        norm = self.sqr_avg()
        return (var + norm) / (var / scale + norm)

    def _update_avg(self, param_name, value, factor):
        biased = self._state.get(param_name + "_biased", 0.0)
        unbias = self._state.get(param_name + "_unbias", 0.0)
        biased = factor * biased + (1.0 - factor) * value
        unbias = factor * unbias + (1.0 - factor)
        self._state[param_name + "_biased"] = biased
        self._state[param_name + "_unbias"] = unbias
        self._state[param_name] = biased / unbias

    def _reset_avg(self, param_name):
        self._state.pop(param_name + "_biased", None)
        self._state.pop(param_name + "_unbias", None)

    def _backward_hook(self, idx, param, grad):
        # This method should be invoked once for each parameter during the
        # backward pass, before gradients are synchronized between replicas.
        if self._local_sqr is None:
            self._local_sqr = torch.zeros(len(self._optimizer.param_groups),
                                          device=grad.device,
                                          dtype=torch.float64)
        # Update the local gradient square sum
        self._local_sqr[idx] += grad.detach().pow(2).sum(dtype=torch.float64)
        if not self._callback_queued:
            Variable._execution_engine.queue_callback(self._queue_callback)
        self._callback_queued = True

    def _queue_callback(self):
        # This method should be invoked after the entire backward pass. We want
        # to make sure self._final_callback is invoked once, only after all
        # gradients have been synchronized between each replica. However, the
        # synchronization code in DistributedDataParallel is also done in a
        # callback, which might not yet be executed. Therefore, we enqueue
        # self._final_callback from this method, which should ensure it is
        # invoked after the gradient synchronization callback.
        self._callback_queued = False
        self._accum_count += 1
        if self._adp.require_backward_grad_sync:
            # Asynchronously sum the local squared-gradient statistics. The
            # actual gradient averaging should also be happening at the same
            # time, until self._final_callback is invoked.
            if self._num_replicas > 1:
                self._async_op = torch.distributed.all_reduce(self._local_sqr,
                                                              async_op=True)
            Variable._execution_engine.queue_callback(self._final_callback)
            self._should_zero_grad = True
        else:
            # Keep on accumulating gradients, should not zero grad.
            self._should_zero_grad = False

    def _final_callback(self):
        # This method should be invoked once the gradients have been
        # synchronized between all replicas and accumulation steps.
        if self._num_replicas > 1:
            self._async_op.wait()

        grads = []
        if self._mp_scaler is not None:
            mixed_precision_scale = self._mp_scaler.get_scale()
        else:
            mixed_precision_scale = 1.0
        for group in self._optimizer.param_groups:
            grads.append([])
            for param in group["params"]:
                if param.grad is None:
                    grads[-1].append(None)
                    continue
                grad = param.grad.detach().float()
                grads[-1].append(
                    grad / mixed_precision_scale / self._accum_count)
        # Note: mixed precision can result in nan/inf gradients,
        # which propogate into our norm and variance estimates.
        # Mixed precision autoscaling skips the skip where
        # there are nan/inf, so we also skip the update here
        grads_normsqr = _normsqr_groups(grads)
        if not np.all(np.isfinite(grads_normsqr)):
            LOG.warning("GradientNoiseScale detected invalid gradient! "
                        "Skipping step.")
            return
        count = self._num_replicas * self._accum_count
        scale = self._accum_scale * self._accum_count
        if count > 1:
            # Average local squared-norm samples.
            local_sqr = self._local_sqr.cpu().numpy() / count
            # Gradient is squared in local_sqr, so need to square the
            # mixed precision scale as well
            local_sqr = (local_sqr / mixed_precision_scale ** 2)
            total_sqr = grads_normsqr
            if self._state["biased"]:
                self._reset_avg("sqr_avg")
                self._reset_avg("var_avg")
            self._state["biased"] = False
            self._prev_grads = None
        else:
            # Single gradient datapoint, use difference estimation.
            if self._prev_grads is not None:
                local_sqr = (_normsqr_groups(self._prev_grads) +
                             grads_normsqr) / 2
                avg_grads = _average_groups(grads, self._prev_grads)
                total_sqr = _normsqr_groups(avg_grads)
                count = 2
                scale = 2 * self._accum_scale
            self._state["biased"] = True
            self._prev_grads = [[g.clone() if g is not None else None
                                 for g in group] for group in grads]
        if count > 1:
            grad_sqr = (count * total_sqr - local_sqr) / (count - 1)
            grad_var = (local_sqr - total_sqr) * scale / (count - 1)
            theta = self._smoothing ** scale
            self._update_avg('sqr_avg', grad_sqr, theta)
            self._update_avg('var_avg', grad_var, theta)
