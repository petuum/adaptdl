import collections
import glob
import math
import os
import pandas

from scipy.interpolate import interp1d, LinearNDInterpolator


def get(name):
    return APPLICATIONS[name]


def memoize(f):
    memo = {}
    def helper(*x):
        if x not in memo:
            memo[x] = f(*x)
        return memo[x]
    return helper


class Application(object):
    def __init__(self, trace_dir,
                 init_batch_size=None, max_batch_size=None,
                 min_local_bsz=None, max_local_bsz=None,
                 max_epochs=None, target_metric=None):
        self.name = os.path.basename(trace_dir)
        validation = {}
        for path in glob.glob(os.path.join(trace_dir, "validation-*.csv")):
            batch_size = int(path.split("-")[-1].split(".")[0])
            validation[batch_size] = pandas.read_csv(path)
        self.validation = collections.OrderedDict(sorted(validation.items()))
        self.placements = \
            pandas.read_csv(os.path.join(trace_dir, "placements.csv"))
        self.placements["num_nodes"] = \
            self.placements.placement.apply(lambda p: len(str(p)))
        self.placements["num_replicas"] = \
            self.placements.placement.apply(lambda p: sum(map(int, str(p))))
        self.scalability = \
            pandas.read_csv(os.path.join(trace_dir, "scalability.csv"))
        self.init_batch_size = init_batch_size or min(self.validation)
        self.max_batch_size = max_batch_size or max(self.validation)
        self.min_local_bsz = min_local_bsz or self.placements.local_bsz.min()
        self.max_local_bsz = max_local_bsz or self.placements.local_bsz.max()
        assert self.max_batch_size >= self.min_local_bsz
        self.max_epochs = max_epochs or min(map(len, self.validation.values()))
        self.target_metric = target_metric

    def _validated_batch_sizes(self, batch_size):
        # Find the lower-bound and upper-bound batch sizes (may be the same).
        lower_bsz = upper_bsz = None
        for bsz in self.validation:
            if bsz <= batch_size:
                lower_bsz = bsz
            if bsz >= batch_size:
                upper_bsz = bsz
                break
        assert lower_bsz is not None and upper_bsz is not None, \
               "{} {}".format(batch_size, list(self.validation))
        assert lower_bsz <= batch_size <= upper_bsz
        return lower_bsz, upper_bsz

    def get_configurations(self, lo_util=0.5, hi_util=0.8):
        # Assuming a cluster of 16 nodes each with 4 GPUs.
        ret = []
        base_jct = None
        base_batch_size = None
        for num_replicas in (1, 2, 4, 6, 8, 12, 16, 24, 32, 48):
            if num_replicas * self.min_local_bsz > self.max_batch_size:
                break
            placement = ()
            while sum(placement) < num_replicas:
                placement = (*placement, min(num_replicas - sum(placement), 4))
            best_jct = None
            best_batch_size = None
            for batch_size, valid in self.validation.items():
                local_bsz = math.ceil(batch_size / sum(placement) - 1e-8)
                if local_bsz < self.min_local_bsz:
                    continue
                accum_steps = math.ceil(local_bsz / self.max_local_bsz - 1e-8) - 1
                #if sum(placement) == 1 and batch_size > self.init_batch_size:
                #    accum_steps = max(1, accum_steps)
                atomic_bsz = math.ceil(local_bsz / (accum_steps + 1) - 1e-8)
                epoch = self.get_completion_epoch(batch_size)
                step_time, sync_time = self.get_throughput(placement, atomic_bsz)
                step_time += accum_steps * (step_time - sync_time)
                jct = valid.iteration[epoch] * step_time
                if best_jct is None or jct < best_jct:
                    best_jct = jct
                    best_batch_size = batch_size
            if num_replicas == 1:
                base_jct = best_jct
                base_batch_size = best_batch_size
            elif best_jct < 12 * 3600 and \
                    lo_util < base_jct / best_jct / num_replicas < hi_util:
                ret.append((num_replicas, best_batch_size, best_jct))
        if not ret:
            ret.append((1, base_batch_size, base_jct))
        return ret

    def get_best_batch_size(self, num_replicas):
        # Assuming a cluster of 16 nodes each with 4 GPUs.
        ret = []
        base_jct = None
        base_batch_size = None
        if num_replicas * self.min_local_bsz > self.max_batch_size:
            return None
        placement = ()
        while sum(placement) < num_replicas:
            placement = (*placement, min(num_replicas - sum(placement), 4))
        best_jct = None
        best_batch_size = None
        for batch_size, valid in self.validation.items():
            local_bsz = math.ceil(batch_size / sum(placement))
            if local_bsz < self.min_local_bsz:
                continue
            if local_bsz > self.max_local_bsz:
                break
            epoch = self.get_completion_epoch(batch_size)
            step_time, _ = self.get_throughput(placement, local_bsz)
            jct = valid.iteration[epoch] * step_time
            if best_jct is None or jct < best_jct:
                best_jct = jct
                best_batch_size = batch_size
        return best_batch_size

    def get_epoch(self, progress):
        return max(df.progress.searchsorted(progress, "right")
                   for df in self.validation.values())

    @memoize
    def get_progress(self, epoch):
        if epoch == 0:
            return 0.0
        return min(df.progress[epoch - 1] for df in self.validation.values())

    @memoize
    def get_completion_epoch(self, batch_size):
        if self.target_metric is None:
            return self.max_epochs - 1
        best_metric = None
        for epoch in range(self.max_epochs):
            next_metric = self.get_best_metric(batch_size, epoch)
            if best_metric is not None:
                sign = self.target_metric - best_metric
                if sign * (self.target_metric - next_metric) <= 0:
                    # Opposite signs, crossed target metric.
                    return epoch
        return epoch

    @memoize
    def get_iteration(self, batch_size, epoch):
        # Returns the number of iterations after completing a given epoch.
        lower_bsz, upper_bsz = self._validated_batch_sizes(batch_size)
        lower_it = self.validation[lower_bsz].iteration[epoch]
        upper_it = self.validation[upper_bsz].iteration[epoch]
        if lower_bsz == upper_bsz:
            assert lower_it == upper_it
            return lower_it
        # Linear interpolation between lower_bsz and upper_bsz.
        return ((batch_size - lower_bsz) * upper_it +
                (upper_bsz - batch_size) * lower_it) / (upper_bsz - lower_bsz)

    @memoize
    def get_best_metric(self, batch_size, epoch):
        # Returns the best observed validation metric before a given epoch.
        if epoch == 0:
            return None
        lower_bsz, upper_bsz = self._validated_batch_sizes(batch_size)
        if (next(iter(self.validation.values())).metric.values[0] <
            next(iter(self.validation.values())).metric.values[-1]):
            # Validation metric increases.
            lower_val = self.validation[lower_bsz].metric[:epoch].max()
            upper_val = self.validation[upper_bsz].metric[:epoch].max()
        else:
            lower_val = self.validation[lower_bsz].metric[:epoch].min()
            upper_val = self.validation[upper_bsz].metric[:epoch].min()
        if lower_bsz == upper_bsz:
            assert lower_val == upper_val
            return lower_val
        # Linear interpolation between lower_bsz and upper_bsz.
        return ((batch_size - lower_bsz) * upper_val +
                (upper_bsz - batch_size) * lower_val) / (upper_bsz - lower_bsz)

    @memoize
    def get_grad_stats(self, batch_size, epoch):
        # Returns the gradient sqr and var estimates during a given epoch.
        lower_bsz, upper_bsz = self._validated_batch_sizes(batch_size)
        lower_sqr = self.validation[lower_bsz].grad_sqr[epoch]
        upper_sqr = self.validation[upper_bsz].grad_sqr[epoch]
        lower_var = self.validation[lower_bsz].grad_var[epoch]
        upper_var = self.validation[upper_bsz].grad_var[epoch]
        if lower_bsz == upper_bsz:
            assert lower_sqr == upper_sqr and lower_var == upper_var
            return lower_sqr, lower_var
        # Linear interpolation between lower_bsz and upper_bsz.
        sqr = ((batch_size - lower_bsz) * upper_sqr +
               (upper_bsz - batch_size) * lower_sqr) / (upper_bsz - lower_bsz)
        var = ((batch_size - lower_bsz) * upper_var +
               (upper_bsz - batch_size) * lower_var) / (upper_bsz - lower_bsz)
        return sqr, var

    @memoize
    def get_throughput(self, placement, local_bsz):
        # Normalize placement to the lexicographically smallest rotation.
        placement = tuple(filter(None, placement))
        placement = min(placement[i:] + placement[:i]
                        for i in range(len(placement)))
        placement_id = int("".join(map(str, placement)))
        xs = ["num_nodes", "num_replicas", "local_bsz"]
        ys = ["step_time", "sync_time"]
        if placement_id in self.placements.placement.values:
            # Found in placement traces, interpolate between local_bsz.
            df = self.placements[self.placements.placement == placement_id]
            interpolator = interp1d(df.local_bsz.values, df[ys].values, axis=0)
            ret = interpolator(local_bsz)
        else:
            # Interpolate between num_nodes, num_replicas, and local_bsz.
            df = self.placements.groupby(xs)[xs + ys].mean()
            df = df.append(self.scalability, ignore_index=True)
            num_nodes, num_replicas = len(placement), sum(placement)
            num_nodes = min(num_nodes, 16)
            interpolator = LinearNDInterpolator(df[xs].values, df[ys].values)
            ret = interpolator([num_nodes, num_replicas, local_bsz])[0]
        assert sum(ret) == sum(ret), "{} {} {}".format(self.name, placement, local_bsz)
        return ret


TRACES_DIR = os.path.join(os.path.dirname(__file__), "traces")
APPLICATIONS = {
    "bert": Application(os.path.join(TRACES_DIR, "bert"), max_epochs=2),
    "cifar10": Application(os.path.join(TRACES_DIR, "cifar10"), max_epochs=100),
    "ncf": Application(os.path.join(TRACES_DIR, "ncf"), max_epochs=10),
    "imagenet": Application(os.path.join(TRACES_DIR, "imagenet"), max_epochs=90),
    "deepspeech2": Application(os.path.join(TRACES_DIR, "deepspeech2"), max_epochs=80),
    "yolov3": Application(os.path.join(TRACES_DIR, "yolov3"), max_epochs=50, max_local_bsz=8),
}
