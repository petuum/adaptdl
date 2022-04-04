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

import collections
import collections.abc
import contextlib
import copy
import pickle

import adaptdl.checkpoint
import adaptdl.collective
from adaptdl.torch.epoch import current_epoch
from adaptdl.torch.data import current_dataloader


class Accumulator(collections.abc.MutableMapping):
    """
    This class helps aggregate simple statistics across all replicas in the
    current job, and across any number of checkpoint-restarts. Can be used to
    compute metrics like loss and accuracy, synchronized across each replica.

    Accumulators imitate python dictionaries, but with a few key differences
    described below. Primarily, its usage and behavior depend on whether it is
    set to *accumulation mode* or to *synchronized mode*.

    1.  **Accumulation mode:** the accumulator is being updated on all
        replicas. Operations like ``accum["key"] += val`` or
        ``accum.update(key=val)`` will aggregate the updates locally on each
        replica, which are lazily synchronized in the background (either upon a
        checkpoint or a switch to synchronized mode). Each replica may make
        different updates, which are summed together when synchronized. While
        accumulation mode is enabled, all read operations on the accumulator
        will behave as if they were performed on an empty ``dict``, ie.
        ``len(accum)`` will always return ``0``. By default, all accumulators
        are set to accumulation mode.
    2.  **Synchronized mode:** the accumulator contains the same data on every
        replica, and the application must ensure that all write operations are
        exactly the same across all replicas. While in synchronized mode, the
        accumulator may be used as if it were a native python ``dict``, and all
        read/write operations are supported. :meth:`Accumulator.synchronized`
        may be used to enter synchronized mode. Upon entering synchronized
        mode, the accumulator will automatically sum all updates from all
        replicas to ensure the same data is available to each replica.

    Using accumulators, many training/validation metrics can be computed
    easily and correctly in an elastic distributed setting. For example, a
    simple validation step which calculates a loss and accuracy can be
    implemented as follows:

    .. code-block:: python

       accum = Accumulator()  # New accumulator starts in accumulation mode.

       for epoch in remaining_epochs_until(60):

           for batch in validloader:
               ...
               accum["loss_sum"] += <loss summed within the batch>
               accum["correct"] += <number of correct predictions>
               accum["total"] += <total number of samples in the batch>

           with accum.synchronized():  # Enter synchronized mode.
               accum["loss_avg"] = accum["loss_sum"] / accum["total"]
               accum["accuracy"] = accum["correct"] / accum["total"]
               print("Loss: {}, Accuracy: {}".format(
                     accum["loss_avg"], accum["accuracy"]))
               accum.clear()
           # Back to accumulation mode.

    Arguments:
        args: Positional arguments same as ``dict``.
        kwargs: Keyword arguments same as ``dict``.

    .. automethod:: __iadd__
    .. automethod:: __isub__
    .. automethod:: __getitem__
    """
    def __init__(self, *args, **kwargs):
        self._sync_count = collections.Counter()
        self._synchronized = None
        self._state = _AccumulatorState(*args, **kwargs)
        adaptdl.checkpoint.load_state(self._state)

    @contextlib.contextmanager
    def synchronized(self):
        """
        A context manager which can be used to define the code to execute in
        *synchronized* mode. Within the context manager, any code can interact
        with this accumulator as if it were a regular Python ``dict``. The
        application must ensure that whatever operations performed within this
        context block are the same across all replicas.

        .. warning::
            Entering this context manager is a distributed synchronization
            point! Please ensure that all replicas enter this context manager
            at the same point in their code.
        """
        if self._synchronized is not None:
            # Already synchronized, don't need to do anything.
            yield self
            return
        epoch = current_epoch()
        # Remove saved results from all finished epochs. Since finished
        # epochs are never replayed, they should never be needed again.
        for key in list(self._state.results_history.keys()):
            if key is not None and key < epoch:
                self._state.results_history.pop(key)
        # Get the number of synchronizations so far in the current epoch.
        count = self._sync_count[epoch]
        self._sync_count[epoch] += 1
        results_list = self._state.results_history[epoch]
        assert count <= len(results_list)
        if count < len(results_list):
            # Results for this synchronization are saved in the history.
            self._synchronized = results_list[count]
            self._state.updates.clear()
        else:
            self._state.sync()  # Sync results and updates across replicas.
            if current_dataloader() is None:
                # Only save into results history if outside of a dataloader
                # iteration, since code inside iterations are not replayed.
                results_list.append(copy.deepcopy(self._state.results))
            self._synchronized = self._state.results
        try:
            yield self
        finally:
            self._synchronized = None

    def update(self, *args, **kwargs):
        """
        Apply a collection of key-update pairs. Unlike ``dict.update``, this
        method *additively* applies the updates to the accumulated values.

        Arguments:
            args: Positional arguments same as ``dict.update``. Can be a
                mapping object or an iterable of key-update pairs.
            kwargs: Keyword arguments same as ``dict.update``. Each keyword is
                the string key corresponding to the provided update.
        """
        for key, val in dict(*args, **kwargs).items():
            self[key] += val

    def subtract(self, *args, **kwargs):
        """
        Apply a collection of key-update pairs. Unlike
        :meth:`Accumulator.update`, this method *subtracts* the updates from
        the accumulated values.

        Arguments:
            args: Positional arguments same as :meth:`Accumulator.update`.
            kwargs: Keyword arguments same as :meth:`Accumulator.update`.
        """
        for key, val in dict(*args, **kwargs).items():
            self[key] -= val

    def __iadd__(self, other):
        """
        Supports the += operation, e.g. ``accum += {key1: val1, key2: val2}``.
        Behaves the same way as ``accum.update({key1: val1, key2: val2})``.

        Arguments:
            other: Mapping object or an iterable of key-update pairs.
        """
        self.update(other)
        return self

    def __isub__(self, other):
        """
        Supports the -= operation, e.g. ``accum -= {key1: val1, key2: val2}``.
        Behaves the same way as ``accum.subtract({key1: val1, key2: val2})``.

        Arguments:
            other: Mapping object or an iterable of key-update pairs.
        """
        self.subtract(other)
        return self

    def __getitem__(self, key):
        """
        Supports indexing, e.g. ``val = accum[key]`` and ``accum[key] += 1``.
        The former (read access) should only be used when the accumulator is in
        synchronized mode.

        Arguments:
            other: Key used to access a value in the accumulator.
        """
        if self._synchronized is not None:
            return self._synchronized.__getitem__(key)
        # In accumulation mode, return a dummy object which captures all
        # updates performed on it, to be later applied by __setitem__.
        return _Value(self, key)

    def __setitem__(self, key, value):
        if self._synchronized is not None:
            return self._synchronized.__setitem__(key, value)
        # Whenever an in-place addition or subtraction is done, like a[k] += v,
        # python will essentially perform 3 steps: (1) tmp = a.__getitem__(k),
        # (2) tmp += v, (3) a.__setitem__(k, tmp). In order to obtain the
        # update v, we let a.__getitem__(k) return an opaque object which
        # implements the __add__ operator to capture the update v in step (2).
        # Then, a.__setitem__(k, tmp) can pull v out of tmp and accumulate it.
        if not isinstance(value, _Value):
            raise TypeError("invalid value type: {}".format(type(value)))
        if value.accum is not self:
            raise ValueError("incompatible {}".format(self.__class__.__name__))
        if key != value.key:
            raise ValueError("incompatible key: {}".format(value.key))
        self._state.updates.setdefault(key, 0)
        self._state.updates[key] += value.update

    # Rest of the abstract methods needed by collections.MutableMapping

    def __contains__(self, key):
        if self._synchronized is not None:
            return self._synchronized.__contains__(key)
        return {}.__contains__(key)

    def __delitem__(self, key):
        if self._synchronized is not None:
            return self._synchronized.__delitem__(key)
        return {}.__delitem__(key)

    def __iter__(self):
        if self._synchronized is not None:
            return self._synchronized.__iter__()
        return {}.__iter__()

    def __len__(self):
        if self._synchronized is not None:
            return self._synchronized.__len__()
        return {}.__len__()

    def __repr__(self):
        if self._synchronized is not None:
            return self._synchronized.__repr__()
        return {}.__repr__()


class _Value(object):
    __slots__ = ["accum", "key", "update"]

    def __init__(self, accum, key):
        # Initialize the opaque object used for supporting "accum[k] += v" and
        # "accum[k] -= v" operations.
        self.accum = accum
        self.key = key
        self.update = 0

    def __add__(self, update):
        if isinstance(update, _Value):
            raise TypeError("invalid update type: {}".format(type(update)))
        self.update += update
        return self

    def __sub__(self, update):
        if isinstance(update, _Value):
            raise TypeError("invalid update type: {}".format(type(update)))
        self.update -= update
        return self


class _AccumulatorState(adaptdl.checkpoint.State):

    # Assume accumulators are initialized in the same order in every replica.
    # Keep a map of epoch -> number of accumulators initialized so far in that
    # epoch, and use that count to construct a unique name for the state.
    init_count = collections.Counter()

    def __init__(self, *args, **kwargs):
        if current_dataloader() is not None:
            raise RuntimeError("accumulator may not be initialized during "
                               "dataloader iteration")
        epoch = current_epoch()
        count = _AccumulatorState.init_count[epoch]
        super().__init__("adaptdl-accumulator-epoch{}-{}".format(epoch, count))
        _AccumulatorState.init_count[epoch] += 1

        self.results_history = collections.defaultdict(list)
        self.results = dict(*args, **kwargs)
        self.updates = {}

    def save(self, fileobj):
        pickle.dump((self.results_history, self.results), fileobj)

    def load(self, fileobj):
        self.results_history, self.results = pickle.load(fileobj)

    def sync(self):
        # Aggregate pending updates across all replicas and apply them.
        updates = adaptdl.collective.allreduce(self.updates, _dict_iadd)
        _dict_iadd(self.results, updates)
        self.updates.clear()


def _dict_iadd(a, b):
    for k, v in b.items():
        if k in a:
            a[k] += v
        else:
            a[k] = v
    return a
