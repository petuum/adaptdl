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

"""
This module provides tools for the top-level loop over epochs during training.
AdaptDL expects the training program to be implemented as loop over several
epochs, each containing a series of loops over datasets (e.g. one loop over the
training set followed by one loop over the validation set). The program can be
interrupted between every iteration of any dataset loop, trigger a checkpoint
to be taken, and restarted using a different set of replicas.

**Due to checkpoint-restarts, parts of the training program may be executed
multiple times (e.g. once after each restart)!** To avoid incorrect execution,
ensure that your code is idempotent_ in the following locations:

1.  Immediately before any epoch loop (using :func:`remaining_epochs_until`).
2.  Immediately before any dataset loop (using
    :class:`adaptdl.torch.data.AdaptiveDataLoader`).

Your code may be non-idempotent in other locations.

.. code-block:: python

    ### IDEMPOTENT CODE ONLY ###

    for epoch in remaining_epochs_until(30):

        ### IDEMPOTENT CODE ONLY ###

        for batch in train_loader:
            # ... any code ...

        ### IDEMPOTENT CODE ONLY ###

        for batch in valid_loader:
            # ... any code ...

        # ... any code ...

    # ... any code ...

    ### END PROGRAM ###

For example, a common non-idempotent operation is learning-rate annealing:

.. code-block:: python

    for epoch in remaining_epochs_until(30):

        lr_scheduler.step()  # (A) WRONG!

        for batch in train_loader:
            # ...

        lr_scheduler.step()  # (B) WRONG!

        for batch in valid_loader:
            # ...

        lr_scheduler.step()  # (C) OK!

Location (A) will be executed again after any checkpoint-restart during either
the training or validation loop, resulting in the learning rate being annealed
several times in one epoch! Similarly with location (B), if checkpoint-restart
happens during the validation loop.

Location (C) results in the correct behavior, because (1) an epoch will not be
repeated once it has finished, and (2) no checkpoint-restarts can occur between
the learning rate annealing and the end of the epoch.

.. _idempotent: https://stackoverflow.com/a/1077421
"""

import logging
import pickle

import adaptdl.checkpoint


logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


def remaining_epochs_until(epoch):
    """
    Iterate over epochs in a way that is consistent with checkpoint-restarts.
    For example:

    .. code-block:: python

        for epoch in remaining_epochs_until(30):
            print(current_epoch())  # Should print 0 through 29

        for epoch in remaining_epochs_until(60):
            print(current_epoch())  # Should print 30 through 59

    If a checkpoint-restart happens during an epoch, all previous epochs will
    be skipped after the program restarts.

    Arguments:
        epoch (int): The epoch number to end at (exclusively).

    Raises:
        RuntimeError: If invoked before a previous epoch loop has ended.
    """
    if current_epoch() is not None:
        raise RuntimeError("overlapping epoch loops detected")
    if finished_epochs() < epoch:
        LOG.info("starting at epoch %s", finished_epochs())
    else:
        LOG.info("skipping all epochs up to %s", epoch)
    while finished_epochs() < epoch:
        _epoch_state().current_epoch = finished_epochs()
        try:
            yield current_epoch()
        finally:
            # Try to catch any exits from epoch loop, including breaks and
            # Exceptions. See https://www.peterbe.com/plog/generatorexit.
            _epoch_state().finished_epochs += 1
            _epoch_state().current_epoch = None


def current_epoch():
    """
    Get the current epoch while iterating with :func:`remaining_epochs_until`.

    Returns:
        int or None: The current epoch number if called from within a
        :func:`remaining_epochs_until` iteration, ``None`` otherwise.
    """
    return _epoch_state().current_epoch


def finished_epochs():
    """
    Get the number of epochs finished using :func:`remaining_epochs_until`.

    Returns:
        int: The number of finished epochs. Equal to :func:`current_epoch`
        if called from within a :func:`remaining_epochs_until` iteration.
    """
    return _epoch_state().finished_epochs


class _EpochState(adaptdl.checkpoint.State):
    def __init__(self):
        super().__init__(".adaptdl-epoch")
        self.finished_epochs = 0
        self.current_epoch = None

    def save(self, fileobj):
        pickle.dump(self.finished_epochs, fileobj)

    def load(self, fileobj):
        self.finished_epochs = pickle.load(fileobj)


def _epoch_state():
    global _EPOCH_STATE
    if _EPOCH_STATE is None:
        _EPOCH_STATE = _EpochState()
        adaptdl.checkpoint.load_state(_EPOCH_STATE)
    return _EPOCH_STATE


_EPOCH_STATE = None
