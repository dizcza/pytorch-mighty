"""
Timer and schedulers
--------------------

.. autosummary::
    :toctree: toctree/monitor

    BatchTimer
    ScheduleStep
    ScheduleExp
"""


import math
from abc import ABC, abstractmethod
from functools import wraps
from typing import Callable


__all__ = [
    "BatchTimer",
    "timer",
    "Schedule",
    "ScheduleStep",
    "ScheduleExp"
]


class BatchTimer:
    """
    A global batch timer.
    """

    def __init__(self):
        self.batch_id = 0
        self.batches_in_epoch = 1  # will be set later on
        self.n_epochs = None  # user-defined no. of epochs to run

    def init(self, batches_in_epoch):
        """
        Initialize the timer by providing the epoch length.

        Parameters
        ----------
        batches_in_epoch : int
            The number of batches in an epoch.

        """
        self.batches_in_epoch = batches_in_epoch

    @property
    def epoch(self):
        """
        Returns
        -------
        int
            Epoch id.
        """
        return int(self.epoch_progress())

    def epoch_progress(self):
        """
        Returns
        -------
        float
            Epoch progress.
        """
        return self.batch_id / self.batches_in_epoch

    def is_epoch_finished(self):
        """
        Returns
        -------
        bool
            Whether it's the end of an epoch (True) or in the middle of
            training (False).
        """
        return self.batch_id > 0 and self.batch_id % self.batches_in_epoch == 0

    def tick(self):
        """
        Increments the number of elapsed batches by 1.
        """
        self.batch_id += 1

    def set_epoch(self, epoch):
        """
        Manually set the epoch.

        Parameters
        ----------
        epoch : int
            A new epoch.

        """
        self.batch_id = self.batches_in_epoch * epoch


timer = BatchTimer()


class Schedule(ABC):
    """
    Schedule the next update program.
    """

    def __init__(self):
        self.last_batch_update = -1

    @abstractmethod
    def next_batch_update(self):
        """
        Returns
        -------
        int
            The next batch id when an update is needed.
        """
        return 0

    def __call__(self, func: Callable):
        @wraps(func)
        def wrapped(*args, **kwargs):
            if self.last_batch_update == -1:
                # restore the last trained batch
                self.last_batch_update = timer.batch_id - 1
            if timer.batch_id >= self.next_batch_update():
                self.last_batch_update = timer.batch_id
                func(*args, **kwargs)

        return wrapped


class ScheduleStep(Schedule):
    """
    Performs an update each ``epoch_step * timer.epoch_size + batch_step``
    batches.

    Parameters
    ----------
    epoch_step : int, optional
        Each epoch step.
        Default: 1
    batch_step : int, optional
        Each batch step.
        Default: 0
    """
    def __init__(self, epoch_step=1, batch_step=0):
        super().__init__()
        self.epoch_step = epoch_step
        self.batch_step = batch_step

    def next_batch_update(self):
        # timer.batches_in_epoch is updated in run-time
        dt = timer.batches_in_epoch * self.epoch_step + self.batch_step
        return self.last_batch_update + dt


class ScheduleExp(Schedule):
    """
    Schedule updates at batches that are powers of two: 1, 2, 4, 8, 16, ...
    Handy for the first epoch.
    """

    def next_batch_update(self):
        if self.last_batch_update > 0:
            next_power = math.floor(math.log2(self.last_batch_update)) + 1
        else:
            next_power = 0
        return 2 ** next_power
