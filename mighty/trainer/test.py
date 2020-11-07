from .trainer import Trainer


__all__ = [
    "Test"
]


class Test(Trainer):
    """
    A Test trainer that only evaluates a model.
    """

    def train_epoch(self, epoch):
        self.timer.batch_id += self.timer.batches_in_epoch
