from .trainer import Trainer


class Test(Trainer):

    def train_epoch(self, epoch):
        self.timer.batch_id += self.timer.batches_in_epoch
