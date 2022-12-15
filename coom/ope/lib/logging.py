import numpy as np
import wandb
from chainer.training import extension


class SummaryReport(extension.Extension):

    def __init__(self, keys, interval=100):
        self._keys = keys
        self._interval = interval
        self._metrics = {}
        for key in keys:
            self._metrics[key] = []

    def __call__(self, trainer):
        for key in self._keys:
            self._metrics[key].append(trainer.observation[key].item())
        step = trainer.updater.iteration
        if step % self._interval == 0:
            wandb.log({'step': step})
            for key in self._keys:
                wandb.log({key: np.mean(self._metrics[key])})
                self._metrics[key] = []
