import numpy as np
import tensorflow as tf
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
            for key in self._keys:
                tf.summary.scalar(key, data=np.mean(self._metrics[key]), step=step)
                self._metrics[key] = []
            tf.summary.flush()
