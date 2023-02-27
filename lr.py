# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class Lr(object):
    def __init__(self,
                 init_lrate,        # initial learning rate
                 min_lrate,         # minimum learning rate
                 max_lrate,         # maximum learning rate
                 warmup_steps,      # warmup step
                 hidden_size,       # model hidden size
                 name="noam_lr",    # model name, no use
                 ):
        self.name = name
        self.init_lrate = init_lrate    # just record the init learning rate
        self.lrate = init_lrate         # active learning rate, change with training
        self.min_lrate = min_lrate
        self.max_lrate = max_lrate
        self.warmup_steps = warmup_steps
        self.hidden_size = hidden_size

        assert self.max_lrate > self.min_lrate, \
            "Minimum learning rate should less than maximum learning rate"

    # suppose the eidx starts from 1
    def before_epoch(self, eidx=None):
        pass

    def after_epoch(self, eidx=None):
        pass

    def step(self, step):
        step = float(step)
        warmup_steps = float(self.warmup_steps)

        multiplier = float(self.hidden_size) ** -0.5
        decay = multiplier * np.minimum((step + 1) * (warmup_steps ** -1.5),
                                        (step + 1) ** -0.5)
        self.lrate = self.init_lrate * decay

    def after_eval(self, eval_score):
        pass

    def get_lr(self):
        """Return the learning rate whenever you want"""
        return max(min(self.lrate, self.max_lrate), self.min_lrate)


def get_lr(params):
    return Lr(
        params.lrate,
        params.min_lrate,
        params.max_lrate,
        params.warmup_steps,
        params.hidden_size
    )
