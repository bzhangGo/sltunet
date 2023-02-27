# coding: utf-8

from lrs import noamlr


def get_lr(params):

    strategy = params.lrate_strategy.lower()

    if strategy == "noam":
        return noamlr.NoamDecayLr(
            params.lrate,
            params.min_lrate,
            params.max_lrate,
            params.warmup_steps,
            params.hidden_size
        )
    else:
        raise NotImplementedError(
            "{} is not supported".format(strategy))
