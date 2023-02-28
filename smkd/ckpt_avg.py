# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob
import argparse
import logging
import collections
from shutil import copyfile

import torch


def parseargs():
    msg = "Average checkpoints"
    usage = "ckpt_avg.py [<args>] [-h | --help]"
    parser = argparse.ArgumentParser(description=msg, usage=usage)

    parser.add_argument("--path", type=str, required=True,
                        help="checkpoint dir")
    parser.add_argument("--checkpoints", type=int, required=True,
                        help="number of checkpoints to use")
    parser.add_argument("--output", type=str, help="output path")
    parser.add_argument("--gpu", type=int, default=0,
                        help="the default gpu device index")

    return parser.parse_args()


def get_checkpoints(path):
    if not os.path.exists(os.path.join(path, "checkpoint")):
        raise ValueError("Cannot find checkpoints in %s" % path)

    checkpoint_names = []

    with open(os.path.join(path, "checkpoint")) as fd:
        for line in fd:
            name = line.strip()
            checkpoint_names.append(os.path.join(path, name))

    return checkpoint_names[::-1]


def checkpoint_exists(path):
    return os.path.exists(path)


def main(FLAGS):
    logging.basicConfig(level=logging.INFO)
    checkpoints = get_checkpoints(FLAGS.path)
    checkpoints = checkpoints[:FLAGS.checkpoints]

    if not checkpoints:
        raise ValueError("No checkpoints provided for averaging.")

    checkpoints = [c for c in checkpoints if checkpoint_exists(c)]

    if not checkpoints:
        raise ValueError(
            "None of the provided checkpoints exist. %s" % FLAGS.checkpoints
        )

    device = torch.device("cpu" if FLAGS.gpu < 0 else "cuda:{}".format(FLAGS.gpu))

    var_base = torch.load(checkpoints[0], map_location=device)
    logging.info("Read from checkpoint %s", checkpoints[0])
    state_key = 'model_state_dict'

    # we have to construct a purely new variable dictionary
    # the fucking parameter sharing way is quite stupid!
    var_dict = collections.OrderedDict()
    for name in var_base[state_key]:
        var_dict[name] = var_base[state_key][name].clone()

    for checkpoint in checkpoints[1:]:
        reader = torch.load(checkpoint, map_location=device)
        for name in var_dict:
            var_dict[name].add_(reader[state_key][name])
        logging.info("Read from checkpoint %s", checkpoint)

    # Average checkpoints
    for name in var_dict:
        if var_dict[name].is_floating_point():
            var_dict[name].div_(len(checkpoints))
        else:
            var_dict[name] //= len(checkpoints)

    # Shift back into var_base
    var_base[state_key] = var_dict

    if not os.path.exists(FLAGS.output):
        os.mkdir(FLAGS.output)

    saved_name = os.path.join(FLAGS.output, "average.pt")
    torch.save(var_base, saved_name)
    with open(os.path.join(FLAGS.output, 'checkpoint'), 'w') as writer:
        writer.write("average.pt\n")

    logging.info("Averaged checkpoints saved in %s", saved_name)

    params_pattern = os.path.join(FLAGS.path, "*.json")
    params_files = glob.glob(params_pattern)

    for name in params_files:
        new_name = name.replace(FLAGS.path.rstrip("/"),
                                FLAGS.output.rstrip("/"))
        copyfile(name, new_name)


if __name__ == "__main__":
    main(parseargs())

