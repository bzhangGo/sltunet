#! /usr/bin/python3
import os.path
import re
import logging
import argparse
import json

import tensorflow as tf
import tensorflow_datasets as tfds
# noinspection PyUnresolvedReferences
import sign_language_datasets.datasets
from sign_language_datasets.datasets.config import SignDatasetConfig
from sign_language_datasets.datasets.dgs_corpus.dgs_utils import get_elan_sentences

from typing import Iterator, Optional, Dict, Union, List

GLOSSES_TO_IGNORE = ["$GEST-OFF", "$$EXTRA-LING-MAN"]


def collapse_gloss(gloss: str) -> str:
    """
    Collapse phonological variations of the same type, and
    - for number signs remove handshape variants
    - keep numerals ($NUM), list glosses ($LIST) and finger alphabet ($ALPHA)
    :param gloss:
    :return:
    """
    try:
        collapsed_gloss_groups = re.search(r"([$A-Z-ÖÄÜ]+[0-9]*)[A-Z]*(:?[0-9A-ZÖÄÜ]*o?f?[0-9]*)", gloss).groups()
        collapsed_gloss = "".join([g for g in collapsed_gloss_groups if g is not None])
    except AttributeError:
        print("Gloss could not be generalized: '%s'", gloss)
        collapsed_gloss = gloss

    return collapsed_gloss


def generalize_dgs_glosses(line: str) -> str:
    """
    This code is taken from:
    https://github.com/bricksdont/easier-gloss-translation/blob/gloss_preprocessing_2/scripts/preprocessing/preprocess_glosses.py

    Removes certain kinds of variation in order to bolster generalization.
    Example:
    ICH1 ETWAS-PLANEN-UND-UMSETZEN1 SELBST1A* KLAPPT1* $GEST-OFF^ BIS-JETZT1 GEWOHNHEIT1* $GEST-OFF^*
    becomes:
    ICH1 ETWAS-PLANEN-UND-UMSETZEN1 SELBST1 KLAPPT1 BIS-JETZT1 GEWOHNHEIT1
    :param line:
    :return:
    """
    # remove ad-hoc deviations from citation forms
    line = line.replace("*", "")

    # remove distinction between type glosses and subtype glosses
    line = line.replace("^", "")

    glosses = line.split(" ")

    collapsed_glosses = []

    for gloss in glosses:
        collapsed_gloss = collapse_gloss(gloss)

        # remove special glosses that cannot possibly help translation
        if collapsed_gloss in GLOSSES_TO_IGNORE:
            continue

        collapsed_glosses.append(collapsed_gloss)

    line = " ".join(collapsed_glosses)

    return line


def load_dataset(data_dir: Optional[str] = None):
    """

    :param data_dir:
    :return:
    """

    config = SignDatasetConfig(name="annotations-include-videos",
                               version="3.0.0",
                               include_video=True,
                               process_video=False,
                               include_pose=None,
                               split="3.0.0-uzh-document")

    dgs_corpus = tfds.load('dgs_corpus', builder_kwargs=dict(config=config), data_dir=data_dir)

    return dgs_corpus


def miliseconds_to_frame_index(ms: int, fps: int) -> int:
    """
    :param ms:
    :param fps:
    :return:
    """
    return int(fps * (ms / 1000))


VideoInfo = Dict[str, Union[str, int]]
Example = Dict[str, Union[str, VideoInfo]]


def generate_examples(dataset: tf.data.Dataset,
                      split_name: str,
                      preprocess_glosses: bool = False) -> Iterator[Example]:
    """

    :param dataset:
    :param split_name: "train", "validation" or "test"
    :param preprocess_glosses:
    :return:
    """

    for datum in dataset[split_name]:

        datum_id = datum["id"].numpy().decode('utf-8')

        elan_path = datum["paths"]["eaf"].numpy().decode('utf-8')
        sentences = get_elan_sentences(elan_path)

        fps = int(datum["fps"].numpy())

        for sentence_id, sentence in enumerate(sentences):

            participant = sentence["participant"].lower()

            video_filepath = datum["paths"]["videos"][participant].numpy().decode('utf-8')
            video_filepath = os.path.realpath(video_filepath)

            glosses = sentence["glosses"]
            gloss_sequence = " ".join([g["gloss"] for g in glosses])

            if gloss_sequence == "":
                continue

            if preprocess_glosses:
                gloss_sequence = generalize_dgs_glosses(gloss_sequence)

            german_sentence = sentence["german"]

            start_frame = miliseconds_to_frame_index(sentence["start"], fps)
            end_frame = miliseconds_to_frame_index(sentence["end"], fps)

            video_info = {"filepath": video_filepath, "fps": fps,
                          "start_frame": start_frame, "end_frame": end_frame}

            example = {"split_name": split_name,
                       "file_id": datum_id,
                       "participant": participant,
                       "sentence_id": str(sentence_id).zfill(5),
                       "gloss_sequence": gloss_sequence,
                       "german_sentence": german_sentence,
                       "video_info": video_info}

            yield example


def write_examples_json(examples: Dict[str, List[Example]], filepath: str):
    """

    :param examples:
    :param filepath:
    :return:
    """

    with open(filepath, "w") as outhandle:
        logging.debug("Writing generated examples to: '%s'" % filepath)
        json.dump(examples, outhandle, indent=4, ensure_ascii=False)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--tfds-data-dir", type=str, default=None,
                        help="TFDS data folder to cache downloads.", required=False)
    parser.add_argument("--output", type=str,
                        help="File where JSON outputs are stored.", required=True)
    parser.add_argument("--preprocess-glosses", action="store_true",
                        help="Specific preprocessing for DGS glosses.", required=False)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    logging.basicConfig(level=logging.DEBUG)
    logging.debug(args)

    dgs_corpus_with_videos = load_dataset(data_dir=args.tfds_data_dir)

    examples = {}

    stats = {}

    for split_name in ["train", "validation", "test"]:
        examples[split_name] = list(generate_examples(dataset=dgs_corpus_with_videos,
                                                      split_name=split_name,
                                                      preprocess_glosses=args.preprocess_glosses))

        stats[split_name] = len(examples[split_name])

    write_examples_json(examples=examples, filepath=args.output)

    logging.debug("Number of examples found:")
    logging.debug(stats)


if __name__ == '__main__':
    main()
