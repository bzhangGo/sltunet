#! /usr/bin/python3

import logging
import os
import argparse
import json
import subprocess
import multiprocessing
import functools

from typing import Tuple, Dict, Union, List, Optional

# ffmpeg -threads 1 -ss 5.0 -i focusnews.113.mp4 -frames:v 500 -c:v libx264 output2.mp4

FFMPEG_TEMPLATE = "{binary_path} -threads {num_threads} -ss {start_time_seconds} -i {input_file} -frames:v {num_frames} " \
                  "-c:v libx264 {output_file}"


def write_video(input_video_filepath: str,
                output_video_filepath: str,
                start_frame: int,
                end_frame: int,
                fps: int,
                multithreaded: bool,
                custom_path: Optional[str]) -> Tuple[str, str, str]:
    """

    :param input_video_filepath:
    :param output_video_filepath:
    :param start_frame:
    :param end_frame:
    :param fps:
    :param multithreaded:
    :param custom_path:
    :return:
    """

    if os.path.exists(output_video_filepath):
        logging.debug("Output video file exists, will skip calling ffmpeg: '%s'" % output_video_filepath)
        return output_video_filepath, "", ""

    logging.debug("Writing file: %s" % output_video_filepath)

    num_frames = end_frame - start_frame

    start_time_seconds = start_frame / fps

    if multithreaded:
        num_threads = "auto"
    else:
        num_threads = "1"

    if custom_path is None:
        binary_path = "ffmpeg"
    else:
        binary_path = custom_path

    ffmpeg_cmd = FFMPEG_TEMPLATE.format(binary_path=binary_path,
                                        input_file=input_video_filepath,
                                        output_file=output_video_filepath,
                                        start_time_seconds=start_time_seconds,
                                        num_frames=num_frames,
                                        num_threads=num_threads)

    logging.debug("Executing:")
    logging.debug(ffmpeg_cmd)

    result = subprocess.run(ffmpeg_cmd, shell=True, capture_output=True)

    return output_video_filepath, result.stdout.decode("utf-8"), result.stderr.decode("utf-8")


VideoInfo = Dict[str, Union[str, int]]
Example = Dict[str, Union[str, VideoInfo]]


def load_examples_json(filepath: str) -> Dict[str, List[Example]]:
    """
    Structure of entries:

    {
    "train": [
        {
            "split_name": "train",
            "file_id": "1289623",
            "sentence_id: 00005,
            "gloss_sequence": "TAUB-GEH\u00d6RLOS1 THEATER6 ERSTENS1 ICH1 DEUTSCH1 TAUB-GEH\u00d6RLOS1 THEATER6 IN1 DEUTSCH1 BEREICH1 FAHREN1 ORT1 DEUTSCH1 BEREICH1",
            "german_sentence": "Ich war viel mit dem Theater unterwegs, mit dem Deutschen Geh\u00f6rlosentheater. Wir waren \u00fcberall an verschiedensten Standorten in Deutschland.",
            "video_info": {
                "filepath": "tfds_datasets_custom/downloads/sign-lang.uni-hamb.de_mein_vide_1289_1289_jXgoTewAU6iV5g9bZxqgw9hscUECqNQhNibuWsbcBTY.mp4",
                "fps": 50,
                "start_frame": 0,
                "end_frame": 286
            }
        }, ...
        ]
        "validation": [ ... ],
        "test": [ ... ],
    }

    :param filepath:
    :return:
    """
    logging.debug("Loading generated examples from: '%s'" % filepath)

    with open(filepath, "r") as infile:
        return json.load(infile)


def write_video_filepaths(examples: List[Example], split_name: str, output_folder: str):
    """

    :param examples:
    :param split_name:
    :param output_folder:
    :return:
    """
    output_filename = ".".join([split_name, "video_paths"])
    output_filepath = os.path.join(output_folder, output_filename)

    logging.debug("Writing: '%s'" % output_filepath)

    with open(output_filepath, "w") as handle:

        for example in examples:

            output_video_filepath = get_video_filepath(example, output_folder)

            handle.write(output_video_filepath + "\n")


def write_gloss_or_text(examples: List[Example], split_name: str, key: str, output_folder: str):
    """

    :param examples:
    :param split_name:
    :param key:
    :param output_folder:
    :return:
    """

    if key == "gloss_sequence":
        suffix = "gloss"
    elif key == "german_sentence":
        suffix = "de"
    else:
        raise ValueError("Do not understand key: %s" % key)

    output_filename = ".".join([split_name, suffix])
    output_filepath = os.path.join(output_folder, output_filename)

    logging.debug("Writing: '%s'" % output_filepath)

    with open(output_filepath, "w") as handle:

        for example in examples:

            sequence = example[key]

            handle.write(sequence + "\n")


def write_strings(examples: List[Example], split_name: str, key: str, output_folder: str):
    """

    :param examples:
    :param split_name:
    :param key:
    :param output_folder:
    :return:
    """

    if key == "video":
        write_video_filepaths(examples=examples, split_name=split_name, output_folder=output_folder)
    else:
        write_gloss_or_text(examples=examples, split_name=split_name, key=key, output_folder=output_folder)


def get_video_filepath(example: Example, output_folder: str) -> str:
    """

    :param example:
    :param output_folder:
    :return:
    """
    split_name = example["split_name"]
    file_id = example["file_id"]
    sentence_id = str(example["sentence_id"])

    output_video_filename = ".".join([split_name, file_id, sentence_id, "mp4"])
    output_video_filepath = os.path.join(output_folder, output_video_filename)

    output_video_filepath = os.path.realpath(output_video_filepath)

    return output_video_filepath


def process_video(example: Example, output_folder: str, ffmpeg_multithreaded: bool, ffmpeg_custom_path: Optional[str]):
    """

    :param example: Structure:
        {
            "split_name": "train",
            "file_id": "1289623",
            "sentence_id: 00005,
            "gloss_sequence": "TAUB-GEH\u00d6RLOS1 THEATER6 ERSTENS1 ICH1 DEUTSCH1 TAUB-GEH\u00d6RLOS1 THEATER6 IN1 DEUTSCH1 BEREICH1 FAHREN1 ORT1 DEUTSCH1 BEREICH1",
            "german_sentence": "Ich war viel mit dem Theater unterwegs, mit dem Deutschen Geh\u00f6rlosentheater. Wir waren \u00fcberall an verschiedensten Standorten in Deutschland.",
            "video_info": {
                "filepath": "tfds_datasets_custom/downloads/sign-lang.uni-hamb.de_mein_vide_1289_1289_jXgoTewAU6iV5g9bZxqgw9hscUECqNQhNibuWsbcBTY.mp4",
                "fps": 50,
                "start_frame": 0,
                "end_frame": 286
            }
    :param output_folder:
    :param ffmpeg_multithreaded:
    :param ffmpeg_custom_path:
    :return:
    """
    output_video_filepath = get_video_filepath(example, output_folder)

    video_info = example["video_info"]

    input_video_filepath = video_info["filepath"]
    start_frame = video_info["start_frame"]
    end_frame = video_info["end_frame"]
    fps = video_info["fps"]

    write_video(input_video_filepath=input_video_filepath,
                output_video_filepath=output_video_filepath,
                start_frame=start_frame,
                end_frame=end_frame,
                fps=fps,
                multithreaded=ffmpeg_multithreaded,
                custom_path=ffmpeg_custom_path)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", type=str,
                        help="File where JSON inputs are stored.", required=True)
    parser.add_argument("--output-folder", type=str,
                        help="Folder where individual output files are stored.", required=True)

    parser.add_argument("--num-workers", type=int, default=1,
                        help="Number of multiprocessing processed for video cutting.", required=False)

    parser.add_argument("--ffmpeg-multithreaded", action="store_true",
                        help="Use all threads for ffmpeg.", required=False)
    parser.add_argument("--ffmpeg-custom-path", type=str, default=None,
                        help="Point to custom ffmpeg binary.", required=False)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    logging.basicConfig(level=logging.DEBUG)
    logging.debug(args)

    video_subfolder = os.path.join(args.output_folder, "videos")
    os.makedirs(video_subfolder, exist_ok=True)

    examples = load_examples_json(args.input)

    pool = multiprocessing.Pool(processes=args.num_workers)

    for split_name in ["train", "validation", "test"]:

        examples_subset = examples[split_name]

        # process videos

        process_partial = functools.partial(process_video,
                                            output_folder=video_subfolder,
                                            ffmpeg_multithreaded=args.ffmpeg_multithreaded,
                                            ffmpeg_custom_path=args.ffmpeg_custom_path)

        # call list as finalizer for imap pool

        list(pool.imap_unordered(process_partial, examples_subset))

        # process strings

        write_strings(examples=examples_subset, key="video",
                      split_name=split_name, output_folder=args.output_folder)

        write_strings(examples=examples_subset, key="gloss_sequence",
                      split_name=split_name, output_folder=args.output_folder)

        write_strings(examples=examples_subset, key="german_sentence",
                      split_name=split_name, output_folder=args.output_folder)


if __name__ == '__main__':
    main()
