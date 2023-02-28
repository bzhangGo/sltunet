import re
import os
import cv2
import pdb
import glob
import pandas
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from functools import partial


def csv2dict_phoneix2014(anno_path, dataset_type, extra_info=None):
    anno_path = f"{anno_path}/{extra_info.format(dataset_type)}"
    inputs_list = pandas.read_csv(anno_path)

    inputs_list = (inputs_list.to_dict()['name|video|start|end|speaker|orth|translation'].values())
    info_dict = dict()
    info_dict['prefix'] = anno_path.rsplit("/", 3)[0] + "/features/fullFrame-210x260px"
    print(f"Generate information dict from {anno_path} for {dataset_type}")
    for file_idx, file_info in tqdm(enumerate(inputs_list), total=len(inputs_list)):
        fileid, folder, _, _, signer, label, _ = file_info.split("|")
        num_frames = len(glob.glob(f"{info_dict['prefix']}/{dataset_type}/{folder}"))
        info_dict[file_idx] = {
            'fileid': fileid,
            'folder': f"{dataset_type}/{fileid}/*.png",
            'signer': signer,
            'label': label,
            'num_frames': num_frames,
            'original_info': (file_info, file_idx),
        }
    return info_dict


def csv2dict_csldaily(anno_path, dataset_type, dataset=None, dataset_split=None):

    info_dict = dict()
    info_dict['prefix'] = anno_path + "/sentence/frames_512x512"

    print(f"Generate information dict from {anno_path} for {dataset_type}")
    actual_file_idx = 0
    for file_idx, file_info in tqdm(enumerate(dataset['info']), total=len(dataset['info'])):

        name = file_info['name']
        if name not in dataset_split or dataset_split[name] != dataset_type:
            continue

        num_frames = len(glob.glob(f"{info_dict['prefix']}/{file_info['name']}/*.jpg"))
        assert num_frames == file_info['length']
        info_dict[actual_file_idx] = {
            'fileid': file_info['name'],
            'folder': file_info['name'] + "/*.jpg",
            'signer': file_info['signer'],
            'label': ' '.join(file_info['label_gloss']),
            'num_frames': num_frames,
            'original_info': (file_info, actual_file_idx),
        }

        actual_file_idx += 1

    return info_dict


def csv2dict_dgs(anno_path, dataset_type):
    gloss_list = open(anno_path + "/%s.bpe.gloss" % dataset_type, 'r').readlines()
    img_list = open(anno_path + "/%s.img" % dataset_type, 'r').readlines()

    info_dict = dict()
    info_dict['prefix'] = None

    print(f"Generate information dict from {anno_path} for {dataset_type}")
    for file_idx, file_info in tqdm(enumerate(zip(gloss_list, img_list)), total=len(gloss_list)):
        gloss, img = file_info
        gloss = gloss.strip()
        img = img.strip()

        if gloss == "":
            gloss = "Unknown"

        num_frames = int(cv2.VideoCapture(img).get(cv2.CAP_PROP_FRAME_COUNT))

        info_dict[file_idx] = {
            'fileid': os.path.basename(img),
            'folder': img,
            'signer': 'Unknown',
            'label': gloss,
            'num_frames': num_frames,
            'original_info': (file_info, file_idx),
        }
    return info_dict


def generate_gt_stm(info, save_path):
    with open(save_path, "w") as f:
        for k, v in info.items():
            if not isinstance(k, int):
                continue
            f.writelines(f"{v['fileid']} 1 {v['signer']} 0.0 1.79769e+308 {v['label']}\n")


def sign_dict_update(total_dict, info):
    for k, v in info.items():
        if not isinstance(k, int):
            continue
        split_label = v['label'].split()
        for gloss in split_label:
            if gloss not in total_dict.keys():
                total_dict[gloss] = 1
            else:
                total_dict[gloss] += 1
    return total_dict


def load_csldaily_split(split_file):
    dataset_split = {}
    with open(split_file, 'r') as reader:
        reader.readline()   # skip header

        for sample in reader:
            name, md = sample.strip().split('|')
            dataset_split[name] = md

    return dataset_split


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Data process for Visual Alignment Constraint for Continuous Sign Language Recognition.')
    parser.add_argument('--dataset', type=str, default='phoenix2014', choices=['phoenix2014', 'csldaily', 'dgs3-t'],
                        help='save prefix')
    parser.add_argument('--dataset-root', type=str, default='dataset/phoenix2014',
                        help='path to the dataset')

    args = parser.parse_args()

    csv2dict = None
    if args.dataset == 'phoenix2014':
        csv2dict = partial(csv2dict_phoneix2014,
                           extra_info="annotations/manual/PHOENIX-2014-T.{}.corpus.csv")
    elif args.dataset == 'csldaily':
        # load dataset split: train/dev/test
        dataset_split = load_csldaily_split(f"{args.dataset_root}/sentence_label/split_1.txt")
        # load dataset info
        with open(f"{args.dataset_root}/sentence_label/csl2020ct_v2.pkl", 'rb') as f:
            dataset = pickle.load(f)

        csv2dict = partial(csv2dict_csldaily, dataset=dataset, dataset_split=dataset_split)
    elif args.dataset == 'dgs3-t':
        csv2dict = csv2dict_dgs
    else:
        raise ValueError(f"Invalid dataset {args.dataset}")

    mode = ["dev", "test", "train"]
    sign_dict = dict()
    if not os.path.exists(f"./{args.dataset}"):
        os.makedirs(f"./{args.dataset}")
    for md in mode:
        # generate information dict
        information = csv2dict(f"{args.dataset_root}", dataset_type=md)
        np.save(f"./{args.dataset}/{md}_info.npy", information)
        # update the total gloss dict
        sign_dict_update(sign_dict, information)
        # generate groudtruth stm for evaluation
        generate_gt_stm(information, f"./{args.dataset}/{args.dataset}-groundtruth-{md}.stm")
    sign_dict = sorted(sign_dict.items(), key=lambda d: d[0])
    save_dict = {}
    for idx, (key, value) in enumerate(sign_dict):
        save_dict[key] = [idx + 1, value]
    np.save(f"./{args.dataset}/gloss_dict.npy", save_dict)
