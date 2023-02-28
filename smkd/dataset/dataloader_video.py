import os
import cv2
import sys
import pdb
import six
import glob
import time
import torch
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import pyarrow as pa
from PIL import Image
import torch.utils.data as data
from utils import video_augmentation

sys.path.append("..")


class BaseFeeder(data.Dataset):
    def __init__(self, prefix, gloss_dict, drop_ratio=1, num_gloss=-1, mode="train", transform_mode=True,
                 datatype="lmdb"):
        self.mode = mode
        self.ng = num_gloss
        self.prefix = prefix
        self.dict = gloss_dict
        self.data_type = datatype
        self.transform_mode = "train" if transform_mode else "test"
        self.inputs_list = np.load(f"{prefix}/{mode}_info.npy", allow_pickle=True).item()
        print(mode, len(self))
        self.data_aug = self.transform()
        print("")

    def __getitem__(self, idx):
        if self.data_type == "video":
            input_data, label, fi = self.read_video(idx)
            input_data, label = self.normalize(input_data, label)
            # input_data, label = self.normalize(input_data, label, fi['fileid'])
            return input_data, torch.LongTensor(label), self.inputs_list[idx]['original_info']
        else:
            raise ValueError(f"Not supported: {self.data_type}")

    def read_video(self, index, num_glosses=-1):
        # load file info
        fi = self.inputs_list[index]
        img_folder = fi['folder']
        if self.inputs_list['prefix'] is not None:
            img_folder = os.path.join(self.inputs_list['prefix'], img_folder)

        suffix = img_folder[-3:].lower()
        img_features = []
        label_list = []

        def _resize_image(x):
            x = cv2.resize(x, (256, 256), interpolation=cv2.INTER_LANCZOS4)
            x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
            return x

        if suffix in ("jpg", "png"):
            img_list = sorted(glob.glob(img_folder))
            img_features = [_resize_image(cv2.imread(img_path)) for img_path in img_list]
        else:   # mp4
            vidcap = cv2.VideoCapture(img_folder)
            success, img = vidcap.read()
            while success:
                # process the image
                img = _resize_image(img)
                img_features.append(img)

                # load next frame
                success, img = vidcap.read()

        for phase in fi['label'].split(" "):
            if phase == '':
                continue
            if phase in self.dict.keys():
                label_list.append(self.dict[phase][0])

        if len(img_features) == 0:
            img_features = [np.zeros((256, 256, 3))]
        if len(label_list) == 0:
            label_list = [0]

        return img_features, label_list, fi

    def normalize(self, video, label, file_id=None):
        video, label = self.data_aug(video, label, file_id)
        video = video.float() / 127.5 - 1
        return video, label

    def transform(self):
        if self.transform_mode == "train":
            print("Apply training transform.")
            return video_augmentation.Compose([
                video_augmentation.RandomCrop(224),
                video_augmentation.RandomHorizontalFlip(0.5),
                video_augmentation.ToTensor(),
                video_augmentation.TemporalRescale(0.2),
            ])
        else:
            print("Apply testing transform.")
            return video_augmentation.Compose([
                video_augmentation.CenterCrop(224),
                video_augmentation.ToTensor(),
            ])

    def byte_to_img(self, byteflow):
        unpacked = pa.deserialize(byteflow)
        imgbuf = unpacked[0]
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')
        return img

    @staticmethod
    def collate_fn(batch):
        batch = [item for item in sorted(batch, key=lambda x: len(x[0]), reverse=True)]
        video, label, info = list(zip(*batch))
        if len(video[0].shape) > 3:
            max_len = len(video[0])
            video_length = torch.LongTensor([np.ceil(len(vid) / 4.0) * 4 + 12 for vid in video])
            left_pad = 6
            right_pad = int(np.ceil(max_len / 4.0)) * 4 - max_len + 6
            max_len = max_len + left_pad + right_pad
            padded_video = [torch.cat(
                (
                    vid[0][None].expand(left_pad, -1, -1, -1),
                    vid,
                    vid[-1][None].expand(max_len - len(vid) - left_pad, -1, -1, -1),
                )
                , dim=0)
                for vid in video]
            padded_video = torch.stack(padded_video)
        else:
            max_len = len(video[0])
            video_length = torch.LongTensor([len(vid) for vid in video])
            padded_video = [torch.cat(
                (
                    vid,
                    vid[-1][None].expand(max_len - len(vid), -1),
                )
                , dim=0)
                for vid in video]
            padded_video = torch.stack(padded_video).permute(0, 2, 1)
        label_length = torch.LongTensor([len(lab) for lab in label])
        if max(label_length) == 0:
            return padded_video, video_length, [], [], info
        else:
            padded_label = []
            for lab in label:
                padded_label.extend(lab)
            padded_label = torch.LongTensor(padded_label)
            return padded_video, video_length, padded_label, label_length, info

    def __len__(self):
        return len(self.inputs_list) - 1

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time


if __name__ == "__main__":
    feeder = BaseFeeder()
    dataloader = torch.utils.data.DataLoader(
        dataset=feeder,
        batch_size=1,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    for data in dataloader:
        pdb.set_trace()
