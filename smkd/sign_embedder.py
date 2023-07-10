
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import yaml
import torch
import importlib
import faulthandler
import numpy as np
import torch.nn as nn
from collections import OrderedDict
from tqdm import tqdm

faulthandler.enable()
try:
    import utils
    from modules.sync_batchnorm import convert_model
    from seq_scripts import seq_train, seq_eval, seq_feature_generation
except:
    from . import utils
    from .modules.sync_batchnorm import convert_model
    from .seq_scripts import seq_train, seq_eval, seq_feature_generation


class SignEmbedding():
    def __init__(self, cfg, gloss_path, sign_video_path, model_path, gpu_id, batch_size):
        sparser = utils.get_parser()
        # disable parsing command line
        import sys
        sys.argv = sys.argv[:1]
        p = sparser.parse_args()
        with open(cfg, 'r') as f:
            try:
                default_arg = yaml.load(f, Loader=yaml.FullLoader)
            except AttributeError:
                default_arg = yaml.load(f)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        sparser.set_defaults(**default_arg)
        args = sparser.parse_args()
        args.test_batch_size = batch_size
 
        self.arg = args
        self.device = utils.GpuDataParallel()
        self.dataset = {}
        self.data_loader = {}
        self.sign_video_path = sign_video_path
        self.model_path = model_path
        self.gpu_id = str(gpu_id)
        self.gloss_dict = np.load(gloss_path, allow_pickle=True).item()
        self.arg.model_args['num_classes'] = len(self.gloss_dict) + 1
        self.model = self.loading()

    def embed(self):
        self.model.eval()

        features = {}
        for batch_idx, data in tqdm(enumerate(self.data_loader["infer"])):
            vid = self.device.data_to_device(data[0])
            vid_lgt = self.device.data_to_device(data[1])
            with torch.no_grad():
                ret_dict = self.model(vid, vid_lgt)
    
            feat_len = ret_dict['feat_len'].cpu().detach().numpy().astype(np.int32)
            visual_features = ret_dict['visual_features'].permute(1, 0, 2)
            for sample_idx in range(len(vid)):
                visual_feature = visual_features[sample_idx][:feat_len[sample_idx]].cpu().detach().numpy().astype(np.float32)
                features[data[-1][sample_idx][1]] = visual_feature
    
        embeddings = {}
        fkeys = sorted(list(features.keys()))
        for i, dkey in enumerate(fkeys):
            feature = features[dkey]
            embeddings["%s" % i] = feature

        return embeddings

    def loading(self):
        self.device.set_device(self.gpu_id)

        print("Loading model")
        model_class = import_class(self.arg.model)
        model = model_class(
            **self.arg.model_args,
            gloss_dict=self.gloss_dict,
            loss_weights=self.arg.loss_weights,
        )
        self.load_model_weights(model, self.model_path)
        model = self.model_to_device(model)
        print("Loading model finished.")

        self.load_data()
        return model

    def model_to_device(self, model):
        model = model.to(self.device.output_device)
        if len(self.device.gpu_list) > 1:
            model.conv2d = nn.DataParallel(
                model.conv2d,
                device_ids=self.device.gpu_list,
                output_device=self.device.output_device)
        model = convert_model(model)
        model.cuda()
        return model

    def load_model_weights(self, model, weight_path):
        state_dict = torch.load(weight_path)
        if len(self.arg.ignore_weights):
            for w in self.arg.ignore_weights:
                if state_dict.pop(w, None) is not None:
                    print('Successfully Remove Weights: {}.'.format(w))
                else:
                    print('Can Not Remove Weights: {}.'.format(w))
        weights = self.modified_weights(state_dict['model_state_dict'], False)
        model.load_state_dict(weights, strict=True)

    @staticmethod
    def modified_weights(state_dict, modified=False):
        state_dict = OrderedDict([(k.replace('.module', ''), v) for k, v in state_dict.items()])
        if not modified:
            return state_dict
        modified_dict = dict()
        return modified_dict

    def load_data(self):
        print("Loading data")
        self.feeder = import_class("dataset.dataloader_video.InferFeeder")
        dataset_list = zip(["infer"], [False])
        for idx, (mode, train_flag) in enumerate(dataset_list):
            arg = self.arg.feeder_args
            arg["mode"] = mode.split("_")[0]
            arg["prefix"] = self.sign_video_path
            arg["transform_mode"] = train_flag
            self.dataset[mode] = self.feeder(gloss_dict=self.gloss_dict, **arg)
            self.data_loader[mode] = self.build_dataloader(self.dataset[mode], mode, train_flag)
        print("Loading data finished.")

    def build_dataloader(self, dataset, mode, train_flag):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.arg.batch_size if mode == "train" else self.arg.test_batch_size,
            shuffle=train_flag if self.arg.phase != "features" else False,
            drop_last=train_flag if self.arg.phase != "features" else False,
            num_workers=self.arg.num_worker,  # if train_flag else 0
            collate_fn=self.feeder.collate_fn,
        )


def import_class(name):
    components = name.rsplit('.', 1)
    try:
        mod = importlib.import_module(components[0])
    except:
        mod = importlib.import_module("smkd."+components[0])
    mod = getattr(mod, components[1])
    return mod

