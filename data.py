# coding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import h5py
import numpy as np
import random
from tensorflow.keras.preprocessing import image
from utils.util import batch_indexer, token_indexer


class Dataset(object):
    def __init__(self, params, img_file, src_file, tgt_file,
                 src_vocab, tgt_vocab, max_len=100, max_img_len=512,
                 batch_or_token='batch',
                 data_leak_ratio=0.5, target_size=None, f=''):
        self.source = src_file
        self.target = tgt_file
        self.image = img_file
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len
        self.max_img_len = max_img_len
        self.batch_or_token = batch_or_token
        self.data_leak_ratio = data_leak_ratio
        self.target_size = target_size
        self.f = f
        self.p = params

        self.leak_buffer = []

        self.img_reader = h5py.File(self.image, 'r')

    def load_data(self, is_training=False):
        with open(self.source, 'r') as src_reader, \
                open(self.target, 'r') as tgt_reader: \

            while True:
                src_line = src_reader.readline()
                tgt_line = tgt_reader.readline()

                if src_line == "" or tgt_line == "":
                    break

                src_line = src_line.strip()
                tgt_line = tgt_line.strip()

                if src_line == "" or tgt_line == "":
                    continue

                src_line_tokens = src_line.strip().split()
                img_index = int(src_line_tokens[0])

                src_line = ' '.join(src_line_tokens[1:])
                # apply stochastic BPE dropout
                if is_training and random.random() > 0.4:
                    src_line = src_line.strip().replace('@@ ', '')
                    tgt_line = tgt_line.strip().replace('@@ ', '')

                    # apply dropout
                    aug=False
                    if '<aug>' in src_line:
                        aug=True
                        src_line = ' '.join(src_line.strip().split()[1:])

                    src_line = self.p.src_bpe.process_line(src_line, dropout=self.p.src_bpe_dropout)
                    tgt_line = self.p.tgt_bpe.process_line(tgt_line, dropout=self.p.tgt_bpe_dropout)

                    if aug:
                        src_line = '<aug> ' + src_line

                yield (
                    self.src_vocab.to_id(src_line.strip().split()[:self.max_len]),
                    self.tgt_vocab.to_id(tgt_line.strip().split()[:self.max_len]),
                    img_index,
                )

    def get_reader(self, is_train=False):
        # We randomly crop and flip images to get 11 duplicated features
        # during training, we randomly sample one feature to simulate data augmentation for sign videos
        N = 11 if is_train else 1
        return random.randint(0, N-1)

    def to_matrix(self, batch, is_train=False):
        batch_size = len(batch)

        src_lens = [len(sample[1]) for sample in batch]
        tgt_lens = [len(sample[2]) for sample in batch]

        src_len = min(self.max_len, max(src_lens))
        tgt_len = min(self.max_len, max(tgt_lens))

        s = np.zeros([batch_size, src_len], dtype=np.int32)
        t = np.zeros([batch_size, tgt_len], dtype=np.int32)
        x = []
        for eidx, sample in enumerate(batch):
            x.append(sample[0])
            src_ids, tgt_ids = sample[1], sample[2]

            s[eidx, :min(src_len, len(src_ids))] = src_ids[:src_len]
            t[eidx, :min(tgt_len, len(tgt_ids))] = tgt_ids[:tgt_len]

        images_path = [sample[3] for sample in batch]
        images = []
        img_idx = []
        dummy = np.zeros([1, self.target_size], dtype=np.float32)
        for image_path in images_path:
            if image_path < 0:
                img_idx.append(0.0)
                images.append(dummy)
                continue
            else:
                img_idx.append(1.0)


            i = self.get_reader(is_train)
            new_image = self.img_reader["%s_%s_%s" % (image_path, self.f, i) if is_train else "%s_%s" % (image_path, self.f)][()]
            images.append(new_image)

        img_lens = [len(img) for img in images]
        img_len = min(max(img_lens), self.max_img_len)
        m = np.zeros([batch_size, img_len, self.target_size], dtype=np.float32)
        mask = np.zeros([batch_size, img_len], dtype=np.float32)
        img_idx = np.asarray(img_idx, dtype=np.float32)

        for eidx, img in enumerate(images):
            m[eidx, :min(img_len, len(img))] = img[:img_len]
            mask[eidx,  :min(img_len, len(img))] = 1.0

        # construct sparse label sequence, for ctc training
        seq_indexes = []
        seq_values = []
        for n, sample in enumerate(batch):
            sequence = sample[1][:src_len]

            seq_indexes.extend(zip([n] * len(sequence), range(len(sequence))))
            seq_values.extend(sequence)

        seq_indexes = np.asarray(seq_indexes, dtype=np.int64)
        seq_values = np.asarray(seq_values, dtype=np.int32)
        seq_shape = np.asarray([batch_size, tgt_len], dtype=np.int64)

        return x, s, t, m, mask, (seq_indexes, seq_values, seq_shape), img_idx

    def batcher(self, size, buffer_size=1000, shuffle=True, train=True):
        def _handle_buffer(_buffer):
            sorted_buffer = sorted(
                _buffer, key=lambda xx: max(len(xx[1]), len(xx[2])))

            if self.batch_or_token == 'batch':
                buffer_index = batch_indexer(len(sorted_buffer), size)
            else:
                buffer_index = token_indexer(
                    [[len(sample[1]), len(sample[2])] for sample in sorted_buffer], size)

            index_over_index = batch_indexer(len(buffer_index), 1)
            if shuffle: np.random.shuffle(index_over_index)

            for ioi in index_over_index:
                index = buffer_index[ioi[0]]
                batch = [sorted_buffer[ii] for ii in index]
                x, s, t, m, mask, spar, img_idx = self.to_matrix(batch, train)
                yield {
                    'src': s,
                    'tgt': t,
                    'img': m,
                    'is_img': img_idx,
                    'mask': mask,
                    'spar': spar,
                    'index': x,
                    'raw': batch,
                }

        buffer = self.leak_buffer
        self.leak_buffer = []
        for i, (src_ids, tgt_ids, img_path) in enumerate(self.load_data(train)):
            buffer.append((i, src_ids, tgt_ids, img_path))
            if len(buffer) >= buffer_size:
                for data in _handle_buffer(buffer):
                    # check whether the data is tailed
                    batch_size = len(data['raw']) if self.batch_or_token == 'batch' \
                        else max(np.sum(data['tgt'] > 0), np.sum(data['src'] > 0))
                    if batch_size < size * self.data_leak_ratio:
                        self.leak_buffer += data['raw']
                    else:
                        yield data
                buffer = self.leak_buffer
                self.leak_buffer = []

        # deal with data in the buffer
        if len(buffer) > 0:
            for data in _handle_buffer(buffer):
                # check whether the data is tailed
                batch_size = len(data['raw']) if self.batch_or_token == 'batch' \
                    else max(np.sum(data['tgt'] > 0), np.sum(data['src'] > 0))
                if train and batch_size < size * self.data_leak_ratio:
                    self.leak_buffer += data['raw']
                else:
                    yield data
