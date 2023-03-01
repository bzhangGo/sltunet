#! /bin/bash

export CUDA_VISIBLE_DEVICES=0


data=preprocessed-corpus/
feature=smkd-sign-features/

python3 run.py --mode test --parameters=\
max_len=256,max_img_len=512,eval_batch_size=32,\
beam_size=8,remove_bpe=True,decode_alpha=1.0,\
gpus=[0],\
eval_task='sign2text',\
src_codes="$data/ende.bpe",tgt_codes="$data/ende.bpe",\
src_vocab_file="$data/vocab.zero.drop",\
tgt_vocab_file="$data/vocab.zero.drop",\
output_dir="avg",\
test_output="trans.txt",\
img_test_file="$feature/test.h5",\
src_test_file="$data/test.bpe.en",\
tgt_test_file="$data/test.bpe.de",\

