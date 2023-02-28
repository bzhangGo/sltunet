#! /bin/bash

export CUDA_VISIBLE_DEVICES=0

data=preprocessed-corpus/
feature=smkd-sign-features/

python3 run.py --mode train --parameters=\
hidden_size=256,embed_size=256,filter_size=4096,\
sep_layer=0,num_encoder_layer=6,num_decoder_layer=6,\
ctc_enable=True,ctc_alpha=0.3,ctc_repeated=True,\
src_bpe_dropout=0.2,tgt_bpe_dropout=0.2,bpe_dropout_stochastic_rate=0.6,\
initializer="uniform_unit_scaling",initializer_gain=0.5,\
dropout=0.3,label_smooth=0.1,attention_dropout=0.3,relu_dropout=0.5,residual_dropout=0.4,\
max_len=256,max_img_len=512,batch_size=80,eval_batch_size=32,\
token_size=1600,batch_or_token='token',beam_size=8,remove_bpe=True,decode_alpha=1.0,\
scope_name="transformer",buffer_size=50000,data_leak_ratio=0.1,\
img_feature_size=1024,img_aug_size=11,\
clip_grad_norm=0.0,\
num_heads=4,\
process_num=2,\
lrate=1.0,\
estop_patience=100,\
warmup_steps=4000,\
epoches=5000,\
update_cycle=16,\
gpus=[0],\
disp_freq=1,\
eval_freq=500,\
sample_freq=100,\
checkpoints=5,\
best_checkpoints=10,\
max_training_steps=30000,\
nthreads=8,\
beta1=0.9,\
beta2=0.998,\
random_seed=1234,\
src_codes="$data/ende.bpe",tgt_codes="$data/ende.bpe",\
src_vocab_file="$data/vocab.zero.drop",\
tgt_vocab_file="$data/vocab.zero.drop",\
img_train_file="$feature/train.h5",\
src_train_file="$data/train.bpe.en.shuf",\
tgt_train_file="$data/train.bpe.de.shuf",\
img_dev_file="$feature/dev.h5",\
src_dev_file="$data/dev.bpe.en",\
tgt_dev_file="$data/dev.bpe.de",\
img_test_file="$feature/test.h5",\
src_test_file="$data/test.bpe.en",\
tgt_test_file="$data/test.bpe.de",\
output_dir="train",\
test_output="",\
shared_source_target_embedding=True,\
