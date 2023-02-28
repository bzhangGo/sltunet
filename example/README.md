# Walk-Through for PHOENIX-2014T

This document describes the rough procedure to train a SLTUnet model.

### Step 1. Download and Preprocess Dataset

1. Get the phoenix2014T dataset from [here](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/) or using
    ```shell
    wget https://www-i6.informatik.rwth-aachen.de/ftp/pub/rwth-phoenix/2016/phoenix-2014-T.v3.tar.gz
    ```
2. Get MuST-C En-De dataset from [FBK](https://ict.fbk.eu/must-c/); note we used the data in v1.0

We applied tokenization and subword modeling to these dataset. See [preprocess_phoenix.sh](./preprocess_phoenix.sh) for reference.

### Step 2. Pretrain SMKD Embeddings

We adopt the [SMKD](https://github.com/ycmin95/VAC_CSLR) method to pretrain sign embeddings and further
adapt it for sign language translation. `smkd` shows the adapted source code.

To pretrain SMKD embeddings,

1) preprocess the dataset
    ```bash
    python preprocess/dataset_preprocess.py --dataset phoenix2014 --dataset-root PHOENIX-2014
    -T-release-v3/PHOENIX-2014-T/
    ```
2) launch training
    ```bash
    python main.py --work-dir exp/resnet34 --config baseline.yaml --device 0,1
    ```
3) checkpoint averaging (optional)
   
   Among all saved checkpoints, select top-K (e.g. 5) checkpoint and put their (abs)path into a file named `checkpoint` under exp/resnet34
   ```bash
   python ckpt_avg.py  --path exp/resnet34 --checkpoints 5 --output avg
   ```
4) extract sign features
    ```bash
    python main.py --load-weights avg/average.pt --phase features --device 0 --num-feature-aug 10 --work-dir exp/resnet34 --config baseline.yaml
    ```
    Then combine different training features
    ```bash
    python sign_feature_cmb.py train\*h5 
    ```
At the end, you will have train/dev/test.h5 files as the sign feature inputs

### Step 3. Train SLTUnet Model

See the given running scripts `train.sh` for reference.

### Step 4. Decoding and Evaluation

1) we saved top-10 checkpoints based on dev set performance. we averaged them before final evaluation.

    ```bash
    python checkpoint_averaging.py  --path path-to-best-ckpt-dir --checkpoints 10 --output avg --gpu 0
    ```

2) See the given running scripts `test.sh` for decoding.

3) Regarding evaluation, please checkout `eval/metrics.py` for details.

For future evaluation and dataset construction, we suggest retaining the punctuations and 
adopt detokenized BLEU. E.g.

```bash
python eval/metrics.py -t slt -hyp model-output-file -ref gold-reference-file
```

