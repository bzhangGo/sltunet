# DGS3-T Construction

**Note before using this dataset, get the license from [the Public DGS Corpus](https://www.sign-lang.uni-hamburg.de/meinedgs/ling/license_en.html) first!**

The dataset construction is maintained in [another repository](https://github.com/bricksdont/datasets). 
Here we offer the instructions to reconstruct the one used in our paper as follows:

```bash

# requirement: ffmpeg
# if not installed, for example download from here: https://johnvansickle.com/ffmpeg/

# download the dataset git
pip install git+https://github.com/bricksdont/datasets.git@fix_fps_check

# generate DGS3-T
python generate_examples_dgs.py --tfds-data-dir tfds_datasets_custom --preprocess-glosses --output examples.json > generate.out 2> generate.err

# split the whole document-level video into sentence-level segments
python slice_videos.py --input examples.json --output-folder output --ffmpeg-custom-path "ffmpeg/ffmpeg-git-20220722-amd64-static/ffmpeg" --num-workers 8 > slice
.out 2> slice.err
```

The DGS3-T dataset is in `output` folder.

