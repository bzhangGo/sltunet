#! /bin/bash

python ../main.py --work-dir exp/resnet34 --config baseline.yaml --device 5,6
# python ../main.py --work-dir exp/resnet34 --config baseline.yaml --load-weights exp/avg/average.pt --device 3 --phase test






