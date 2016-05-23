#!/bin/bash

./tools/test_craft.py --gpu 0 \
     --def test.prototxt \
     --net model.caffemodel \
     --imdb ilsvrc_2013_val2 \
     --comp \
     --cfg ./experiments/cfgs/craft.yml \
     --bbox_mean bbox_means.pkl \
     --bbox_std bbox_stds.pkl
