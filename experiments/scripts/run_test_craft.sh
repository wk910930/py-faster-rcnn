#!/bin/bash

./tools/test_craft.py --gpu 0 \
     --def /home/kwang/deploy.prototxt \
     --net /home/kwang/hkbn_4d_fast_rcnn_iter_130000.caffemodel \
     --imdb ilsvrc_2013_val2 \
     --cfg ./experiments/cfgs/craft.yml \
     --bbox_mean /home/kwang/bbox_means.pkl \
     --bbox_std /home/kwang/bbox_stds.pkl

