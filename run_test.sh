#!/bin/bash

./tools/test_craft.py --gpu 0 \
     --def /home/kwang/ilsvrc/fast-rcnn-ilsvrc-test/models/gn_hkbn/test_4d_700.prototxt \
     --net /home/kwang/model_zoo/bn_yangbin_pretrain_xyzeng_proposals_5000_stepsize/models/hkbn_4d_fast_rcnn_iter_110000.caffemodel \
     --imdb ilsvrc_2013_val2 \
     --cfg /home/kwang/ilsvrc/fast-rcnn-ilsvrc-test/experiments/cfgs/ilsvrc_700.yml \
     --bbox_mean /home/kwang/ilsvrc/fast-rcnn-ilsvrc-test/output/rpn90/bbox_means.pkl \
     --bbox_std /home/kwang/ilsvrc/fast-rcnn-ilsvrc-test/output/rpn90/bbox_stds.pkl

