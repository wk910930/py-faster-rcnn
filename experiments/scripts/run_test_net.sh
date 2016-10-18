#!/bin/bash

./tools/test_net.py --gpu 0 \
     --def /home/kwang/model_zoo/cvpr2017/bn/bn_3k_img/deploy.prototxt \
     --net /home/kwang/model_zoo/cvpr2017/bn/bn_3k_img/models/hkbn_4d_fast_rcnn_iter_120000.caffemodel \
     --imdb ilsvrc_2013_val2 \
     --comp \
     --cfg ./experiments/cfgs/frcnn.yml \
     --bbox_mean /home/kwang/proposals/ilsvrc/liuyu/bbox_means.pkl \
     --bbox_std /home/kwang/proposals/ilsvrc/liuyu/bbox_stds.pkl \
     --reasoning_def /home/kwang/py-faster-rcnn/models/reasoning/repeat-exactly-same/deploy.prototxt \
     --reasoning_net /home/kwang/py-faster-rcnn/models/reasoning/repeat-exactly-same/models/bn_iter_120000.caffemodel
