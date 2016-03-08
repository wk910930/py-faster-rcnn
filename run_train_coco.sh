#!/usr/bin/bash

./tools/train_net.py --gpu 0 \
    --solver models/coco/VGG16/fast_rcnn/solver.prototxt \
    --weights data/imagenet_models/VGG16.v2.caffemodel \
    --imdb coco_2014_train \
    --iters 50000
