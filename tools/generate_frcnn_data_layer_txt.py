#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2016 CUHK
# Written by Wang Kun
# --------------------------------------------------------

"""Generate txt file as the input to craft::frcnn_train_data_layer"""

import _init_paths
from datasets.coco import coco
from roi_data_layer.roidb import prepare_roidb
from roi_data_layer.roidb import add_bbox_regression_targets

imdb = coco('train', '2014')
prepare_roidb(imdb)
roidb = imdb.roidb
mean, std = add_bbox_regression_targets(roidb)

with open('rois_coco_train_2014.txt', 'w') as f:
    for image_index in xrange(len(roidb)):
        if image_index % 1000 == 0:
            print '{}/{}'.format(image_index, len(roidb))
        box = roidb[image_index]
        num_windows = box['boxes'].shape[0]
        if num_windows == 0:
            print '{} does not have proposals!'.format(box['image'])
            continue
        if box['channel'] not in (1, 3):
            print '{} has strange channels[{}]!'.format(box['image'], box['channel'])
        # image_index img_path channels height width
        f.write('# {}\n{}\n{}\n{}\n{}\n'.format(
            image_index, box['image'], box['channel'], box['height'], box['width']))
        # flipped
        f.write('{}\n'.format(0))
        # num_windows
        f.write('{}\n'.format(num_windows))
        class_index = box['max_classes']
        overlap = box['max_overlaps']
        for k in xrange(num_windows):
            # class_index
            f.write('{} '.format(class_index[k]))
            x1 = box['boxes'][k, 0]
            y1 = box['boxes'][k, 1]
            x2 = box['boxes'][k, 2]
            y2 = box['boxes'][k, 3]
            # overlap
            f.write('%.2f ' % overlap[k])
            # x1 y1 x2 y2
            f.write('{} {} {} {} '.format(x1, y1, x2, y2))
            dx = box['bbox_targets'][k, 1]
            dy = box['bbox_targets'][k, 2]
            dw = box['bbox_targets'][k, 3]
            dh = box['bbox_targets'][k, 4]
            # dx dy dw dh
            f.write('%.2f %.2f %.2f %.2f\n' % (dx, dy, dw, dh))
