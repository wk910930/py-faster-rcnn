#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2018 CUHK
# Written by Wang Kun
# --------------------------------------------------------

import cPickle
import PIL
import _init_paths
import numpy as np
from datasets.coco import coco

def prepare_gt_roidb(imdb, gt_roidb):
    """
    Add image meta information, i.e. height, width, channels and image path.
    """
    sizes = [PIL.Image.open(imdb.image_path_at(i)).size
             for i in xrange(imdb.num_images)]
    bands = [PIL.Image.open(imdb.image_path_at(i)).getbands()
             for i in xrange(imdb.num_images)]
    for i in xrange(len(imdb.image_index)):
        gt_roidb[i]['image'] = imdb.image_path_at(i)
        gt_roidb[i]['width'] = sizes[i][0]
        gt_roidb[i]['height'] = sizes[i][1]
        gt_roidb[i]['channel'] = len(bands[i])

def main():
    image_set = 'train'
    year = '2014'
    label_map_dict = {3:1, 6:1, 8:1, 2:2, 4:2, 1:3}
    image_min_size = 256
    output = 'coco_{}_{}.txt'.format(year, image_set)

    # Load images and gt
    imdb = coco(image_set, year)
    gt_roidb = imdb.gt_roidb()
    prepare_gt_roidb(imdb, gt_roidb)

    valid_img_idx = 0
    with open(output, 'w') as f:
        for image_index in xrange(len(gt_roidb)):
            image = gt_roidb[image_index]

            if image['height'] < image_min_size or image['width'] < image_min_size:
                continue

            assert image['boxes'].shape[0] == image['gt_classes'].shape[0]

            gt_idx_list = []
            for key in label_map_dict:
                gt_idx_list += np.where(image['gt_classes'] == key)[0].tolist()
            for k in gt_idx_list:
                if image['gt_overlaps'][k, key] == -1:
                    gt_idx_list.remove(k)
            num_gts = len(gt_idx_list)
            if num_gts == 0:
                continue

            valid_img_idx += 1
            # Write to disk
            f.write('# {}\n'.format(valid_img_idx))
            f.write('{}\n'.format(image['image']))
            f.write('{} {} {} {}\n'.format(image['channel'], image['height'], image['width'], 1))
            f.write('{}\n'.format(0))

            f.write('{}\n'.format(num_gts))
            for k in gt_idx_list:
                # class_index
                class_index = label_map_dict[image['gt_classes'][k]]
                f.write('{} '.format(class_index))
                # x1 y1 x2 y2
                x1 = image['boxes'][k, 0]
                y1 = image['boxes'][k, 1]
                x2 = image['boxes'][k, 2]
                y2 = image['boxes'][k, 3]
                f.write('{} {} {} {}\n'.format(x1, y1, x2, y2))
            f.write('\n')

            if image_index % 1000 == 0:
                print '{}/{}'.format(image_index, len(gt_roidb))

if __name__ == '__main__':
    main()
