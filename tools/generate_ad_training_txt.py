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
    # Configuration
    image_set = 'train'
    year = '2014'
    valid_label = 1
    output = 'coco_{}_{}.txt'.format(year, image_set)

    # Load images and gt
    imdb = coco(image_set, year)
    gt_roidb = imdb.gt_roidb()
    prepare_gt_roidb(imdb, gt_roidb)

    # Filter gt_roidb
    print 'Filtering out images with no gt boxes...'
    remove_indices = []
    for i in xrange(len(gt_roidb)):
        if gt_roidb[i]['boxes'].shape[0] == 0:
            remove_indices.append(i)
    gt_roidb = [i for j, i in enumerate(gt_roidb) if j not in remove_indices]
    print '{} images are filtered'.format(len(remove_indices))

    valid_img_idx = 0
    with open(output, 'w') as f:
        for image_index in xrange(len(gt_roidb)):
            image = gt_roidb[image_index]
            assert image['boxes'].shape[0] == image['gt_classes'].shape[0]

            valid_gt_box_idx_list = np.where(image['gt_classes'] == valid_label)[0]
            num_gts = len(valid_gt_box_idx_list)
            # Skip images without valid_label
            if num_gts == 0:
                continue

            valid_img_idx += 1
            # Write to disk
            f.write('# {}\n'.format(valid_img_idx))
            f.write('{}\n'.format(image['image']))
            f.write('{} {} {} {}\n'.format(image['channel'], image['height'], image['width'], 1))
            f.write('{}\n'.format(0))
            f.write('{}\n'.format(num_gts))
            for k in valid_gt_box_idx_list:
                # class_index
                class_index = image['gt_classes'][k]
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
