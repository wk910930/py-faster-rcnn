#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2016 CUHK
# Written by Kun Wang
# --------------------------------------------------------

"""
Demo script showing detections in given dataset.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import cv2

CLASSES = ('__background__',  # always index 0
           'accordion', 'airplane', 'ant', 'antelope', 'apple',
           'armadillo', 'artichoke', 'axe', 'baby bed', 'backpack',
           'bagel', 'balance beam', 'banana', 'band aid', 'banjo',
           'baseball', 'basketball', 'bathing cap', 'beaker', 'bear',
           'bee', 'bell pepper', 'bench', 'bicycle', 'binder',
           'bird', 'bookshelf', 'bow', 'bow tie', 'bowl',
           'brassiere', 'burrito', 'bus', 'butterfly', 'camel',
           'can opener', 'car', 'cart', 'cattle', 'cello',
           'centipede', 'chain saw', 'chair', 'chime', 'cocktail shaker',
           'coffee maker', 'computer keyboard', 'computer mouse', 'corkscrew', 'cream',
           'croquet ball', 'crutch', 'cucumber', 'cup or mug', 'diaper',
           'digital clock', 'dishwasher', 'dog', 'domestic cat', 'dragonfly',
           'drum', 'dumbbell', 'electric fan', 'elephant', 'face powder',
           'fig', 'filing cabinet', 'flower pot', 'flute', 'fox',
           'french horn', 'frog', 'frying pan', 'giant panda', 'goldfish',
           'golf ball', 'golfcart', 'guacamole', 'guitar', 'hair dryer',
           'hair spray', 'hamburger', 'hammer', 'hamster', 'harmonica',
           'harp', 'hat with a wide brim', 'head cabbage', 'helmet', 'hippopotamus',
           'horizontal bar', 'horse', 'hotdog', 'iPod', 'isopod',
           'jellyfish', 'koala bear', 'ladle', 'ladybug', 'lamp',
           'laptop', 'lemon', 'lion', 'lipstick', 'lizard',
           'lobster', 'maillot', 'maraca', 'microphone', 'microwave',
           'milk can', 'miniskirt', 'monkey', 'motorcycle', 'mushroom',
           'nail', 'neck brace', 'oboe', 'orange', 'otter',
           'pencil box', 'pencil sharpener', 'perfume', 'person', 'piano',
           'pineapple', 'ping-pong ball', 'pitcher', 'pizza', 'plastic bag',
           'plate rack', 'pomegranate', 'popsicle', 'porcupine', 'power drill',
           'pretzel', 'printer', 'puck', 'punching bag', 'purse',
           'rabbit', 'racket', 'ray', 'red panda', 'refrigerator',
           'remote control', 'rubber eraser', 'rugby ball', 'ruler', 'salt or pepper shaker',
           'saxophone', 'scorpion', 'screwdriver', 'seal', 'sheep',
           'ski', 'skunk', 'snail', 'snake', 'snowmobile',
           'snowplow', 'soap dispenser', 'soccer ball', 'sofa', 'spatula',
           'squirrel', 'starfish', 'stethoscope', 'stove', 'strainer',
           'strawberry', 'stretcher', 'sunglasses', 'swimming trunks', 'swine',
           'syringe', 'table', 'tape player', 'tennis ball', 'tick',
           'tie', 'tiger', 'toaster', 'traffic light', 'train',
           'trombone', 'trumpet', 'turtle', 'tv or monitor', 'unicycle',
           'vacuum', 'violin', 'volleyball', 'waffle iron', 'washer',
           'water bottle', 'watercraft', 'whale', 'wine bottle', 'zebr')

def load_det_list(det_list):
    file_names = []
    with open(det_list) as f:
        for line in f:
            file_names.append(line.rstrip('\n'))
    return file_names

def load_dets(txt_file):
    """
    dets: [cls_id, score, x1, y1, x2, y2]
    """
    dets = np.loadtxt(txt_file, delimiter=' ')
    return dets

def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='blue', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(CLASSES[class_name[i]], score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('score >= {:.1f}').format(thresh), fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()

if __name__ == '__main__':
    det_list = '/home/kwang/Documents/ImageNet/det_lists/16test.txt'
    im_dir = '/home/kwang/Documents/ImageNet/ILSVRC2016/ILSVRC2016_DET_test'
    txt_file = '/home/kwang/Desktop/ilsvrc2016/evaluate/test_results_7on1.txt'
    # load image names
    file_names = load_det_list(det_list)

    dets = load_dets(txt_file)
    view_orders = np.random.permutation(len(file_names))
    for im_ind in xrange(52000, len(file_names), 100):
        file_name = file_names[im_ind]
        print '{} {}'.format(im_ind + 1, file_name)

        im_file = os.path.join(im_dir, file_name) + '.JPEG'
        im = cv2.imread(im_file)
        tmp_dets = dets[dets[:, 0] == im_ind+1, 2:]
        tmp_cls_ids = dets[dets[:, 0] == im_ind+1, 1].astype(int)
        tmp_dets = tmp_dets[:, (1, 2, 3, 4, 0)]

        vis_detections(im, tmp_cls_ids, tmp_dets, thresh=0.1)
        plt.show()
