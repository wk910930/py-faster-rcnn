#!/usr/bin/env python

# --------------------------------------------------------
# Multitask Network Cascade
# Modified from py-faster-rcnn (https://github.com/rbgirshick/py-faster-rcnn)
# Copyright (c) 2016, Haozhi Qi
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

# Standard module
import argparse
import sys
import pprint
import PIL
import numpy as np
# User-defined module
import _init_paths
from datasets.factory import get_imdb
from fast_rcnn.config import cfg, cfg_from_file, get_output_dir
from mnc.train import get_training_roidb, get_training_maskdb, train_net
import caffe


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--solver', dest='solver',
                        help='solver prototxt',
                        default=None, type=str)
    parser.add_argument('--iters', dest='max_iters',
                        help='number of iterations to train',
                        default=40000, type=int)
    parser.add_argument('--weights', dest='pretrained_model',
                        help='initialize with pretrained model weights',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to train on',
                        default='sbd_2012_train', type=str)
    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
                        action='store_true')
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

def attach_roidb(imdb_names):
    def get_roidb(imdb_name):
        imdb = get_imdb(imdb_name)
        print 'Loaded dataset `{:s}` for training'.format(imdb.name)
        imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
        print 'Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD)
        roidb = get_training_roidb(imdb)
        return roidb

    roidbs = [get_roidb(s) for s in imdb_names.split('+')]
    roidb = roidbs[0]
    if len(roidbs) > 1:
        raise NotImplementedError
    else:
        imdb = get_imdb(imdb_names)
    return imdb, roidb

def attach_maskdb(imdb_names):
    def get_maskdb(imdb_name):
        imdb = get_imdb(imdb_name)
        print 'Loaded dataset `{:s}` for training'.format(imdb.name)
        imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
        print 'Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD)
        imdb.set_mask_method(cfg.TRAIN.PROPOSAL_METHOD)
        print 'Set mask method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD)
        maskdb = get_training_maskdb(imdb)
        return maskdb

    maskdbs = [get_maskdb(s) for s in imdb_names.split('+')]
    maskdb = maskdbs[0]
    if len(maskdbs) > 1:
        raise NotImplementedError
    else:
        imdb = get_imdb(imdb_names)
    return imdb, maskdb

if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    cfg.GPU_ID = args.gpu_id

    print('Using config:')
    pprint.pprint(cfg)

    if not args.randomize:
        # fix the random seeds (numpy and caffe) for reproducibility
        np.random.seed(cfg.RNG_SEED)
        caffe.set_random_seed(cfg.RNG_SEED)

    # set up caffe
    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)

    # get imdb, roidb and maskdb from specified imdb_name
    # TODO(kun): Currently we get roidb and maskdb separately, which means
    #   we are creating two imdbs. We can consider to merge them to save time.
    imdb_roi, roidb = attach_roidb(args.imdb_name)
    imdb_mask, maskdb = attach_maskdb(args.imdb_name)
    print '{:d} roidb entries'.format(len(roidb))
    print '{:d} maskdb entries'.format(len(maskdb))
    assert imdb_roi.image_index == imdb_mask.image_index

    output_dir = get_output_dir(imdb_roi)  # imdb_mask also works
    print 'Output will be saved to `{:s}`'.format(output_dir)

    train_net(args.solver, roidb, maskdb, output_dir,
              pretrained_model=args.pretrained_model,
              max_iters=args.max_iters)
