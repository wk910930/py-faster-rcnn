# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2016 CUHK
# Licensed under The MIT License [see LICENSE for details]
# Written by Kun Wang
# --------------------------------------------------------

import os
from random import shuffle

import numpy as np
from scipy import io as sio
import matplotlib.image as mpimg

import caffe
from fast_rcnn.bbox_transform import find_valid_ref_bboxes

class ReasoningDataLayer(caffe.Layer):
    """
    Data layer for training reasoning network
    """
    def setup(self, bottom, top):
        self.top_names = ['feat', 'labels']
        # read input parameters
        # params is a python dictionary with layer parameters
        params = eval(self.param_str)
        # check the parameters for validity
        check_params(params)
        # store input as class variables
        self.num_images = params['num_images']
        self.num_edges = params['num_edges']
        self.batch_size = self.num_images * self.num_edges
        self.feat_length = params['feat_length']
        # create a batch loader to load data
        self.batch_loader = BatchLoader(params, None)
        # reshape tops
        # since we use a fixed input image size, we can shape the data layer
        # once. Else, we'd have to do it in the reshape call.
        top[0].reshape(self.batch_size, self.feat_length, 1, 1)
        top[1].reshape(self.batch_size)

    def forward(self, bottom, top):
        """
        Load data.
        """
        feat_blob = np.zeros((self.batch_size, self.feat_length, 1, 1), dtype=np.float32)
        label_blob = np.zeros((self.batch_size), dtype=np.float32)
        for itt in range(self.num_images):
            # Use the batch loader to load the next image.
            feat, label = self.batch_loader.load_next_image()
            feat_blob[itt * self.num_edges: (itt + 1) * self.num_edges, :, :, :] = feat
            label_blob[itt * self.num_edges: (itt + 1) * self.num_edges] = label
        # Add directly to the caffe data layer
        top[0].data[...] = feat_blob
        top[1].data[...] = label_blob

    def reshape(self, bottom, top):
        """
        There is no need to reshape the data, since the input is of fixed size
        (rows and columns)
        """
        pass

    def backward(self, top, propagate_down, bottom):
        """
        These layers does not back propagate
        """
        pass

class BatchLoader(object):

    """
    This class abstracts away the loading of images.
    """

    def __init__(self, params, result):
        self.result = result
        self.num_images = params['num_images']
        self.num_edges = params['num_edges']
        self.batch_size = self.num_images * self.num_edges
        self.feat_length = params['feat_length']
        self.mat_root = params['mat_root']
        self.im_root = params['im_root']
        self.fg_thresh = params['fg_thresh']
        self.bg_thresh = params['bg_thresh']
        self.fg_fraction = params['fg_fraction']
        self.num_fg = int(self.num_edges * self.fg_fraction)
        self.num_bg = self.num_edges - self.num_fg
        # get list of image indexes.
        list_file = params['split']
        self.indexlist = [line.rstrip('\n') for line in open(list_file)]
        self._cur = 0  # current image

        print "BatchLoader initialized with {} images".format(
            len(self.indexlist))

    def load_next_image(self):
        """
        Load the next image in a batch.
        """
        # Did we finish an epoch?
        if self._cur == len(self.indexlist):
            self._cur = 0
            shuffle(self.indexlist)

        # Load an image and mat
        index = self.indexlist[self._cur]  # Get the image index
        mat_file = index + '.mat'
        img_file = index + '.JPEG'

        mat_dict = sio.loadmat(os.path.join(self.mat_root, mat_file))
        im = mpimg.imread(os.path.join(self.im_root, img_file))

        box_proposals = mat_dict['box_proposals'].astype(np.float)
        num_bbox = box_proposals.shape[0]

        fg_indices = np.where(np.squeeze(mat_dict['overlap'].astype(np.float32)) >= self.fg_thresh)[0]
        bg_indices = np.where(np.squeeze(mat_dict['overlap'].astype(np.float32)) < self.bg_thresh)[0]
        num_fg = fg_indices.shape[0]
        num_bg = bg_indices.shape[0]
        assert (num_fg + num_bg) > 0, 'we need at least one sample'

        # configure
        pivot_scale = 2.0
        iou_thresh = 0.01

        pivot_ref_overlaps = find_valid_ref_bboxes(box_proposals, im.shape, pivot_scale)

        feat_blob = np.zeros((self.num_edges, self.feat_length, 1, 1), dtype=np.float32)
        label_blob = np.zeros((self.num_edges), dtype=np.float32)

        for i in xrange(num_fg):
            pivot_index = fg_indices[np.random.randint(num_fg)]
            ref_indices = np.where(pivot_ref_overlaps[pivot_index, :] >= iou_thresh)[0]
            num_ref = ref_indices.shape[0]
            ref_index = ref_indices[np.random.randint(num_ref)]
            pivot_feat = mat_dict['global_pool'][[pivot_index], :]
            ref_feat = mat_dict['global_pool'][[ref_index], :]
            feat_concat = np.concatenate((pivot_feat, ref_feat), axis=1)
            labels = mat_dict['class_index'][0, pivot_index]
            feat_blob[i, :, :, :] = feat_concat
            assert labels > 0, '[{}] wrong positive label'.format(labels)
            label_blob[i] = labels

        for i in xrange(num_bg):
            pivot_index = bg_indices[np.random.randint(num_bg)]
            ref_indices = np.where(pivot_ref_overlaps[pivot_index, :] >= iou_thresh)[0]
            num_ref = ref_indices.shape[0]
            ref_index = ref_indices[np.random.randint(num_ref)]
            pivot_feat = mat_dict['global_pool'][[pivot_index], :]
            ref_feat = mat_dict['global_pool'][[ref_index], :]
            feat_concat = np.concatenate((pivot_feat, ref_feat), axis=1)
            labels = mat_dict['class_index'][0, pivot_index]
            feat_blob[num_fg + i, :, :, :] = feat_concat
            labels = 0
            label_blob[num_fg + i] = labels

        self._cur += 1
        return feat_blob, label_blob

def check_params(params):
    """
    A utility function to check the parameters for the data layers.
    """
    required = ['split', 'mat_root', 'im_root', 'num_images', 'num_edges', 'feat_length', 'fg_thresh', 'bg_thresh', 'fg_fraction']
    for item in required:
        assert item in params.keys(), 'Params must include {}'.format(item)
