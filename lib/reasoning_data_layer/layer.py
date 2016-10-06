# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2016 CUHK
# Licensed under The MIT License [see LICENSE for details]
# Written by Kun Wang
# --------------------------------------------------------

import os
from scipy import io as sio
import numpy as np
import matplotlib.image as mpimg
from random import shuffle

import caffe
from fast_rcnn.config import cfg
from fast_rcnn.bbox_transform import find_valid_ref_bboxes

class ReasoningDataLayer(caffe.Layer):
    """
    Data layer for training reasoning network
    """
    def setup(self, bottom, top):
        self.top_names = ['feat', 'lable']
        # read input parameters
        # params is a python dictionary with layer parameters
        params = eval(self.param_str)
        # check the parameters for validity
        check_params(params)
        # store input as class variables
        self.num_pivot = params['num_pivot']
        self.num_ref = params['num_ref']
        self.feat_length = params['feat_length']
        self.batch_size = self.num_pivot * self.num_ref
        # create a batch loader to load data
        self.batch_loader = BatchLoader(params, None)
        # reshape tops
        # since we use a fixed input image size, we can shape the data layer
        # once. Else, we'd have to do it in the reshape call.
        top[0].reshape(self.batch_size, self.feat_length * 2, 1, 1)
        top[1].reshape(self.batch_size)

    def forward(self, bottom, top):
        """
        Load data.
        """
        # Use the batch loader to load the next image.
        feat_blob, label_blob = self.batch_loader.load_next_image()
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
        self.num_pivot = params['num_pivot']
        self.num_ref = params['num_ref']
        self.batch_size = self.num_pivot * self.num_ref
        self.mat_root = params['mat_root']
        self.im_root = params['im_root']
        self.feat_length = params['feat_length']
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

        # Load an image
        index = self.indexlist[self._cur]  # Get the image index
        mat_file = index + '.mat'
        img_file = index + '.JPEG'
        mat_dict = sio.loadmat(os.path.join(self.mat_root, mat_file))

        box_proposals = mat_dict['box_proposals'].astype(np.float)
    
        pivot_scale = 2.0
        iou_thresh = 0.01
        num_bbox = box_proposals.shape[0]

        im = mpimg.imread(os.path.join(self.im_root, img_file))

        pivot_ref_overlaps = find_valid_ref_bboxes(box_proposals, im.shape, pivot_scale)

        feat_blob = np.zeros((self.batch_size, self.feat_length * 2, 1, 1), dtype=np.float32)
        label_blob = np.zeros((self.batch_size), dtype=np.float32)
        for i in xrange(self.batch_size):
            pivot_index = np.random.randint(num_bbox)
            ref_indices = np.where(pivot_ref_overlaps[pivot_index, :] >= iou_thresh)[0]
            ref_index = ref_indices[np.random.randint(ref_indices.shape[0])]
            pivot_feat = mat_dict['global_pool'][[pivot_index], :]
            ref_feat = mat_dict['global_pool'][[ref_index], :]
            feat_concat = np.concatenate((pivot_feat, ref_feat), axis=1)
            labels = mat_dict['class_index'][:, pivot_index]
            feat_blob[i, :, :, :] = feat_concat
            label_blob[i] = labels

        self._cur += 1
        return feat_blob, label_blob

def check_params(params):
    """
    A utility function to check the parameters for the data layers.
    """
    required = ['split', 'mat_root', 'im_root', 'num_pivot', 'num_ref', 'feat_length']
    for r in required:
        assert r in params.keys(), 'Params must include {}'.format(r)
