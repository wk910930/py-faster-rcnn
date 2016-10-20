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
from fast_rcnn.bbox_transform import bbox_transform

class ReasoningDataLayer(caffe.Layer):
    """
    Data layer for training reasoning network
    """
    def setup(self, bottom, top):
        self.top_names = ['feat', 'spatial', 'scores', 'labels']
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
        top[0].reshape(self.batch_size, self.feat_length)
        top[1].reshape(self.batch_size, 4)
        top[2].reshape(self.batch_size, 402)
        top[3].reshape(self.batch_size)

    def forward(self, bottom, top):
        """
        Load data.
        """
        feat_blob = np.zeros((self.batch_size, self.feat_length), dtype=np.float32)
        spatial_blob = np.zeros((self.batch_size, 4), dtype=np.float32)
        scores_blob = np.zeros((self.batch_size, 402), dtype=np.float32)
        label_blob = np.zeros((self.batch_size), dtype=np.float32)
        for itt in range(self.num_images):
            # Use the batch loader to load the next image.
            feat, spatial, scores, label = self.batch_loader.load_next_image()
            feat_blob[itt * self.num_edges: (itt + 1) * self.num_edges, :] = feat
            spatial_blob[itt * self.num_edges: (itt + 1) * self.num_edges, :] = spatial
            scores_blob[itt * self.num_edges: (itt + 1) * self.num_edges, :] = scores
            label_blob[itt * self.num_edges: (itt + 1) * self.num_edges] = label
        # Add directly to the caffe data layer
        top[0].data[...] = feat_blob
        top[1].data[...] = spatial_blob
        top[2].data[...] = scores_blob
        top[3].data[...] = label_blob

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
        if self._cur == 0:
            shuffle(self.indexlist)

        # Load an image and mat
        index = self.indexlist[self._cur]  # Get the image index
        mat_file = index + '.mat'
        img_file = index + '.JPEG'
        if 'flip' in img_file:
            img_file = img_file[5:]

        mat_dict = sio.loadmat(os.path.join(self.mat_root, mat_file))
        im = mpimg.imread(os.path.join(self.im_root, img_file))

        box_proposals = mat_dict['box_proposals'].astype(np.float)
        class_index = np.squeeze(mat_dict['class_index'])
        overlap = np.squeeze(mat_dict['overlap'].astype(np.float32))
        feats = mat_dict['global_pool'].astype(np.float32)
        box_scores = mat_dict['scores'].astype(np.float32)
        assert box_proposals.shape[0] == class_index.shape[0], 'num mismatch'
        assert box_proposals.shape[0] == overlap.shape[0], 'num mismatch'
        assert box_proposals.shape[0] == feats.shape[0], 'num mismatch'

        fg_indices = np.where(overlap >= self.fg_thresh)[0]
        bg_indices = np.where(overlap < self.bg_thresh)[0]
        assert (fg_indices.shape[0] + bg_indices.shape[0]) > 0, 'we need at least one sample'

        if fg_indices.shape[0] == 0:
            fg_num = 0
            bg_num = self.num_edges
        elif bg_indices.shape[0] == 0:
            bg_num = 0
            fg_num - self.num_edges
        else:
            fg_num = self.num_fg
            bg_num = self.num_bg

        assert fg_num + bg_num == self.num_edges

        # configure
        pivot_scale = 2.0
        iou_thresh = 0.01

        pivot_ref_overlaps = find_valid_ref_bboxes(box_proposals, im.shape, pivot_scale)

        feat_blob = np.zeros((self.num_edges, self.feat_length), dtype=np.float32)
        spatial_blob = np.zeros((self.num_edges, 4), dtype=np.float32)
        scores_blob = np.zeros((self.num_edges, 402), dtype=np.float32)
        label_blob = np.zeros((self.num_edges), dtype=np.float32)

        for i in xrange(fg_num):
            pivot_index = np.random.choice(fg_indices, 1)
            ref_indices = np.where(pivot_ref_overlaps[pivot_index, :] >= iou_thresh)[0]
            ref_index = np.random.choice(ref_indices, 1)
            pivot_feat = np.squeeze(feats[pivot_index, :], axis=(2,3))
            ref_feat = np.squeeze(feats[ref_index, :], axis=(2,3))
            feat_blob[i, :] = np.concatenate((pivot_feat, ref_feat), axis=1)
            # label
            labels = class_index[pivot_index]
            assert labels > 0, '[{}] wrong positive label'.format(labels)
            label_blob[i] = labels
            # spatial
            pivot_bbox = box_proposals[pivot_index, :]
            ref_bbox = box_proposals[ref_index, :]
            target = bbox_transform(ref_bbox, pivot_bbox)[0, :]
            spatial_blob[i, :] = target
            # scores
            pivot_score = box_scores[pivot_index, :]
            ref_score = box_scores[ref_index, :]
            scores_blob[i, :] = np.concatenate((pivot_score, ref_score), axis=1)

        for i in xrange(bg_num):
            pivot_index = np.random.choice(bg_indices, 1)
            ref_indices = np.where(pivot_ref_overlaps[pivot_index, :] >= iou_thresh)[0]
            ref_index = np.random.choice(ref_indices, 1)
            pivot_feat = np.squeeze(feats[pivot_index, :], axis=(2, 3))
            ref_feat = np.squeeze(feats[ref_index, :], axis=(2, 3))
            feat_blob[fg_num + i, :] = np.concatenate((pivot_feat, ref_feat), axis=1)
            # label
            labels = 0
            label_blob[fg_num + i] = labels
            # spatial
            pivot_bbox = box_proposals[pivot_index, :]
            ref_bbox = box_proposals[ref_index, :]
            target = bbox_transform(ref_bbox, pivot_bbox)[0, :]
            spatial_blob[i, :] = target
            # scores
            pivot_score = box_scores[pivot_index, :]
            ref_score = box_scores[ref_index, :]
            scores_blob[i, :] = np.concatenate((pivot_score, ref_score), axis=1)

        self._cur += 1
        return feat_blob, spatial_blob, scores_blob, label_blob

def check_params(params):
    """
    A utility function to check the parameters for the data layers.
    """
    required = ['split', 'mat_root', 'im_root', 'num_images', 'num_edges', 'feat_length', 'fg_thresh', 'bg_thresh', 'fg_fraction']
    for item in required:
        assert item in params.keys(), 'Params must include {}'.format(item)
