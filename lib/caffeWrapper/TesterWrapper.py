# --------------------------------------------------------
# Multitask Network Cascade
# Modified from py-faster-rcnn (https://github.com/rbgirshick/py-faster-rcnn)
# Copyright (c) 2016, Haozhi Qi
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import os
import cPickle
import scipy
import numpy as np
import cv2
import heapq

import caffe
from utils.timer import Timer
from nms.nms_wrapper import apply_nms_mask_single
from fast_rcnn.config import cfg, get_output_dir
from fast_rcnn.bbox_transform import bbox_transform_inv
from fast_rcnn.bbox_transform import clip_boxes_mnc
from utils.blob import prep_im_for_blob, im_list_to_blob, pred_rois_for_blob
from datasets.transform.mask_transform import cpu_mask_voting, gpu_mask_voting


class TesterWrapper(object):
    """
    A simple wrapper around Caffe's test forward
    """
    def __init__(self, test_prototxt, imdb, test_model, task_name):
        # Pre-processing, test whether model stored in binary file or npy files
        self.net = caffe.Net(test_prototxt, test_model, caffe.TEST)
        self.net.name = os.path.splitext(os.path.basename(test_model))[0]
        self.imdb = imdb
        self.output_dir = get_output_dir(imdb, self.net)
        self.task_name = task_name
        # We define some class variables here to avoid defining them many times in every method
        self.num_images = len(self.imdb.image_index)
        self.num_classes = self.imdb.num_classes
        # heuristic: keep an average of 40 detections per class per images prior to nms
        self.max_per_set = 40 * self.num_images
        # heuristic: keep at most 100 detection per class per image prior to NMS
        self.max_per_image = 100

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def get_result(self):
        output_dir = self.output_dir
        det_file = os.path.join(output_dir, 'res_boxes.pkl')
        seg_file = os.path.join(output_dir, 'res_masks.pkl')
        if self.task_name == 'vis_seg':
            self.vis_segmentation_result()
        elif self.task_name == 'seg':
            if os.path.isfile(det_file) and os.path.isfile(seg_file):
                with open(det_file, 'rb') as f:
                    seg_box = cPickle.load(f)
                with open(seg_file, 'rb') as f:
                    seg_mask = cPickle.load(f)
            else:
                seg_box, seg_mask = self.get_segmentation_result()
                with open(det_file, 'wb') as f:
                    cPickle.dump(seg_box, f, cPickle.HIGHEST_PROTOCOL)
                with open(seg_file, 'wb') as f:
                    cPickle.dump(seg_mask, f, cPickle.HIGHEST_PROTOCOL)
            print 'Evaluating segmentation using MNC 5 stage inference'
            self.imdb.evaluate_segmentation(seg_box, seg_mask, output_dir)
        else:
            print 'task name only support \'seg\', and \'vis_seg\''
            raise NotImplementedError

    def vis_segmentation_result(self):
        self.imdb.visualization_segmentation(self.output_dir)

    def get_segmentation_result(self):
        # detection threshold for each class
        # (this is adaptively set based on the max_per_set constraint)
        thresh = -np.inf * np.ones(self.num_classes)
        # top_scores will hold one min heap of scores per class (used to enforce
        # the max_per_set constraint)
        top_scores = [[] for _ in xrange(self.num_classes)]
        # all detections and segmentation are collected into a list:
        # Since the number of dets/segs are of variable size
        all_boxes = [[[] for _ in xrange(self.num_images)]
                     for _ in xrange(self.num_classes)]
        all_masks = [[[] for _ in xrange(self.num_images)]
                     for _ in xrange(self.num_classes)]

        _t = {'im_detect': Timer(), 'misc': Timer()}
        for i in xrange(self.num_images):
            im = cv2.imread(self.imdb.image_path_at(i))
            _t['im_detect'].tic()
            masks, boxes, seg_scores = self._segmentation_forward(im)
            _t['im_detect'].toc()
            if not cfg.TEST.USE_MASK_MERGE:
                for j in xrange(1, self.num_classes):
                    inds = np.where(seg_scores[:, j] > thresh[j])[0]
                    cls_scores = seg_scores[inds, j]
                    cls_boxes = boxes[inds, :]
                    cls_masks = masks[inds, :]
                    top_inds = np.argsort(-cls_scores)[:self.max_per_image]
                    cls_scores = cls_scores[top_inds]
                    cls_boxes = cls_boxes[top_inds, :]
                    cls_masks = cls_masks[top_inds, :]
                    # push new scores onto the min heap
                    for val in cls_scores:
                        heapq.heappush(top_scores[j], val)
                    # if we've collected more than the max number of detection,
                    # then pop items off the min heap and update the class threshold
                    if len(top_scores[j]) > self.max_per_set:
                        while len(top_scores[j]) > self.max_per_set:
                            heapq.heappop(top_scores[j])
                        thresh[j] = top_scores[j][0]
                    # Add new boxes into record
                    box_before_nms = np.hstack((cls_boxes, cls_scores[:, np.newaxis]))\
                        .astype(np.float32, copy=False)
                    mask_before_nms = cls_masks.astype(np.float32, copy=False)
                    all_boxes[j][i], all_masks[j][i] = apply_nms_mask_single(box_before_nms, mask_before_nms, cfg.TEST.NMS)
            else:
                if cfg.TEST.USE_GPU_MASK_MERGE:
                    result_mask, result_box = gpu_mask_voting(masks, boxes, seg_scores, self.num_classes,
                                                              self.max_per_image, im.shape[1], im.shape[0])
                else:
                    result_box, result_mask = cpu_mask_voting(masks, boxes, seg_scores, self.num_classes,
                                                              self.max_per_image, im.shape[1], im.shape[0])
                # no need to create a min heap since the output will not exceed max number of detection
                for j in xrange(1, self.num_classes):
                    all_boxes[j][i] = result_box[j-1]
                    all_masks[j][i] = result_mask[j-1]

            print 'process image %d/%d, forward average time %f' % (i, self.num_images,
                                                                    _t['im_detect'].average_time)

        for j in xrange(1, self.num_classes):
            for i in xrange(self.num_images):
                inds = np.where(all_boxes[j][i][:, -1] > thresh[j])[0]
                all_boxes[j][i] = all_boxes[j][i][inds, :]
                all_masks[j][i] = all_masks[j][i][inds]

        return all_boxes, all_masks

    def _detection_forward(self, im):
        """ Detect object classes in an image given object proposals.
        Arguments:
            im (ndarray): color image to test (in BGR order)
        Returns:
            box_scores (ndarray): R x K array of object class scores (K includes
                background as object category 0)
            all_boxes (ndarray): R x (4*K) array of predicted bounding boxes
        """
        forward_kwargs, im_scales = self._prepare_mnc_args(im)
        blobs_out = self.net.forward(**forward_kwargs)
        # There are some data we need to get:
        # 1. ROIS (with bbox regression)
        rois = self.net.blobs['rois'].data.copy()
        # un-scale back to raw image space
        boxes = rois[:, 1:5] / im_scales[0]
        box_deltas = blobs_out['bbox_pred']
        pred_boxes = bbox_transform_inv(boxes, box_deltas)
        pred_boxes, _ = clip_boxes_mnc(pred_boxes, im.shape)
        # 2. Detection score
        scores = blobs_out['cls_prob']
        return scores, pred_boxes

    def _segmentation_forward(self, im):
        forward_kwargs, im_scales = self._prepare_mnc_args(im)
        blobs_out = self.net.forward(**forward_kwargs)
        # output we need to collect:
        # 1. output from phase1'
        rois_phase1 = self.net.blobs['rois'].data.copy()
        masks_phase1 = self.net.blobs['mask_proposal'].data[...]
        scores_phase1 = self.net.blobs['seg_cls_prob'].data[...]
        # 2. output from phase2
        rois_phase2 = self.net.blobs['rois_ext'].data[...]
        masks_phase2 = self.net.blobs['mask_proposal_ext'].data[...]
        scores_phase2 = self.net.blobs['seg_cls_prob_ext'].data[...]
        # Boxes are in resized space, we un-scale them back
        rois_phase1 = rois_phase1[:, 1:5] / im_scales[0]
        rois_phase2 = rois_phase2[:, 1:5] / im_scales[0]
        rois_phase1, _ = clip_boxes_mnc(rois_phase1, im.shape)
        rois_phase2, _ = clip_boxes_mnc(rois_phase2, im.shape)
        # concatenate two stages to get final network output
        masks = np.concatenate((masks_phase1, masks_phase2), axis=0)
        boxes = np.concatenate((rois_phase1, rois_phase2), axis=0)
        scores = np.concatenate((scores_phase1, scores_phase2), axis=0)
        return masks, boxes, scores

    def _prepare_mnc_args(self, im):
        # Prepare image data blob
        blobs = {'data': None}
        processed_ims = []
        im, im_scale_factors = \
            prep_im_for_blob(im, cfg.PIXEL_MEANS, cfg.TEST.SCALES[0], cfg.TRAIN.MAX_SIZE)
        processed_ims.append(im)
        blobs['data'] = im_list_to_blob(processed_ims)
        # Prepare image info blob
        im_scales = [np.array(im_scale_factors)]
        assert len(im_scales) == 1, 'Only single-image batch implemented'
        im_blob = blobs['data']
        blobs['im_info'] = np.array(
            [[im_blob.shape[2], im_blob.shape[3], im_scales[0]]],
            dtype=np.float32)
        # Reshape network inputs and do forward
        self.net.blobs['data'].reshape(*blobs['data'].shape)
        self.net.blobs['im_info'].reshape(*blobs['im_info'].shape)
        forward_kwargs = {
            'data': blobs['data'].astype(np.float32, copy=False),
            'im_info': blobs['im_info'].astype(np.float32, copy=False)
        }
        return forward_kwargs, im_scales
