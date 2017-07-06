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
from fast_rcnn.bbox_transform import clip_boxes
from utils.blob import prep_im_for_blob, im_list_to_blob, pred_rois_for_blob
from mnc.mask_transform import cpu_mask_voting, gpu_mask_voting

def im_detect(net, im):
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
    net.blobs['data'].reshape(*blobs['data'].shape)
    net.blobs['im_info'].reshape(*blobs['im_info'].shape)
    forward_kwargs = {
        'data': blobs['data'].astype(np.float32, copy=False),
        'im_info': blobs['im_info'].astype(np.float32, copy=False)
    }
    blobs_out = net.forward(**forward_kwargs)
    # output we need to collect:
    # 1. output from phase1'
    rois_phase1 = net.blobs['rois'].data.copy()
    masks_phase1 = net.blobs['mask_proposal'].data[...]
    scores_phase1 = net.blobs['seg_cls_prob'].data[...]
    # 2. output from phase2
    rois_phase2 = net.blobs['rois_ext'].data[...]
    masks_phase2 = net.blobs['mask_proposal_ext'].data[...]
    scores_phase2 = net.blobs['seg_cls_prob_ext'].data[...]
    # Boxes are in resized space, we un-scale them back
    rois_phase1 = rois_phase1[:, 1:5] / im_scales[0]
    rois_phase2 = rois_phase2[:, 1:5] / im_scales[0]
    rois_phase1 = clip_boxes(rois_phase1, im.shape)
    rois_phase2 = clip_boxes(rois_phase2, im.shape)
    # concatenate two stages to get final network output
    masks = np.concatenate((masks_phase1, masks_phase2), axis=0)
    boxes = np.concatenate((rois_phase1, rois_phase2), axis=0)
    scores = np.concatenate((scores_phase1, scores_phase2), axis=0)
    return masks, boxes, scores

def test_net(net, imdb, max_per_image=100):
    """Test a Fast R-CNN network on an image database."""
    num_images = len(imdb.image_index)
    max_per_set = 40 * num_images
    # detection threshold for each class
    # (this is adaptively set based on the max_per_set constraint)
    thresh = -np.inf * np.ones(imdb.num_classes)
    # top_scores will hold one min heap of scores per class (used to enforce
    # the max_per_set constraint)
    top_scores = [[] for _ in xrange(imdb.num_classes)]
    # all detections and segmentation are collected into a list:
    # Since the number of dets/segs are of variable size
    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb.num_classes)]
    all_masks = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb.num_classes)]

    output_dir = get_output_dir(imdb, net)

    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}

    for i in xrange(num_images):
        im = cv2.imread(imdb.image_path_at(i))
        _t['im_detect'].tic()
        masks, boxes, seg_scores = im_detect(net, im)
        _t['im_detect'].toc()

        _t['misc'].tic()
        if not cfg.TEST.USE_MASK_MERGE:
            for j in xrange(1, imdb.num_classes):
                inds = np.where(seg_scores[:, j] > thresh[j])[0]
                cls_scores = seg_scores[inds, j]
                cls_boxes = boxes[inds, :]
                cls_masks = masks[inds, :]
                top_inds = np.argsort(-cls_scores)[:max_per_image]
                cls_scores = cls_scores[top_inds]
                cls_boxes = cls_boxes[top_inds, :]
                cls_masks = cls_masks[top_inds, :]
                # push new scores onto the min heap
                for val in cls_scores:
                    heapq.heappush(top_scores[j], val)
                # if we've collected more than the max number of detection,
                # then pop items off the min heap and update the class threshold
                if len(top_scores[j]) > max_per_set:
                    while len(top_scores[j]) > max_per_set:
                        heapq.heappop(top_scores[j])
                    thresh[j] = top_scores[j][0]
                # Add new boxes into record
                box_before_nms = np.hstack((cls_boxes, cls_scores[:, np.newaxis]))\
                    .astype(np.float32, copy=False)
                mask_before_nms = cls_masks.astype(np.float32, copy=False)
                all_boxes[j][i], all_masks[j][i] = apply_nms_mask_single(box_before_nms, mask_before_nms, cfg.TEST.NMS)
        else:
            if cfg.TEST.USE_GPU_MASK_MERGE:
                result_mask, result_box = gpu_mask_voting(masks, boxes, seg_scores, imdb.num_classes,
                                                          max_per_image, im.shape[1], im.shape[0])
            else:
                result_box, result_mask = cpu_mask_voting(masks, boxes, seg_scores, imdb.num_classes,
                                                          max_per_image, im.shape[1], im.shape[0])
            # no need to create a min heap since the output will not exceed max number of detection
            for j in xrange(1, imdb.num_classes):
                all_boxes[j][i] = result_box[j-1]
                all_masks[j][i] = result_mask[j-1]
        _t['misc'].toc()

        print 'im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
              .format(i + 1, num_images, _t['im_detect'].average_time,
                      _t['misc'].average_time)

    for j in xrange(1, imdb.num_classes):
        for i in xrange(num_images):
            inds = np.where(all_boxes[j][i][:, -1] > thresh[j])[0]
            all_boxes[j][i] = all_boxes[j][i][inds, :]
            all_masks[j][i] = all_masks[j][i][inds]

    det_file = os.path.join(output_dir, 'res_boxes.pkl')
    seg_file = os.path.join(output_dir, 'res_masks.pkl')
    with open(det_file, 'wb') as f:
        cPickle.dump(all_boxes, f, cPickle.HIGHEST_PROTOCOL)
    with open(seg_file, 'wb') as f:
        cPickle.dump(all_masks, f, cPickle.HIGHEST_PROTOCOL)

    print 'Evaluating segmentation using MNC 5 stage inference'
    imdb.evaluate_segmentation(all_boxes, all_masks, output_dir)
