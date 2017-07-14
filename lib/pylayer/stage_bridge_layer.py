# --------------------------------------------------------
# Multitask Network Cascade
# Written by Haozhi Qi
# Copyright (c) 2016, Haozhi Qi
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import caffe
import numpy as np
import yaml
from mnc.mask_transform import intersect_mask
from fast_rcnn.config import cfg
from fast_rcnn.bbox_transform import bbox_transform, bbox_transform_inv
from fast_rcnn.bbox_transform import clip_boxes_mnc
from utils.cython_bbox import bbox_overlaps


class StageBridgeLayer(caffe.Layer):
    """
    This layer take input from bounding box prediction
    and output a set of new rois after applying transformation
    It will also provide mask/bbox regression targets
    during training phase
    """
    def setup(self, bottom, top):
        layer_params = yaml.load(self.param_str_)
        # bottom 0 is ~ n ROIs to train Fast RCNN
        # bottom 1 is ~ n * 4(1+c) bbox prediction
        # bottom 2 is ~ n * (1+c) bbox scores (seg classification)
        self._phase = str(self.phase)
        if self._phase == 'TRAIN':
            self._feat_stride = layer_params['feat_stride']
            self._num_classes = layer_params['num_classes']

        # meaning of top blobs speak for themselves
        self._top_name_map = {}
        if self._phase == 'TRAIN':
            top[0].reshape(1, 5)
            self._top_name_map['rois'] = 0
            top[1].reshape(1, 1)
            self._top_name_map['labels'] = 1
            top[2].reshape(1, 1, cfg.MASK_SIZE, cfg.MASK_SIZE)
            self._top_name_map['mask_targets'] = 2
            top[3].reshape(1, 1, cfg.MASK_SIZE, cfg.MASK_SIZE)
            self._top_name_map['mask_weight'] = 3
            top[4].reshape(1, 4)
            self._top_name_map['gt_mask_info'] = 4
            top[5].reshape(1, self._num_classes * 4)
            self._top_name_map['bbox_targets'] = 5
            top[6].reshape(1, self._num_classes * 4)
            self._top_name_map['bbox_inside_weights'] = 6
            top[7].reshape(1, self._num_classes * 4)
            self._top_name_map['bbox_outside_weights'] = 7
        elif self._phase == 'TEST':
            top[0].reshape(1, 5)
            self._top_name_map['rois'] = 0
        else:
            print 'Unrecognized phase'
            raise NotImplementedError

    def reshape(self, bottom, top):
        # reshape happens during forward
        pass

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def forward(self, bottom, top):
        if str(self.phase) == 'TRAIN':
            blobs = self.forward_train(bottom, top)
        elif str(self.phase) == 'TEST':
            blobs = self.forward_test(bottom, top)
        else:
            print 'Unrecognized phase'
            raise NotImplementedError

        for blob_name, blob in blobs.iteritems():
            top[self._top_name_map[blob_name]].reshape(*blob.shape)
            top[self._top_name_map[blob_name]].data[...] = blob.astype(np.float32, copy=False)

    def forward_train(self, bottom, top):
        """
        During forward, we need to do several things:
        1. Apply bounding box regression output which has highest
           classification score to proposed ROIs
        2. Sample ROIs based on there current overlaps, assign labels
           on them
        3. Make mask regression targets and positive/negative weights,
           just like the proposal_target_layer
        """
        rois = bottom[0].data
        bbox_deltas = bottom[1].data
        # Apply bounding box regression according to maximum segmentation score
        seg_scores = bottom[2].data
        self._bbox_reg_labels = seg_scores[:, 1:].argmax(axis=1) + 1

        gt_boxes = bottom[3].data
        gt_masks = bottom[4].data
        im_info = bottom[5].data[0, :]
        mask_info = bottom[6].data

        # select bbox_deltas according to
        artificial_deltas = np.zeros((rois.shape[0], 4))
        for i in xrange(rois.shape[0]):
            artificial_deltas[i, :] = bbox_deltas[i, 4*self._bbox_reg_labels[i]:4*(self._bbox_reg_labels[i]+1)]
        artificial_deltas[self._bbox_reg_labels == 0, :] = 0

        all_rois = np.zeros((rois.shape[0], 5))
        all_rois[:, 0] = 0
        all_rois[:, 1:5] = bbox_transform_inv(rois[:, 1:5], artificial_deltas)
        zeros = np.zeros((gt_boxes.shape[0], 1), dtype=gt_boxes.dtype)
        all_rois = np.vstack(
            (all_rois, np.hstack((zeros, gt_boxes[:, :-1])))
        )
        all_rois[:, 1:5], self._clip_keep = clip_boxes_mnc(all_rois[:, 1:5], im_info[:2])

        labels, rois_out, fg_inds, keep_inds, mask_targets, top_mask_info, bbox_targets, bbox_inside_weights = \
            self._sample_output(all_rois, gt_boxes, im_info[2], gt_masks, mask_info)
        bbox_outside_weights = np.array(bbox_inside_weights > 0).astype(np.float32)
        self._keep_inds = keep_inds

        mask_weight = np.zeros((rois_out.shape[0], 1, cfg.MASK_SIZE, cfg.MASK_SIZE))
        mask_weight[0:len(fg_inds), :, :, :] = 1

        blobs = {
            'rois': rois_out,
            'labels': labels,
            'mask_targets': mask_targets,
            'mask_weight': mask_weight,
            'gt_mask_info': top_mask_info,
            'bbox_targets': bbox_targets,
            'bbox_inside_weights': bbox_inside_weights,
            'bbox_outside_weights': bbox_outside_weights
        }
        return blobs

    def _sample_output(self, all_rois, gt_boxes, im_scale, gt_masks, mask_info):
        overlaps = bbox_overlaps(
            np.ascontiguousarray(all_rois[:, 1:5], dtype=np.float),
            np.ascontiguousarray(gt_boxes[:, :4], dtype=np.float))
        gt_assignment = overlaps.argmax(axis=1)
        max_overlaps = overlaps.max(axis=1)
        labels = gt_boxes[gt_assignment, 4]
        # Sample foreground indexes
        fg_inds = np.where(max_overlaps >= cfg.TRAIN.BBOX_THRESH)[0]
        bg_inds = np.where(max_overlaps < cfg.TRAIN.BBOX_THRESH)[0]
        keep_inds = np.append(fg_inds, bg_inds).astype(int)
        # Select sampled values from various arrays:
        labels = labels[keep_inds]
        # Clamp labels for the background RoIs to 0
        labels[len(fg_inds):] = 0
        rois = all_rois[keep_inds]

        bbox_target_data = _compute_targets(
        rois[:, 1:5], gt_boxes[gt_assignment[keep_inds], :4], labels)

        bbox_targets, bbox_inside_weights = _get_bbox_regression_labels(
            bbox_target_data, self._num_classes)

        scaled_rois = rois[:, 1:5] / float(im_scale)
        scaled_gt_boxes = gt_boxes[:, :4] / float(im_scale)

        pos_masks = np.zeros((len(keep_inds), 1,  cfg.MASK_SIZE,  cfg.MASK_SIZE))
        top_mask_info = np.zeros((len(keep_inds), 12))
        top_mask_info[len(fg_inds):, :] = -1

        for i, val in enumerate(fg_inds):
            gt_box = scaled_gt_boxes[gt_assignment[val]]
            gt_box = np.around(gt_box).astype(int)
            ex_box = np.around(scaled_rois[i]).astype(int)
            gt_mask = gt_masks[gt_assignment[val]]
            gt_mask_info = mask_info[gt_assignment[val]]
            gt_mask = gt_mask[0:gt_mask_info[0], 0:gt_mask_info[1]]
            # regression targets is the intersection of bounding box and gt mask
            ex_mask = intersect_mask(ex_box, gt_box, gt_mask)
            pos_masks[i, ...] = ex_mask
            top_mask_info[i, 0] = gt_assignment[val]
            top_mask_info[i, 1] = gt_mask_info[0]
            top_mask_info[i, 2] = gt_mask_info[1]
            top_mask_info[i, 3] = labels[i]
            top_mask_info[i, 4:8] = ex_box
            top_mask_info[i, 8:12] = gt_box

        return labels, rois, fg_inds, keep_inds, pos_masks, top_mask_info, bbox_targets, bbox_inside_weights

    def forward_test(self, bottom, top):
        rois = bottom[0].data
        bbox_deltas = bottom[1].data
        # get ~ n * 4(1+c) new rois
        all_rois = bbox_transform_inv(rois[:, 1:5], bbox_deltas)
        scores = bottom[2].data
        im_info = bottom[3].data
        # get highest scored category's bounding box regressor
        score_max = scores.argmax(axis=1)
        rois_out = np.zeros((rois.shape[0], 5))
        # Single batch training
        rois_out[:, 0] = 0
        for i in xrange(len(score_max)):
            rois_out[i, 1:5] = all_rois[i, 4*score_max[i]:4*(score_max[i]+1)]
        rois_out[:, 1:5], _ = clip_boxes_mnc(rois_out[:, 1:5], im_info[0, :2])
        blobs = {
            'rois': rois_out
        }
        return blobs

def _get_bbox_regression_labels(bbox_target_data, num_classes):
    """Bounding-box regression targets (bbox_target_data) are stored in a
    compact form N x (class, tx, ty, tw, th)

    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets).

    Returns:
        bbox_target (ndarray): N x 4K blob of regression targets
        bbox_inside_weights (ndarray): N x 4K blob of loss weights
    """

    assert bbox_target_data.shape[1] == 5
    clss = bbox_target_data[:, 0]
    bbox_targets = np.zeros((clss.size, 4 * num_classes), dtype=np.float32)
    bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
    inds = np.where(clss > 0)[0]
    for ind in inds:
        cls = int(clss[ind])
        start = 4 * cls
        end = start + 4
        bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
        bbox_inside_weights[ind, start:end] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS
    return bbox_targets, bbox_inside_weights

def _compute_targets(ex_rois, gt_rois, labels):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 4

    targets = bbox_transform(ex_rois, gt_rois)
    if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
        # Optionally normalize targets by a precomputed mean and stdev
        targets = ((targets - np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS))
                / np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS))
    return np.hstack(
            (labels[:, np.newaxis], targets)).astype(np.float32, copy=False)
