# --------------------------------------------------------
# Multitask Network Cascade
# Written by Haozhi Qi
# Copyright (c) 2016, Haozhi Qi
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import os
import scipy
import numpy as np
import cPickle

from datasets.imdb import imdb
from fast_rcnn.config import cfg
from datasets.voc_seg_eval import voc_eval_sds

class sbd(imdb):
    """
    Semantic Boundaries Dataset

    A subclass for datasets.imdb.imdb
    This class contains information of ROIDB and MaskDB
    This class implements roidb and maskdb related functions
    """
    def __init__(self, image_set, year, devkit_path=None):
        imdb.__init__(self, 'sbd_' + year + '_' + image_set)
        self._year = year
        self._image_set = image_set
        self._devkit_path = self._get_default_path() if devkit_path is None \
                            else devkit_path
        self._data_path = os.path.join(self._devkit_path, 'dataset')
        self._classes = ('__background__', # always index 0
                         'aeroplane', 'bicycle', 'bird', 'boat',
                         'bottle', 'bus', 'car', 'cat', 'chair',
                         'cow', 'diningtable', 'dog', 'horse',
                         'motorbike', 'person', 'pottedplant',
                         'sheep', 'sofa', 'train', 'tvmonitor')
        self._image_ext = '.jpg'
        self._image_index = self._load_image_set_index()

        assert os.path.exists(self._devkit_path), \
                'SBDdevkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        image_path = os.path.join(self._data_path, 'img',
            self._image_index[i] + self._image_ext)
        assert os.path.exists(image_path), \
            'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._devkit_path + /SBDdevkit/dataset/val.txt
        image_set_file = os.path.join(self._data_path, self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index

    def _get_default_path(self):
        """
        Return the default path where SBD is expected to be installed.
        """
        return os.path.join(cfg.DATA_DIR, 'SBDdevkit')

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = [self._load_sbd_annotations(index)
                    for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def gt_maskdb(self):
        """
        Return the database of ground-truth masks.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_maskdb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                gt_maskdb = cPickle.load(fid)
            print '{} gt maskdb loaded from {}'.format(self.name, cache_file)
            return gt_maskdb

        gt_roidb = self.gt_roidb()
        gt_maskdb = [self._load_sbd_mask_annotations(index, gt_roidb[i])
                     for i, index in enumerate(self.image_index)]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_maskdb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_maskdb

    def _load_sbd_annotations(self, index):
        """
        Load bounding-box info from MAT files in the SBD format.
        """

        inst_file_name = os.path.join(self._data_path, 'inst', index + '.mat')
        gt_inst_mat = scipy.io.loadmat(inst_file_name)
        gt_inst_data = gt_inst_mat['GTinst']['Segmentation'][0][0]
        unique_inst = np.unique(gt_inst_data)
        background_ind = np.where(unique_inst == 0)[0]
        unique_inst = np.delete(unique_inst, background_ind)

        cls_file_name = os.path.join(self._data_path, 'cls', index + '.mat')
        gt_cls_mat = scipy.io.loadmat(cls_file_name)
        gt_cls_data = gt_cls_mat['GTcls']['Segmentation'][0][0]

        num_objs = len(unique_inst)
        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros(num_objs, dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)

        for ind, inst_mask in enumerate(unique_inst):
            im_mask = (gt_inst_data == inst_mask)
            im_cls_mask = np.multiply(gt_cls_data, im_mask)
            unique_cls_inst = np.unique(im_cls_mask)
            background_ind = np.where(unique_cls_inst == 0)[0]
            unique_cls_inst = np.delete(unique_cls_inst, background_ind)
            assert len(unique_cls_inst) == 1
            gt_classes[ind] = unique_cls_inst[0]
            [r, c] = np.where(im_mask > 0)
            boxes[ind, 0] = np.min(c)
            boxes[ind, 1] = np.min(r)
            boxes[ind, 2] = np.max(c)
            boxes[ind, 3] = np.max(r)
            overlaps[ind, unique_cls_inst[0]] = 1.0

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes': boxes,
                'gt_classes': gt_classes,
                'gt_overlaps': overlaps,
                'flipped': False}

    def _load_sbd_mask_annotations(self, index, gt_roidb):
        """
        Load gt_masks information from SBD's additional data
        """

        inst_file_name = os.path.join(self._data_path, 'inst', index + '.mat')
        gt_inst_mat = scipy.io.loadmat(inst_file_name)
        gt_inst_data = gt_inst_mat['GTinst']['Segmentation'][0][0]
        unique_inst = np.unique(gt_inst_data)
        background_ind = np.where(unique_inst == 0)[0]
        unique_inst = np.delete(unique_inst, background_ind)

        cls_file_name = os.path.join(self._data_path, 'cls', index + '.mat')
        gt_cls_mat = scipy.io.loadmat(cls_file_name)
        gt_cls_data = gt_cls_mat['GTcls']['Segmentation'][0][0]

        gt_masks = []
        for ind, inst_mask in enumerate(unique_inst):
            box = gt_roidb['boxes'][ind]
            cls_ind = gt_roidb['gt_classes'][ind]

            im_mask = (gt_inst_data == inst_mask)
            im_cls_mask = np.multiply(gt_cls_data, im_mask)

            unique_cls_inst = np.unique(im_cls_mask)
            background_ind = np.where(unique_cls_inst == 0)[0]
            unique_cls_inst = np.delete(unique_cls_inst, background_ind)

            assert len(unique_cls_inst) == 1
            assert unique_cls_inst[0] == cls_ind

            mask = im_mask[box[1]:box[3]+1, box[0]:box[2]+1]
            gt_masks.append(mask)

        # Also record the maximum dimension to create fixed dimension array
        # when do forwarding
        mask_max_x = max(gt_masks[i].shape[1] for i in xrange(len(gt_masks)))
        mask_max_y = max(gt_masks[i].shape[0] for i in xrange(len(gt_masks)))

        return {'gt_masks': gt_masks,
                'mask_max': [mask_max_x, mask_max_y],
                'flipped': False}

    def append_flipped_masks(self):
        num_images = self.num_images
        for i in xrange(num_images):
            masks = self.maskdb[i]['gt_masks']
            masks_flip = []
            for mask_ind in xrange(len(masks)):
                mask_flip = np.fliplr(masks[mask_ind])
                masks_flip.append(mask_flip)
            entry = {'gt_masks': masks_flip,
                     'mask_max': self.maskdb[i]['mask_max'],
                     'flipped': True}
            self.maskdb.append(entry)
        self._image_index = self._image_index * 2

    def _reformat_result(self, masks, thresh):
        """
        Reformat masks to be binary (0/1) and in shape of (n, sz, sz)
        """

        num_images = self.num_images
        num_classes = self.num_classes
        reformat_masks = [[[] for _ in xrange(num_images)]
                          for _ in xrange(num_classes)]

        for cls_inds in xrange(1, num_classes):
            for img_inds in xrange(num_images):
                if len(masks[cls_inds][img_inds]) == 0:
                    continue
                num_inst = masks[cls_inds][img_inds].shape[0]
                # Reshape
                reformat_masks[cls_inds][img_inds] = \
                    masks[cls_inds][img_inds].reshape(num_inst, cfg.MASK_SIZE, cfg.MASK_SIZE)
                # Binarize
                reformat_masks[cls_inds][img_inds] = \
                    reformat_masks[cls_inds][img_inds] >= thresh

        return reformat_masks

    def _write_voc_seg_results_file(self, all_boxes, all_masks, output_dir):
        """
        Write results as a pkl file.
        Notice: This is different from detection task
        since it is difficult to write masks to txt
        """

        # Always reformat result in case that masks are not binary (0/1)
        # or is in shape (n, sz*sz) instead of (n, sz, sz)
        all_masks = self._reformat_result(all_masks, cfg.BINARIZE_THRESH)
        for cls_inds, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print 'Writing {} VOC results file'.format(cls)
            # Detection
            filename = os.path.join(output_dir, cls + '_det.pkl')
            with open(filename, 'wr') as f:
                cPickle.dump(all_boxes[cls_inds], f, cPickle.HIGHEST_PROTOCOL)
            # Segmentation
            filename = os.path.join(output_dir, cls + '_seg.pkl')
            with open(filename, 'wr') as f:
                cPickle.dump(all_masks[cls_inds], f, cPickle.HIGHEST_PROTOCOL)

    def _do_python_eval(self, output_dir):
        imageset_file = os.path.join(
            self._data_path, self._image_set + '.txt')
        cache_dir = os.path.join(self._devkit_path, 'annotations_cache')

        # Define this as true according to SDS's evaluation protocol
        use_07_metric = True
        print 'VOC07 metric? ' + ('Yes' if use_07_metric else 'No')
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)

        print '~~~~~~ Evaluation use min overlap = 0.5 ~~~~~~'
        aps = []
        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            det_filename = os.path.join(output_dir, cls + '_det.pkl')
            seg_filename = os.path.join(output_dir, cls + '_seg.pkl')
            ap = voc_eval_sds(det_filename, seg_filename, self._data_path,
                              imageset_file, cls, cache_dir,
                              self._classes, ov_thresh=0.5)
            aps += [ap]
            print 'AP for {} = {:.2f}'.format(cls, ap*100)
        print 'Mean AP@0.5 = {:.2f}'.format(np.mean(aps)*100)

        print '~~~~~~ Evaluation use min overlap = 0.7 ~~~~~~'
        aps = []
        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            det_filename = os.path.join(output_dir, cls + '_det.pkl')
            seg_filename = os.path.join(output_dir, cls + '_seg.pkl')
            ap = voc_eval_sds(det_filename, seg_filename, self._data_path,
                              imageset_file, cls, cache_dir,
                              self._classes, ov_thresh=0.7)
            aps += [ap]
            print 'AP for {} = {:.2f}'.format(cls, ap*100)
        print 'Mean AP@0.7 = {:.2f}'.format(np.mean(aps)*100)

        print('~~~~~~~~')
        print('')
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** Python eval code.')
        print('-- Thanks, The Management')
        print('--------------------------------------------------------------')

    def evaluate_segmentation(self, all_boxes, all_masks, output_dir):
        self._write_voc_seg_results_file(all_boxes, all_masks, output_dir)
        self._do_python_eval(output_dir)
