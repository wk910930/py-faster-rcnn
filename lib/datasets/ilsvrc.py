# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2016 Sensetime, CUHK
# Written by Yang Bin, Wang Kun
# --------------------------------------------------------

import  os
import cPickle
import uuid
import scipy.io as sio
import scipy.sparse
import numpy as np
import xml.etree.ElementTree as ET
from datasets.imdb import imdb
from fast_rcnn.config import cfg

class ilsvrc(imdb):
    """ ILSVRC """
    def __init__(self, image_set, year, devkit_path=None):
        imdb.__init__(self, 'ilsvrc_' + year + '_' + image_set)
        # ILSVRC specific config options
        self.config = {'top_k' : 2000,
                       'use_salt' : True,
                       'cleanup' : True}
        # name, paths
        self._year = year
        self._image_set = image_set
        self._devkit_path = self._get_default_path() if devkit_path is None \
                                         else devkit_path
        self._data_path = os.path.join(self._devkit_path, 'ILSVRC' + self._year)
        self._classes = ('__background__',  # always index 0
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
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        self.set_proposal_method('slide')
        self.competition_mode(False)
        # Dataset splits that have ground-truth annotations (test splits
        # do not have gt annotations)
        self._gt_splits = ('trainval', 'val1')

        assert os.path.exists(self._devkit_path), \
                'ILSVRCdevkit path does not exist: {}'.format(self._devkit_path)
        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = os.path.join(self._data_path, 'Data', 'DET',
                                  index + '.JPEG')
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def _get_default_path(self):
        """
        Return the default path where ILSVRC is expected to be installed.
        """
        return os.path.join(cfg.DATA_DIR, 'ILSVRCdevkit' + self._year)

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._devkit_path + /ILSVRCdevkit2013/ILSVRC2013/ImageSets/Main/val2.txt
        image_set_file = os.path.join(self._data_path, 'ImageSets', 'Main',
                                      self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index

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

        gt_roidb = [self._load_ilsvrc_annotation(index)
                    for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb

    def slide_roidb(self):
        """
        Return the database of regions of interest.
        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_slide_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} slide roidb loaded from {}'.format(self.name, cache_file)
            return roidb
        if self._image_set in self._gt_splits:
            gt_roidb = self.gt_roidb()
            slide_roidb = self._load_slide_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, slide_roidb)
        else:
            roidb = self._load_slide_roidb(None)
        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote slide roidb to {}'.format(cache_file)
        return roidb

    def _load_slide_roidb(self, gt_roidb):
        box_list = []
        filename = os.path.abspath(os.path.join(self.cache_path, '..', 'slide_data', self.name + '.mat'))
        assert os.path.exists(filename), 'Slide anchor data not found at: {}'.format(filename)
        raw_data = sio.loadmat(filename)['boxes'].ravel()

        for i in xrange(raw_data.shape[0]):
            box_list.append(raw_data[i][:, 0:4] - 1)

        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_ilsvrc_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        filename = os.path.join(self._data_path, 'Annotations', index + '.xml')
        tree = ET.parse(filename)
        objs = tree.findall('object')
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            cls = self._class_to_ind[obj.find('name').text.lower().strip()]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False,
                'seg_areas' : seg_areas}

    def _write_ilsvrc_results_file(self, all_boxes, res_file):
        with open(res_file, 'wt') as f:
            for im_ind, index in enumerate(self.image_index):
                for cls_ind, cls in enumerate(self.classes):
                    if cls == '__background__':
                        continue
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    keep_inds = np.where(dets[:, -1] >= 0.01)[0]
                    dets = dets[keep_inds, :]
                    # Expects 1-based indices
                    for k in xrange(dets.shape[0]):
                        f.write('{:d} {:d} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(im_ind + 1, cls_ind, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))

    def evaluate_detections(self, all_boxes, output_dir):
        res_file = os.path.join(output_dir, ('detections_' +
                                         self._image_set +
                                         self._year +
                                         '_results'))
        if self.config['use_salt']:
            res_file += '_{}'.format(str(uuid.uuid4()))
        res_file += '.txt'
        self._write_ilsvrc_results_file(all_boxes, res_file)
        # Optionally cleanup results txt file
        if self.config['cleanup']:
            os.remove(res_file)

    def competition_mode(self, on):
        if on:
            self.config['use_salt'] = False
            self.config['cleanup'] = False
        else:
            self.config['use_salt'] = True
            self.config['cleanup'] = True
