# --------------------------------------------------------
# Multitask Network Cascade
# Modified from py-faster-rcnn (https://github.com/rbgirshick/py-faster-rcnn)
# Copyright (c) 2016, Haozhi Qi
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

from fast_rcnn.config import cfg
from gpu_nms import gpu_nms
from cpu_nms import cpu_nms


def nms(dets, thresh):
    """Dispatch to either CPU or GPU NMS implementations."""

    if dets.shape[0] == 0:
        return []
    if cfg.USE_GPU_NMS:
        return gpu_nms(dets, thresh, device_id=cfg.GPU_ID)
    else:
        return cpu_nms(dets, thresh)

def apply_nms_mask_single(box, mask, thresh):
    if box == []:
        return box, mask
    keep = nms(box, thresh)
    if len(keep) == 0:
        return box, mask
    return box[keep, :].copy(), mask[keep, :].copy()
