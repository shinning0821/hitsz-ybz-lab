# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from fcos_core.structures.image_list import to_image_list


class BatchCollator(object):
    """
    From a list of samples from the dataset,
    returns the batched images and targets.
    This should be passed to the DataLoader
    """

    def __init__(self, size_divisible=0):
        self.size_divisible = size_divisible

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        images = to_image_list(transposed_batch[0], self.size_divisible)
        targets = transposed_batch[1]
        masks = transposed_batch[2]
        img_ids = transposed_batch[3]
        pt_map = transposed_batch[4]
        centerness = transposed_batch[5]
        bkg = transposed_batch[6]
        obj = transposed_batch[7]
        return images, targets, masks, img_ids, pt_map, centerness, bkg, obj


class BBoxAugCollator(object):
    """
    From a list of samples from the dataset,
    returns the images and targets.
    Images should be converted to batched images in `im_detect_bbox_aug`
    """

    def __call__(self, batch):
        return list(zip(*batch))
