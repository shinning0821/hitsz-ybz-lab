import os

import torch
import torch.utils.data
from PIL import Image
import numpy as np
import sys
import cv2
from torchvision.transforms import functional as F

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


from fcos_core.structures.bounding_box import BoxList
from pycocotools.coco import COCO



class PointDataset(torch.utils.data.Dataset):

    CLASSES = (
        "__background__ ",
        "NuClei",
    )

    def __init__(self, data_dir, data_name, split, use_difficult=False,transforms=None):
        self.data_dir = data_dir
        self.data_name = data_name
        self.split = split
        self.root = data_dir
        self.image_set = split
        self.keep_difficult = use_difficult
        self.transforms = transforms
        image_sets_file = os.path.join(self.data_dir, "%s.txt" % (self.split))
        self.ids = PointDataset._read_image_ids(image_sets_file)
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        cls = PointDataset.CLASSES
        self.class_to_ind = dict(zip(cls, range(len(cls))))
        self.coco = COCO(os.path.join(self.data_dir, 'pl_annotation.json'))

    def __getitem__(self, index):
        img_id = self.ids[index]
        img = self._read_image(img_id)
        pt_map = self._read_pt_map(img_id)
        centerness = self._read_centerness(img_id)
        bkg = self._read_bkg(img_id)
        obj = self._read_obj(img_id)
        mask = self.read_gt(img_id)
        target = self.get_groundtruth(img_id)
        target = target.clip_to_image(remove_empty=True)
        if self.transforms is not None:
            img, target, mask, pt_map, centerness,bkg,obj = self.transforms(img, target, mask, pt_map, centerness,bkg,obj)


        return img, target, mask, index, pt_map, centerness, bkg, obj


    def __len__(self):
        return len(self.ids)

    @staticmethod
    def _read_image_ids(image_sets_file):
        ids = []
        with open(image_sets_file) as f:
            for line in f:
                ids.append(line.rstrip())
        return ids

    def _read_image(self, image_id):
        image_file = os.path.join(self.data_dir, "Images", "%s.png" % image_id)
        image = Image.open(image_file).convert("RGB")
        image = np.array(image)
        return image

    def read_gt(self, image_id):    # gt, 用作对比实验，不用于弱监督训练
        gt_file = os.path.join(self.data_dir, 'Masks', "%s.png" % image_id)
        gt = cv2.imread(gt_file, -1)
        gt = np.where(gt > 0, 1, gt)    # 直接变成一个二分类任务
        gt = np.array(gt, np.float32)
        gt = np.expand_dims(gt, axis=2)
        return gt

    def _read_pt_map(self, image_id):
        pt_file = os.path.join(self.data_dir, 'Points',"%s.png" % image_id)
        pt = cv2.imread(pt_file, cv2.IMREAD_GRAYSCALE)
        pt = np.array(pt, np.float32)
        pt = np.expand_dims(pt, axis=2)
        pt = pt/255.0
        return pt

    def _read_centerness(self, image_id):
        centerness_file = os.path.join(self.data_dir, 'centerness', "%s.png" % image_id)
        centerness = cv2.imread(centerness_file, cv2.IMREAD_GRAYSCALE)
        centerness = np.array(centerness, np.float32)
        centerness = np.expand_dims(centerness, axis=2)
        return centerness

    def _read_bkg(self, image_id):
        bkg_file = os.path.join(self.data_dir, 'Bkgs', "%s.png" % image_id)
        bkg = cv2.imread(bkg_file, -1)
        bkg = np.array(bkg, np.float32)
        bkg = np.expand_dims(bkg, axis=2)
        return bkg

    def _read_obj(self, image_id):
        obj_file = os.path.join(self.data_dir, 'Objs', "%s.png" % image_id)
        obj = cv2.imread(obj_file, cv2.IMREAD_GRAYSCALE)
        obj = np.array(obj, np.float32)
        obj = np.expand_dims(obj, axis=2)
        return obj

    def get_groundtruth(self, image_id):
        anno = self._preprocess_annotation(image_id)
        height, width = anno["im_info"]
        target = BoxList(anno["boxes"], (width, height), mode="xyxy")
        target.add_field("labels", anno["labels"])
        return target

    def _preprocess_annotation(self, img_id):
        img_ids = []
        # 根据文件名获取图像的 img_id
        imgs = self.coco.loadImgs(self.coco.getImgIds())
        for img in imgs:
            if img['file_name'] == (img_id+'.png'):
                img_ids.append(img['id'])
        h = img['height']
        w = img['width']
        bboxes = []
        labels = []
        for img_id in img_ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            img_annotations = self.coco.loadAnns(ann_ids)
            for img_annotation in img_annotations:
                bboxes.append(img_annotation["bbox"])
                labels.append(1)
        if len(bboxes) == 0:
                bboxes.append([0, 0, 2, 2])
                labels.append(0)
        
        im_info = tuple(map(int, (h, w)))
        res = {
            "boxes": torch.tensor(bboxes, dtype=torch.float32),
            "labels": torch.tensor(labels),
            "im_info": im_info,
        }
        return res

    def get_img_info(self, index):
        img_id = self.ids[index]
        image_file = os.path.join(self.data_dir, "Images", "%s.png" % img_id)
        img = cv2.imread(image_file)
        h, w, _ = img.shape
        im_info = tuple(map(int, (h, w)))
        return {"height": im_info[0], "width": im_info[1], "id":img_id}

    def map_class_id_to_class_name(self, class_id):
        return PointDataset.CLASSES[class_id]


class PointTestDataset(torch.utils.data.Dataset):

    CLASSES = (
        "__background__ ",
        "NuClei",
    )

    def __init__(self, data_dir, data_name, split, use_difficult=False,transforms=None):
        self.data_dir = data_dir
        self.data_name = data_name
        self.split = split
        self.root = data_dir
        self.image_set = split
        self.keep_difficult = use_difficult
        self.transforms = transforms
        image_sets_file = os.path.join(self.data_dir, "%s.txt" % (self.split))
        self.ids = PointDataset._read_image_ids(image_sets_file)
        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}
        cls = PointDataset.CLASSES
        self.class_to_ind = dict(zip(cls, range(len(cls))))
        self.coco = COCO(os.path.join(self.data_dir, 'annotation.json'))

    def __getitem__(self, index):
        img_id = self.ids[index]
        img = self._read_image(img_id)
        pt_map = self._read_pt_map(img_id)
        centerness = self._read_centerness(img_id)
        bkg = self._read_bkg(img_id)
        obj = self._read_obj(img_id)
        mask = self.read_gt(img_id)
        target = self.get_groundtruth(img_id)
        target = target.clip_to_image(remove_empty=True)
        if self.transforms is not None:
            img, target, mask, pt_map, centerness,bkg,obj = self.transforms(img, target, mask, pt_map, centerness,bkg,obj)


        return img, target, mask, index, pt_map, centerness,bkg,obj


    def __len__(self):
        return len(self.ids)

    @staticmethod
    def _read_image_ids(image_sets_file):
        ids = []
        with open(image_sets_file) as f:
            for line in f:
                ids.append(line.rstrip())
        return ids

    def _read_image(self, image_id):
        image_file = os.path.join(self.data_dir, "Images", "%s.png" % image_id)
        image = Image.open(image_file).convert("RGB")
        image = np.array(image)
        return image

    def read_gt(self, image_id):    # gt, 用作对比实验，不用于弱监督训练
        gt_file = os.path.join(self.data_dir, 'Masks', "%s.png" % image_id)
        gt = cv2.imread(gt_file, -1)
        gt = np.where(gt > 0, 1, gt)    # 直接变成一个二分类任务
        gt = np.array(gt, np.float32)
        gt = np.expand_dims(gt, axis=2)
        return gt

    def _read_pt_map(self, image_id):
        pt_file = os.path.join(self.data_dir, 'Points',"%s.png" % image_id)
        pt = cv2.imread(pt_file, cv2.IMREAD_GRAYSCALE)
        pt = np.array(pt, np.float32)
        pt = np.expand_dims(pt, axis=2)
        pt = pt/255.0
        return pt

    def _read_centerness(self, image_id):
        centerness_file = os.path.join(self.data_dir, 'centerness', "%s.png" % image_id)
        centerness = cv2.imread(centerness_file, cv2.IMREAD_GRAYSCALE)
        centerness = np.array(centerness, np.float32)
        centerness = np.expand_dims(centerness, axis=2)
        return centerness

    def _read_bkg(self, image_id):
        bkg_file = os.path.join(self.data_dir, 'Bkgs', "%s.png" % image_id)
        bkg = cv2.imread(bkg_file, cv2.IMREAD_GRAYSCALE)
        bkg = np.array(bkg, np.float32)
        bkg = np.expand_dims(bkg, axis=2)
        return bkg

    def _read_obj(self, image_id):
        obj_file = os.path.join(self.data_dir, 'Objs', "%s.png" % image_id)
        obj = cv2.imread(obj_file, cv2.IMREAD_GRAYSCALE)
        obj = np.array(obj, np.float32)
        obj = np.expand_dims(obj, axis=2)
        return obj

    def get_groundtruth(self, image_id):
        anno = self._preprocess_annotation(image_id)
        height, width = anno["im_info"]
        target = BoxList(anno["boxes"], (width, height), mode="xyxy")
        target.add_field("labels", anno["labels"])
        return target

    def read_inst_map(self, image_id):    # gt, 用作对比实验，不用于弱监督训练
        gt_file = os.path.join(self.data_dir, 'Masks', "%s.png" % image_id)
        gt = cv2.imread(gt_file, -1)
        gt = np.array(gt, np.float32)
        gt = np.expand_dims(gt, axis=2)
        return gt


    def _preprocess_annotation(self, img_id):
        img_ids = []
        # 根据文件名获取图像的 img_id
        imgs = self.coco.loadImgs(self.coco.getImgIds())
        for img in imgs:
            if img['file_name'] == (img_id+'.png'):
                img_ids.append(img['id'])
        h = img['height']
        w = img['width']
        bboxes = []
        labels = []
        for img_id in img_ids:
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            img_annotations = self.coco.loadAnns(ann_ids)
            for img_annotation in img_annotations:
                bboxes.append(img_annotation["bbox"])
                labels.append(1)
        if len(bboxes) == 0:
                bboxes.append([0, 0, 2, 2])
                labels.append(0)
        
        im_info = tuple(map(int, (h, w)))
        res = {
            "boxes": torch.tensor(bboxes, dtype=torch.float32),
            "labels": torch.tensor(labels),
            "im_info": im_info,
        }
        return res

    def get_img_info(self, index):
        img_id = self.ids[index]
        image_file = os.path.join(self.data_dir, "Images", "%s.png" % img_id)
        img = cv2.imread(image_file)
        h, w, _ = img.shape
        im_info = tuple(map(int, (h, w)))
        return {"height": im_info[0], "width": im_info[1], "id":img_id}

    def map_class_id_to_class_name(self, class_id):
        return PointDataset.CLASSES[class_id]