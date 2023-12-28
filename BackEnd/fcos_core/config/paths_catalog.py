# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""Centralized catalog of paths."""

import os
from fcos_core.config import cfg

class DatasetCatalog(object):
    DATA_DIR = "datasets"
    DATASETS = {
        'consep_train1': {
            "data_dir": "Train",
            "split": "train"
        },

        'consep_test1': {
            "data_dir": "Test",
            "split": "test"
        },

        'CoNSeP_point_train0': {
            "data_dir": "total",
            "split": "train0"
        },

        'CoNSeP_point_val0': {
            "data_dir": "total",
            "split": "val0"
        },

        'CoNSeP_point_test0': {
            "data_dir": "total",
            "split": "test0"
        },

        'CoNSeP_point_train1': {
            "data_dir": "total",
            "split": "train1"
        },

        'CoNSeP_point_val1': {
            "data_dir": "total",
            "split": "val1"
        },

        'CoNSeP_point_test1': {
            "data_dir": "total",
            "split": "test1"
        },

        'CoNSeP_point_train2': {
            "data_dir": "total",
            "split": "train2"
        },

        'CoNSeP_point_val2': {
            "data_dir": "total",
            "split": "val2"
        },

        'CoNSeP_point_test2': {
            "data_dir": "total",
            "split": "test2"
        },

        'CoNSeP_point_train3': {
            "data_dir": "total",
            "split": "train3"
        },

        'CoNSeP_point_val3': {
            "data_dir": "total",
            "split": "val3"
        },

        'CoNSeP_point_test3': {
            "data_dir": "total",
            "split": "test3"
        },

        'CoNSeP_point_train4': {
            "data_dir": "total",
            "split": "train4"
        },

        'CoNSeP_point_val4': {
            "data_dir": "total",
            "split": "val4"
        },
        
        'CoNSeP_point_test4': {
            "data_dir": "total",
            "split": "test4"
        },

        'CryoNuSeg_train1': {
            "data_dir": "train",
            "split": "train"
        },

        'CryoNuSeg_valid1': {
            "data_dir": "valid",
            "split": "valid"
        },

        'CryoNuSeg_test1': {
            "data_dir": "test",
            "split": "test"
        },

        'CryoNuSeg_point_train0': {
            "data_dir": "total",
            "split": "train0"
        },

        'CryoNuSeg_point_test0': {
            "data_dir": "total",
            "split": "test0"
        },

        'CryoNuSeg_point_train1': {
            "data_dir": "total",
            "split": "train1"
        },

        'CryoNuSeg_point_test1': {
            "data_dir": "total",
            "split": "test1"
        },

        'CryoNuSeg_point_train2': {
            "data_dir": "total",
            "split": "train2"
        },

        'CryoNuSeg_point_test2': {
            "data_dir": "total",
            "split": "test2"
        },

        'CryoNuSeg_point_train3': {
            "data_dir": "total",
            "split": "train3"
        },

        'CryoNuSeg_point_test3': {
            "data_dir": "total",
            "split": "test3"
        },

        'CryoNuSeg_point_train4': {
            "data_dir": "total",
            "split": "train4"
        },

        'CryoNuSeg_point_test4': {
            "data_dir": "total",
            "split": "test4"
        },


        'TNBC_point_train0': {
            "data_dir": "total",
            "split": "train0"
        },

        'TNBC_point_val0': {
            "data_dir": "total",
            "split": "val0"
        },

        'TNBC_point_test0': {
            "data_dir": "total",
            "split": "test0"
        },

        'TNBC_point_train1': {
            "data_dir": "total",
            "split": "train1"
        },

        'TNBC_point_val1': {
            "data_dir": "total",
            "split": "val1"
        },

        'TNBC_point_test1': {
            "data_dir": "total",
            "split": "test1"
        },

        'TNBC_point_train2': {
            "data_dir": "total",
            "split": "train2"
        },

        'TNBC_point_val2': {
            "data_dir": "total",
            "split": "val2"
        },

        'TNBC_point_test2': {
            "data_dir": "total",
            "split": "test2"
        },

        'TNBC_point_train3': {
            "data_dir": "total",
            "split": "train3"
        },

        'TNBC_point_val3': {
            "data_dir": "total",
            "split": "val3"
        },

        'TNBC_point_test3': {
            "data_dir": "total",
            "split": "test3"
        },

        'TNBC_point_train4': {
            "data_dir": "total",
            "split": "train4"
        },

        'TNBC_point_val4': {
            "data_dir": "total",
            "split": "val4"
        },
        
        'TNBC_point_test4': {
            "data_dir": "total",
            "split": "test4"
        },
    }

    @staticmethod
    def get(name):
        if "coco" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs["img_dir"]),
                ann_file=os.path.join(data_dir, attrs["ann_file"]),
            )
            return dict(
                factory="COCODataset",
                args=args,
            )


        elif "lym_train" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(data_dir, attrs["data_dir"]),
                data_name = cfg.DATASETS.NAME,
                split=attrs["split"],
            )
            return dict(
                factory="LYMDataset",
                args=args,
            )


        elif "lym_test" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(data_dir, attrs["data_dir"]),
                data_name=cfg.DATASETS.NAME,
                split=attrs["split"],
            )
            return dict(
                factory="LYMTestDataset",
                args=args,
            )

        elif "consep_train" in name:
            data_dir = '/data114_1/wzy/AAAI23/dataset/CoNSeP'
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(data_dir, attrs["data_dir"]),
                data_name='',
                split=attrs["split"],
            )
            return dict(
                factory="CoNSePDataset",
                args=args,
            )


        elif "consep_test" in name:
            data_dir = '/data114_1/wzy/AAAI23/dataset/CoNSeP'
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(data_dir, attrs["data_dir"]),
                data_name='',
                split=attrs["split"],
            )
            return dict(
                factory="CoNSePTestDataset",
                args=args,
            )

        elif "CoNSeP_point_train" in name:
            data_dir = '/data114_1/wzy/AAAI23/dataset/CoNSeP'
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(data_dir, attrs["data_dir"]),
                data_name='',
                split=attrs["split"],
            )
            return dict(
                factory="PointDataset",
                args=args,
            )

        elif ("CoNSeP_point_val") in name or ("CoNSeP_point_test") in name:
            data_dir = '/data114_1/wzy/AAAI23/dataset/CoNSeP'
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(data_dir, attrs["data_dir"]),
                data_name='',
                split=attrs["split"],
            )
            return dict(
                factory="PointTestDataset",
                args=args,
            )


        elif "TNBC_point_train" in name:
            data_dir = '/data114_1/wzy/AAAI23/dataset/TNBC'
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(data_dir, attrs["data_dir"]),
                data_name='',
                split=attrs["split"],
            )
            return dict(
                factory="PointDataset",
                args=args,
            )

        elif ("TNBC_point_val") in name or ("TNBC_point_test") in name:
            data_dir = '/data114_1/wzy/AAAI23/dataset/TNBC'
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(data_dir, attrs["data_dir"]),
                data_name='',
                split=attrs["split"],
            )
            return dict(
                factory="PointTestDataset",
                args=args,
            )

        elif "CryoNuSeg_train" in name:
            data_dir = '/data114_1/wzy/AAAI23/dataset/CryoNuSeg'
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(data_dir, attrs["data_dir"]),
                data_name='',
                split=attrs["split"],
            )
            return dict(
                factory="CoNSePDataset",
                args=args,
            )


        elif "CryoNuSeg_valid" in name:
            data_dir = '/data114_1/wzy/AAAI23/dataset/CryoNuSeg'
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(data_dir, attrs["data_dir"]),
                data_name='',
                split=attrs["split"],
            )
            return dict(
                factory="CoNSePTestDataset",
                args=args,
            )

        elif "CryoNuSeg_test" in name:
            data_dir = '/data114_1/wzy/AAAI23/dataset/CryoNuSeg'
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(data_dir, attrs["data_dir"]),
                data_name='',
                split=attrs["split"],
            )
            return dict(
                factory="CoNSePTestDataset",
                args=args,
            )

        elif "CryoNuSeg_point_train" in name:
            data_dir = '/data114_1/wzy/AAAI23/dataset/CryoNuSeg'
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(data_dir, attrs["data_dir"]),
                data_name='',
                split=attrs["split"],
            )
            return dict(
                factory="PointDataset",
                args=args,
            )

        elif "CryoNuSeg_point_test" in name:
            data_dir = '/data114_1/wzy/AAAI23/dataset/CryoNuSeg'
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(data_dir, attrs["data_dir"]),
                data_name='',
                split=attrs["split"],
            )
            return dict(
                factory="PointTestDataset",
                args=args,
            )

        elif "voc" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                data_dir=os.path.join(data_dir, attrs["data_dir"]),
                split=attrs["split"],
            )
            return dict(
                factory="PascalVOCDataset",
                args=args,
            )
        raise RuntimeError("Dataset not available: {}".format(name))


class ModelCatalog(object):
    S3_C2_DETECTRON_URL = "https://dl.fbaipublicfiles.com/detectron"
    C2_IMAGENET_MODELS = {
        "MSRA/R-50": "ImageNetPretrained/MSRA/R-50.pkl",
        "MSRA/R-50-GN": "ImageNetPretrained/47261647/R-50-GN.pkl",
        "MSRA/R-101": "ImageNetPretrained/MSRA/R-101.pkl",
        "MSRA/R-101-GN": "ImageNetPretrained/47592356/R-101-GN.pkl",
        "FAIR/20171220/X-101-32x8d": "ImageNetPretrained/20171220/X-101-32x8d.pkl",
        "FAIR/20171220/X-101-64x4d": "ImageNetPretrained/20171220/X-101-64x4d.pkl",
    }

    C2_DETECTRON_SUFFIX = "output/train/{}coco_2014_train%3A{}coco_2014_valminusminival/generalized_rcnn/model_final.pkl"
    C2_DETECTRON_MODELS = {
        "35857197/e2e_faster_rcnn_R-50-C4_1x": "01_33_49.iAX0mXvW",
        "35857345/e2e_faster_rcnn_R-50-FPN_1x": "01_36_30.cUF7QR7I",
        "35857890/e2e_faster_rcnn_R-101-FPN_1x": "01_38_50.sNxI7sX7",
        "36761737/e2e_faster_rcnn_X-101-32x8d-FPN_1x": "06_31_39.5MIHi1fZ",
        "35858791/e2e_mask_rcnn_R-50-C4_1x": "01_45_57.ZgkA7hPB",
        "35858933/e2e_mask_rcnn_R-50-FPN_1x": "01_48_14.DzEQe4wC",
        "35861795/e2e_mask_rcnn_R-101-FPN_1x": "02_31_37.KqyEK4tT",
        "36761843/e2e_mask_rcnn_X-101-32x8d-FPN_1x": "06_35_59.RZotkLKI",
        "37129812/e2e_mask_rcnn_X-152-32x8d-FPN-IN5k_1.44x": "09_35_36.8pzTQKYK",
        # keypoints
        "37697547/e2e_keypoint_rcnn_R-50-FPN_1x": "08_42_54.kdzV35ao"
    }

    @staticmethod
    def get(name):
        if name.startswith("Caffe2Detectron/COCO"):
            return ModelCatalog.get_c2_detectron_12_2017_baselines(name)
        if name.startswith("ImageNetPretrained"):
            return ModelCatalog.get_c2_imagenet_pretrained(name)
        raise RuntimeError("model not present in the catalog {}".format(name))

    @staticmethod
    def get_c2_imagenet_pretrained(name):
        prefix = ModelCatalog.S3_C2_DETECTRON_URL
        name = name[len("ImageNetPretrained/"):]
        name = ModelCatalog.C2_IMAGENET_MODELS[name]
        url = "/".join([prefix, name])
        return url

    @staticmethod
    def get_c2_detectron_12_2017_baselines(name):
        # Detectron C2 models are stored following the structure
        # prefix/<model_id>/2012_2017_baselines/<model_name>.yaml.<signature>/suffix
        # we use as identifiers in the catalog Caffe2Detectron/COCO/<model_id>/<model_name>
        prefix = ModelCatalog.S3_C2_DETECTRON_URL
        dataset_tag = "keypoints_" if "keypoint" in name else ""
        suffix = ModelCatalog.C2_DETECTRON_SUFFIX.format(dataset_tag, dataset_tag)
        # remove identification prefix
        name = name[len("Caffe2Detectron/COCO/"):]
        # split in <model_id> and <model_name>
        model_id, model_name = name.split("/")
        # parsing to make it match the url address from the Caffe2 models
        model_name = "{}.yaml".format(model_name)
        signature = ModelCatalog.C2_DETECTRON_MODELS[name]
        unique_name = ".".join([model_name, signature])
        url = "/".join([prefix, model_id, "12_2017_baselines", unique_name, suffix])
        return url
