from __future__ import division
import os
import logging
from collections import defaultdict
from datetime import datetime
import itertools
import numpy as np
import torch 
import torch.nn.functional as F
import six
from fcos_core.structures.bounding_box import BoxList
from fcos_core.structures.boxlist_ops import boxlist_iou
from PIL import Image
from tools.alignment import gen_inst_map
from tools.metrics import get_fast_aji, get_dice_2,get_fast_pq,get_fast_aji_plus

def do_consep_evaluation(dataset, predictions, output_folder, ovthresh):
    # for the user to choose
    pred_boxlists = []
    gt_boxlists = []

    pred_masks = []
    gt_masks = []

    inst_masks = []
    inst_gts = []

    for image_id, prediction in enumerate(predictions[0]):
        img_info = dataset.get_img_info(image_id)
        image_width = img_info["width"]
        image_height = img_info["height"]

        bbox = prediction[0]
        bbox = bbox.resize((image_width, image_height))
        pred_boxlists.append(bbox)

        gt_boxlist = dataset.get_groundtruth(img_info['id'])        
        gt_boxlists.append(gt_boxlist)
        
        mask = torch.tensor(prediction[1]).unsqueeze(0)
        mask = torch.sigmoid(mask)
        mask = F.interpolate(mask, scale_factor=2)
        mask = torch.argmax(mask,dim=1)
        mask = mask.squeeze(0).numpy().astype(np.uint8)

        inst_mask = gen_inst_map(bbox.bbox.numpy(),mask.copy())
        inst_masks.append(inst_mask.copy())
        inst_gts.append(dataset.read_inst_map(img_info['id']).copy())

        pred_masks.append(mask.copy())
        gt_masks.append(dataset.read_gt(img_info['id']).copy())
        
        
    result = eval_detection_consep(
        pred_boxlists=pred_boxlists,
        gt_boxlists=gt_boxlists,
        iou_thresh=ovthresh,
    )

    seg_result = eval_segmentation_consep(
        pred_masks=pred_masks,
        gt_masks=gt_masks,
    )

    if(result["f1"] > 0.2 ):
        instance_result = eval_instance_consep(
            inst_masks=inst_masks,
            inst_gts=inst_gts,
        )
        metrics1 = {'prec': result["prec"], 'rec': result["rec"], 'f1': result["f1"],'dice':seg_result["dice"]\
        ,'inst_dice':instance_result["dice"],'aji':instance_result["aji"],'dq':instance_result["dq"],'sq':instance_result["sq"],'pq':instance_result["pq"]}
    else:
        metrics1 = {'prec': result["prec"], 'rec': result["rec"], 'f1': result["f1"],'dice':seg_result["dice"]}

    print(metrics1)

    logger = logging.getLogger("DDTNet.inference")
    result_str = "f1: {:.4f}\n".format(result["f1"])
    logger.info(result_str)
    losses = predictions[1]
    loss = losses.loss.avg

    #DDTNet
    loss_cls = losses.loss_cls.avg
    loss_reg = losses.loss_reg.avg
    loss_centerness = losses.loss_centerness.avg
    loss_mask = losses.loss_mask.avg
    loss_str = "loss: {:.4f},loss_cls: {:.4f},loss_reg: {:.4f},loss_centerness: {:.4f},loss_mask: {:.4f}\n".format(loss, loss_cls,loss_reg,loss_centerness,loss_mask)
    metrics2 = {'loss': loss, 'loss_cls': loss_cls, 'loss_reg': loss_reg, 'loss_centerness':loss_centerness, 'loss_mask': loss_mask}


    logger.info(loss_str)

    return dict(metrics1=metrics1,metrics2=metrics2)




def eval_detection_consep(pred_boxlists, gt_boxlists, iou_thresh):
    """Evaluate on consep dataset.
    Args:
        pred_boxlists(list[BoxList]): pred boxlist, has labels and scores fields.
        gt_boxlists(list[BoxList]): ground truth boxlist, has labels field.
        iou_thresh: iou thresh
    Returns:
        dict represents the results
    """
    
    assert len(gt_boxlists) == len(
        pred_boxlists
    ), "Length of gt and pred lists need to be same."
    prec, rec, f1 = calc_detection_consep_prec_rec(
        pred_boxlists=pred_boxlists, gt_boxlists=gt_boxlists, iou_thresh=iou_thresh, gt_difficults=None
    )
    # 只需要知道前景框的准确率就好了
    return {"prec": np.nanmean(prec), "rec": np.nanmean(rec), "f1": np.nanmean(f1)}


def calc_detection_consep_prec_rec(gt_boxlists, pred_boxlists, iou_thresh=0.5, gt_difficults=None):
    """Calculate precision, recall and f1 based on evaluation code of PASCAL VOC.
    This function calculates precision, recall and f1 of
    predicted bounding boxes obtained from a dataset which has :math:`N`
    images.
    The code is based on the evaluation code used in PASCAL VOC Challenge.
   """
    n_pos = defaultdict(int)
    score = defaultdict(list)
    match = defaultdict(list)

    single_prec = []
    single_rec = []
    single_f1 = []

    for gt_boxlist, pred_boxlist in zip(gt_boxlists, pred_boxlists):
        single_result = []
        
        pred_bbox = pred_boxlist.bbox.numpy()
        pred_label = pred_boxlist.get_field("labels").numpy()
        pred_score = pred_boxlist.get_field("scores").numpy()
        gt_bbox = gt_boxlist.bbox.numpy()
        gt_label = gt_boxlist.get_field("labels").numpy()

        if gt_difficults is None:
            gt_difficult = np.zeros(gt_bbox.shape[0], dtype=bool)
        else:
            gt_difficult = gt_boxlist.get_field("difficult").numpy()

        for l in np.unique(np.concatenate((pred_label, gt_label)).astype(int)):            
            pred_mask_l = pred_label == l
            pred_bbox_l = pred_bbox[pred_mask_l]
            pred_score_l = pred_score[pred_mask_l]
            # sort by score
            order = pred_score_l.argsort()[::-1]
            pred_bbox_l = pred_bbox_l[order]
            pred_score_l = pred_score_l[order]
            
            gt_mask_l = gt_label == l
            gt_bbox_l = gt_bbox[gt_mask_l]
            gt_difficult_l = gt_difficult[gt_mask_l]

            n_pos[l] += np.logical_not(gt_difficult_l).sum()
            score[l].extend(pred_score_l)

            if len(pred_bbox_l) == 0:
                continue
            if len(gt_bbox_l) == 0:
                match[l].extend((0,) * pred_bbox_l.shape[0])
                continue

            # VOC evaluation follows integer typed bounding boxes.
            pred_bbox_l = pred_bbox_l.copy()
            pred_bbox_l[:, 2:] += 1
            gt_bbox_l = gt_bbox_l.copy()
            gt_bbox_l[:, 2:] += 1
            iou = boxlist_iou(
                BoxList(pred_bbox_l, gt_boxlist.size),
                BoxList(gt_bbox_l, gt_boxlist.size),
            ).numpy()

            gt_index = iou.argmax(axis=1)
            # print(pred_bbox_l.shape)
            # print(gt_bbox_l.shape)
            # print(iou.shape)
            # print(gt_index.shape)
            # set -1 if there is no matching ground truth
            gt_index[iou.max(axis=1) < iou_thresh] = -1
            del iou

            selec = np.zeros(gt_bbox_l.shape[0], dtype=bool)
            for gt_idx in gt_index:
                if gt_idx >= 0:
                    # if gt_difficult_l[gt_idx]:
                    #     match[l].append(-1)

                    if not selec[gt_idx]:
                        match[l].append(1)
                        single_result.append(1)
                    else:
                        match[l].append(0)
                        single_result.append(0)
                    selec[gt_idx] = True
                else:
                    match[l].append(0)
                    single_result.append(0)

            single_result = np.array(single_result, dtype=np.int8)
            single_tp = np.cumsum(single_result == 1)[-1]
            single_fp = np.cumsum(single_result == 0)[-1]
            
            temp_prec = single_tp/(single_fp+single_tp)
            temp_rec = single_tp/np.logical_not(gt_difficult_l).sum()
            single_prec.append(temp_prec)
            single_rec.append(temp_rec)
            single_f1.append(2*temp_prec*temp_rec/(temp_prec+temp_rec))
            


    n_fg_class = max(n_pos.keys()) + 1
    prec = [None] * n_fg_class
    rec = [None] * n_fg_class
    f1 = [None] * n_fg_class

    for l in n_pos.keys():
        score_l = np.array(score[l])
        match_l = np.array(match[l], dtype=np.int8)

        order = score_l.argsort()[::-1]
        match_l = match_l[order]

        tp = np.cumsum(match_l == 1)
        fp = np.cumsum(match_l == 0)
        # If an element of fp + tp is 0,
        # the corresponding element of prec[l] is nan.
        prec[l] = tp / (fp + tp)
        # If n_pos[l] is 0, rec[l] is None.
        if n_pos[l] > 0:
            rec[l] = tp / n_pos[l]
        f1[l] = 2 * prec[l] * rec[l] / (prec[l] + rec[l] + 1e-6)

    
    # return prec, rec, f1
    return np.array(single_prec),np.array(single_rec),np.array(single_f1)



def eval_segmentation_consep(pred_masks,gt_masks):
    """Evaluate on consep dataset.
    Args:
        pred_boxlists(list[BoxList]): pred boxlist, has labels and scores fields.
        gt_boxlists(list[BoxList]): ground truth boxlist, has labels field.
        iou_thresh: iou thresh
    Returns:
        dict represents the results
    """
    assert len(pred_masks) == len(
        gt_masks
    ), "Length of gt and pred lists need to be same."
    dice = calc_segmentation_consep_dice(
        pred_masks=pred_masks, gt_masks=gt_masks
    )
    # 只需要知道前景框的准确率就好了
    return {"dice": np.nanmean(dice)}


def calc_segmentation_consep_dice(pred_masks,gt_masks):
    dice_list = []
    for i in range(len(pred_masks)):
        pred_mask = pred_masks[i]
        gt_mask = gt_masks[i][:,:,0]
        # cls = 1
        # outputs_cls = pred_mask == cls
        # labels_cls = gt_mask == cls
        # intersection = (outputs_cls & labels_cls).sum()
        # union = (outputs_cls | labels_cls).sum()
        # A = outputs_cls.sum()
        # B = labels_cls.sum()
        # dice = 2*intersection/(A+B)
        smooth = 1e-5  # 平滑项，用于避免分母为0
        intersection = (pred_mask * gt_mask).sum()
        union = pred_mask.sum() + gt_mask.sum()
        dice = (2. * intersection + smooth) / (union + smooth)
        dice_list.append(dice)
    
    return np.array(dice_list)


def eval_instance_consep(inst_masks,inst_gts):
    """Evaluate on consep dataset.
    Args:
        pred_boxlists(list[BoxList]): pred boxlist, has labels and scores fields.
        gt_boxlists(list[BoxList]): ground truth boxlist, has labels field.
        iou_thresh: iou thresh
    Returns:
        dict represents the results
    """
    assert len(inst_masks) == len(
        inst_gts
    ), "Length of gt and pred lists need to be same."

    aji,dice,dq,sq,pq = calc_instance_consep_dice(
        inst_masks=inst_masks, inst_gts=inst_gts
    )
    # 只需要知道前景框的准确率就好了
    return {"aji": np.nanmean(aji), "dice": np.nanmean(dice), "dq": np.nanmean(dq), "sq": np.nanmean(sq), "pq": np.nanmean(pq)}


def calc_instance_consep_dice(inst_masks,inst_gts):
    aji_list = []
    dice_list = []
    dq_list = []
    sq_list = []
    pq_list = []

    for i in range(len(inst_masks)):
        inst_mask = inst_masks[i].astype(np.uint8)
        inst_gt = inst_gts[i][:,:,0].astype(np.uint8)

        true_id_list = list(np.unique(inst_gt))
        for i,id  in enumerate(true_id_list):
            if id == 0:
                continue
            else:
                inst_gt[inst_gt == id] = i

        pred_id_list = list(np.unique(inst_mask))
        for i,id  in enumerate(pred_id_list):
            if id == 0:
                continue
            else:
                inst_mask[inst_mask == id] = i
        
        
        aji = get_fast_aji_plus(inst_gt,inst_mask)
        dice2 = get_dice_2(inst_gt,inst_mask)
        dq,sq,pq = get_fast_pq(inst_gt,inst_mask,0.5)[0]

        aji_list.append(aji)    
        dice_list.append(dice2)
        dq_list.append(dq)
        sq_list.append(sq)
        pq_list.append(pq)

    return np.array(aji_list),np.array(dice_list),np.array(dq_list),np.array(sq_list),np.array(pq_list)