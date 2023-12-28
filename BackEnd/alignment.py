# 用来结合语义分割和目标检测的结果
import cv2
import numpy as np
import pandas as pd
import os
from skimage.morphology import remove_small_objects
from scipy.ndimage import binary_fill_holes
from skimage import morphology, measure


def deal_mask(mask):

    for sem_id in [0,1]:
            # 0 is background semantic class.
            if sem_id == 0:
                continue
            sem_id_mask = mask == sem_id
            # fill instance holes
            sem_id_mask = binary_fill_holes(sem_id_mask)
            # sem_id_mask = remove_small_objects(sem_id_mask, 5)
    return sem_id_mask.astype(np.uint8)
    


def gen_inst_map(bboxes,pred_mask): 
    pred_mask = deal_mask(pred_mask)
    Premask = pred_mask
    pre0 = np.zeros_like(Premask)
    for i,bbox in enumerate(bboxes):
        pre = np.zeros_like(Premask)
        x1 = round(bbox[0])
        x2 = round(bbox[2])
        y1 = round(bbox[1])
        y2 = round(bbox[3])
        
        if x1 < 0: x1=0
        if x2 >255 : x2=255
        if y1 < 0: y1=0
        if y2 >255 : y2=255

        num,mask = cv2.connectedComponents(Premask[y1:y2, x1:x2], connectivity=4)
        if num > 1:
            area = []
            pre[y1:y2, x1:x2] = 0
            STATS = cv2.connectedComponentsWithStats(cv2.convertScaleAbs(mask), connectivity=4)[2]
            area = STATS[:, cv2.CC_STAT_AREA]
            maxInd = np.argmax(area[1:]) + 1
            mask[mask != maxInd] = 0
            mask[mask == maxInd] = 1
            pre[y1:y2, x1:x2] = mask 
        elif num == 1:
            pre[y1:y2, x1:x2] = Premask[y1:y2, x1:x2] 
        Premask[pre != 0] = 0
        pre0[pre != 0] = i + 1

    se1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    pre0 = cv2.morphologyEx(pre0,cv2.MORPH_CLOSE,se1)
    pre0 = cv2.dilate(pre0, se1)
    # pre = np.dstack((pre, np.zeros_like(Premask), np.zeros_like(Premask)))
    return pre0