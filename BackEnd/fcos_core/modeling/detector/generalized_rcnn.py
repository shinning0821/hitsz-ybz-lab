# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn
from torch.nn import functional as F

from fcos_core.structures.image_list import to_image_list

from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads
from pytorch_grad_cam import ActivationsAndGradients
from pytorch_grad_cam import MyCAM
from ..contrast_head.contrastive_head import ContrastiveHead
from torch.nn.functional import cosine_similarity


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()
        
        self.backbone = build_backbone(cfg)
        self.rpn = build_rpn(cfg, self.backbone.out_channels)
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)
        self.mask = cfg.MODEL.DDTNET_ON
        target_layers = [self.backbone.body.layer1,self.backbone.body.layer2,self.backbone.body.layer3,self.backbone.body.layer4]
        self.activations_and_grads = ActivationsAndGradients(
            self.backbone, target_layers, reshape_transform=None)
        self.cam = MyCAM(target_layers=target_layers,use_cuda=torch.cuda.is_available())
        self.con_head = ContrastiveHead()

    def forward(self, images, targets=None, gt=None, center=None, pt_map=None, bkg=None, obj=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        images = to_image_list(images)
        features = self.activations_and_grads(images.tensors)
        
        if gt is not None:
            boxes, mask, proposal_losses = self.rpn(images, features, targets, gt, center,pt_map,bkg,obj)
        else:
            if self.mask:
                boxes, mask, proposal_losses = self.rpn(images, features, targets)
            else:
                boxes, proposal_losses = self.rpn(images, features, targets)
        
        if self.training:
            self.zero_grad()
            cam_index = self._compute_cam_loss_(mask)
            cam_index.backward(retain_graph=True)
            activations_list = [a.cpu().data.numpy()
                                for a in self.activations_and_grads.activations]    
            grads_list = [g.cpu().data.numpy()                                      
                          for g in self.activations_and_grads.gradients]
            cam_image = self.cam(input_tensor=images.tensors,activations_list=activations_list,grads_list=grads_list)
            

            feature_map = F.interpolate(features[0], size=(256,256), mode='bilinear', align_corners=True)
            GT = torch.stack(gt).squeeze(1)
            feature_map = feature_map.permute(0, 2, 3, 1)
            pos_anchor = torch.mean(feature_map[GT==1], dim=0).unsqueeze(0)
            neg_anchor = torch.mean(feature_map[GT==0], dim=0).unsqueeze(0)

            max_index = 0
            min_err = 1

            pos = torch.stack(pt_map).squeeze(1).long().detach()
            neg = torch.stack(bkg).squeeze(1).long().detach()

            for i in range(len(cam_image)):
                tensor_cam = torch.from_numpy(cam_image[i]).squeeze(1).cuda()
                
                cam_pos = torch.zeros(tensor_cam.shape).cuda()
                cam_neg = torch.zeros(tensor_cam.shape).cuda()
                cam_pos[tensor_cam>0.7] = 1
                cam_neg[tensor_cam<0.3] = 1
                

                count1 = torch.sum((pos == 1)) - torch.sum(pos * cam_pos)
                count2 = torch.sum((neg == 1)) - torch.sum(neg * cam_neg)
                
                err = (count1 + count2)/(cam_pos.shape[0]*cam_pos.shape[1]*cam_pos.shape[2])
                if(err < min_err):
                    min_err = err
                    max_index = i

            max_index = 3
            cam_image = cam_image[max_index].squeeze(1)
            self.zero_grad()
            contrastive_loss,enhance_feature = self.con_head(features[-2], cam_image, pt_map, bkg)

            import numpy as np
            cam_pos = np.zeros(cam_image.shape)
            cam_neg = np.zeros(cam_image.shape)
            cam_pos[cam_image>0.8] = 1
            cam_neg[cam_image<0.2] = 1
            
            cam_pos = torch.from_numpy(cam_pos).long().cuda()
            cam_neg = torch.from_numpy(cam_neg).long().cuda()
            
            if(min_err <  0.2):
                cam_loss = self.cam_bce_loss(mask[0], cam_pos, cam_neg) * (0.2-err)
            else:
                cam_loss = self.cam_bce_loss(mask[0], cam_pos, cam_neg) * 0
                
            
            

        
        proposals = None
        if self.roi_heads: # 这个确实是rpn only
                x, result, detector_losses = self.roi_heads
                (features, proposals, targets)
        else:
            # RPN-only models don't have roi_heads
            x = features
            detector_losses = {}

        if targets is not None:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            
            if self.training:
                losses.update(contrastive_loss)
                losses.update({'loss_cam': cam_loss})
                return losses

            if self.mask:
                return boxes, mask, losses
            else:
                return boxes, losses
        else:
            if self.mask:
                return boxes, mask, None
            else:
                return boxes, None


    def _compute_cam_loss_(self, mask):
        # 计算cam所需要的梯度信息

        # 首先将mask上采样到原图大小
        mask = mask[0]
        mask = torch.sigmoid(mask)
        mask = F.interpolate(mask, scale_factor=2)

        pred = mask.clone()
        pred = (pred > 0.5).squeeze(1)
        loss = 0
        for i in range(mask.shape[0]):
            loss += (pred[i] * mask[i,0,:,:]).sum()
        return loss


    def cam_bce_loss(self, pred, pos, neg):
        size = pos[0].shape
        pred = F.interpolate(pred, size=[size[0], size[1]])
        f_loss = F.cross_entropy(pred, pos, 
                        ignore_index=0)
        b_loss = F.cross_entropy(pred, 1-neg, 
                            ignore_index=1) 
        return f_loss + b_loss