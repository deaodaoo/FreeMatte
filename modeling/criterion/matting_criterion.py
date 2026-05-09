import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

import cv2
import numpy as np
import skfmm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import distance_transform_edt
import os
import shutil

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from configs.base.public_algorithm import compute_eikonal_from_tensor, compute_absolute_bg_mask

GLOBAL_TRAIN_UP_COUNTER = 0

class MattingCriterion(nn.Module):
    def __init__(self,
                 *,
                 losses,
                 aux_losses=None,
                 ):
        super(MattingCriterion, self).__init__()
        self.losses = losses
        self.aux_losses = aux_losses
    
    def loss_gradient_penalty(self, sample_map, preds, targets):
        preds = preds['phas']
        targets = targets['phas']

        #sample_map for unknown area
        scale = sample_map.shape[0]*262144/torch.sum(sample_map)

        #gradient in x
        sobel_x_kernel = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]]).type(dtype=preds.type())
        delta_pred_x = F.conv2d(preds, weight=sobel_x_kernel, padding=1)
        delta_gt_x = F.conv2d(targets, weight=sobel_x_kernel, padding=1)

        #gradient in y 
        sobel_y_kernel = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]]).type(dtype=preds.type())
        delta_pred_y = F.conv2d(preds, weight=sobel_y_kernel, padding=1)
        delta_gt_y = F.conv2d(targets, weight=sobel_y_kernel, padding=1)

        #loss
        loss = (F.l1_loss(delta_pred_x*sample_map, delta_gt_x*sample_map)* scale + \
            F.l1_loss(delta_pred_y*sample_map, delta_gt_y*sample_map)* scale + \
            0.01 * torch.mean(torch.abs(delta_pred_x*sample_map))* scale +  \
            0.01 * torch.mean(torch.abs(delta_pred_y*sample_map))* scale)

        return dict(loss_gradient_penalty=loss)

    def unknown_l1_loss(self, sample_map, preds, targets):
        
        scale = sample_map.shape[0]*512*512/torch.sum(sample_map)

        loss = F.l1_loss(preds['phas']*sample_map, targets['phas']*sample_map)*scale
        return dict(unknown_l1_loss=loss)

    def known_l1_loss(self, sample_map, outputs, targets, **kwargs):
        new_sample_map = torch.zeros_like(sample_map)
        new_sample_map[sample_map==0] = 1
        
        if torch.sum(new_sample_map) == 0:
            scale = 0
        else:
            scale = new_sample_map.shape[0]*512*512/torch.sum(new_sample_map)
        loss = F.l1_loss(outputs['phas']*new_sample_map, targets['trimap']*new_sample_map)*scale
        return dict(known_l1_loss=loss)


    def known_l1_eikonal_sparse_loss(self, sample_map, outputs, targets, kernel_size=11, bg_ratio=0.01, batched_inputs = None, **kwargs):
        mask = targets['mask'].float()
        pad = kernel_size // 2
        dilated_mask = F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=pad)
        eroded_mask = 1.0 - F.max_pool2d(1.0 - mask, kernel_size=kernel_size, stride=1, padding=pad)
        full_bg_mask = 1.0 - dilated_mask
        B, C, H, W = full_bg_mask.shape
        device = outputs['phas'].device
        images = batched_inputs['image'].to(device)
        u_x_list = []
        for i in range(B):
            u_x_item = compute_eikonal_from_tensor(images[i], targets['mask'][i])
            u_x_list.append(u_x_item)
        u_x = torch.stack(u_x_list, dim=0) # [B, 1, H, W]
        sparse_bg_mask = compute_absolute_bg_mask(u_x, full_bg_mask)
        new_sample_map = eroded_mask + sparse_bg_mask
        valid_pixels = torch.sum(new_sample_map)
        scale = 0.0 if valid_pixels == 0 else new_sample_map.numel() / valid_pixels
        loss = F.l1_loss(outputs['phas'] * new_sample_map, targets['mask'] * new_sample_map) * scale
        
        return dict(known_l1_loss=loss)

    
    def consistency_loss(self, sample_map, outputs, targets, **kwargs):
        loss = F.l1_loss(outputs['images_dist'], outputs['phas_dist']) * 10
        return dict(consistency_loss=loss)


    def get_loss(self, loss, sample_map, outputs, targets, batched_inputs=None, prev_model = None):
        loss_map = {
            'consistency': self.consistency_loss,
            'known_eikonal_sparse': self.known_l1_eikonal_sparse_loss,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](sample_map, outputs, targets, batched_inputs=batched_inputs, prev_model=prev_model)

    def forward(self, outputs, targets, current_epoch=None, total_epochs=None, batched_inputs=None, prev_model=None):
        self.losses = ['known_eikonal_sparse', 'consistency']

        sample_map = torch.zeros_like(targets['trimap'])
        sample_map[targets['trimap'] == 0.5] = 1

        losses = dict()
        for loss in self.losses:
            losses.update(self.get_loss(loss, sample_map, outputs, targets, batched_inputs, prev_model))
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                for loss in self.aux_losses:
                    l_dict = self.get_loss(loss, sample_map, aux_outputs, targets, batched_inputs, prev_model)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        return losses
