import torch
import torch.nn as nn
from transformers import SamModel, SamProcessor
import numpy as np
import torchvision

import torch.nn.functional as F


class ViTMatte(nn.Module):
    def __init__(self,
                 *,
                 backbone,
                 criterion,
                 pixel_mean,
                 pixel_std,
                 input_format,
                 size_divisibility,
                 decoder,
                 ):
        super(ViTMatte, self).__init__()
        self.backbone = backbone
        # self.backbone_detail = backbone_detail
        self.criterion = criterion
        self.input_format = input_format
        self.size_divisibility = size_divisibility
        self.decoder = decoder
        self.register_buffer(
            "pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False
        )
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        assert (
                self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"

    @property
    def device(self):
        return self.pixel_mean.device


    def forward(self, batched_inputs, current_epoch=None, total_epochs=None, prev_model = None):
        images, images_w_mask, blured_images_w_mask, images_res, targets, masks, image_names, H, W = self.preprocess_inputs(batched_inputs)

        features = self.backbone(images_w_mask)

        outputs = self.decoder(features, images, images_res, targets)

        if self.training:
            assert targets is not None
            losses = self.criterion(outputs, targets, current_epoch, total_epochs, batched_inputs, prev_model)
            return losses
        else:
            outputs['phas'] = outputs['phas'][:, :, :H, :W]
            return outputs

    def preprocess_inputs(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = batched_inputs["image"].to(self.device)

        self.image = images[-1, :, :, :].permute(1, 2, 0).detach().cpu().numpy()
        images = (images - self.pixel_mean) / self.pixel_std
        
        targets = dict()

        if "alpha" in batched_inputs:
            mask = batched_inputs["alpha"].to(self.device)
            if len(mask.shape) == 3:
                mask = mask.unsqueeze(1)
            targets['mask'] = mask

        # if self.training:
        if 'trimap' in batched_inputs:
            trimap = batched_inputs['trimap'].to(self.device)

            # trimap = trimap.float() / 255.
            trimap = trimap.float()
            trimap[trimap < 85] = 0
            trimap[trimap >= 170] = 1
            trimap[trimap >= 85] = 0.5

            targets.update(dict(trimap=trimap))
        

        B, C, H, W = images.shape
        if images.shape[-1] % 32 != 0 or images.shape[-2] % 32 != 0:
            new_H = (32 - images.shape[-2] % 32) + H
            new_W = (32 - images.shape[-1] % 32) + W
            new_images = torch.zeros((images.shape[0], images.shape[1], new_H, new_W)).to(self.device)
            new_images[:, :, :H, :W] = images[:, :, :, :]
            del images
            images = new_images

            new_mask = torch.zeros((mask.shape[0], mask.shape[1], new_H, new_W)).to(self.device)
            new_mask[:, :, :H, :W] = mask[:, :, :, :]
            mask = new_mask


        images_w_mask = torch.cat([images, mask], dim=1)

        kernel_size = 11
        sigma = 1.0
        blured_images_w_mask = torchvision.transforms.functional.gaussian_blur(images_w_mask, kernel_size, sigma)

        blured_images = blured_images_w_mask[:, :3, :, :]
        images_res = images - blured_images

        image_names = batched_inputs['image_name']

        return images, images_w_mask, blured_images_w_mask, images_res, targets, mask, image_names, H, W

    def get_prompts_from_trimap(self, trimaps, names):
        # trimaps: torch.Tensor, shape (B,1,H,W) or (B,H,W); values can be 0/0.5/1
        if not torch.is_tensor(trimaps):
            trimaps = torch.as_tensor(trimaps)

        if trimaps.dim() == 4:
            # (B,1,H,W) -> (B,H,W)
            if trimaps.size(1) != 1:
                raise ValueError(f"Expected trimaps channel=1, got {tuple(trimaps.shape)}")
            trimaps = trimaps[:, 0]
        elif trimaps.dim() != 3:
            raise ValueError(f"trimaps must have shape (B,1,H,W) or (B,H,W), got {tuple(trimaps.shape)}")

        input_boxes = []
        for b in range(trimaps.size(0)):
            ys, xs = torch.where(trimaps[b] > 0)
            if ys.numel() == 0:
                H, W = trimaps[b].shape
                input_boxes.append([[0, 0, W - 1, H - 1]])
                print("Empty/invalid trimap for box extraction at batch index {}".format(b))
                print(names[b])
                print(names)

                unique_vals, counts = torch.unique(trimaps[b], return_counts=True)
                print(f"Trimap value distribution: {dict(zip(unique_vals.tolist(), counts.tolist()))}")

                continue

            x_min = int(xs.min().item())
            x_max = int(xs.max().item())
            y_min = int(ys.min().item())
            y_max = int(ys.max().item())
            input_boxes.append([[x_min, y_min, x_max, y_max]])

        return input_boxes
