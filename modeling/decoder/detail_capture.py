import torch
from torch import nn
from torch.nn import functional as F
import os
import sys

from configs.base.public_algorithm import compute_eikonal_from_tensor


def consistency(image, alpha, kernel_size=11):
    b, c, h, w = image.shape
    mean = image.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = image.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    image = image * std + mean
    unfold_image = F.unfold(image, kernel_size=kernel_size, padding=kernel_size // 2).view(b, c, kernel_size ** 2, h, w)
    image_dist = torch.norm(image.view(b, c, 1, h, w) - unfold_image, 2, dim=1)
    image_dist, indices = torch.topk(image_dist, k=kernel_size, dim=1, largest=False)
    unfold_alpha = F.unfold(alpha, kernel_size=kernel_size, padding=kernel_size // 2).view(b, kernel_size ** 2, h, w)
    alpha_dist = torch.gather(alpha - unfold_alpha, dim=1, index=indices)

    return image_dist, alpha_dist, indices


def consistency_txs(image, alpha, txs, kernel_size=11, repeat_rgb=2, gamma=5.0):
    b, c, h, w = image.shape
    
    mean = image.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = image.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    image_raw = image * std + mean
    
    unfold_image = F.unfold(
        image_raw, 
        kernel_size=kernel_size, 
        padding=kernel_size // 2
    ).view(b, 3, kernel_size ** 2, h, w)
    
    image_dist_all = torch.norm(image_raw.view(b, 3, 1, h, w) - unfold_image, p=2, dim=1)
    _, indices = torch.topk(image_dist_all, k=kernel_size, dim=1, largest=False)

    image_dist = torch.gather(image_dist_all, dim=1, index=indices) # (b, k, h, w)

    fs_min = txs.view(b, -1).min(dim=1)[0].view(b, 1, 1, 1)
    fs_max = txs.view(b, -1).max(dim=1)[0].view(b, 1, 1, 1)
    fs_norm = (txs - fs_min) / (fs_max - fs_min + 1e-8) # (b, 1, h, w)

    gate = 1.0 - 0.5 * fs_norm
    image_dist = image_dist * gate 
    
    unfold_alpha = F.unfold(
        alpha, 
        kernel_size=kernel_size, 
        padding=kernel_size // 2
    ).view(b, kernel_size ** 2, h, w)
    
    alpha_dist = torch.gather(alpha - unfold_alpha, dim=1, index=indices)

    return image_dist, alpha_dist, indices


class Basic_Conv3x3(nn.Module):
    """
    Basic convolution layers including: Conv3x3, BatchNorm2d, ReLU layers.
    """

    def __init__(
            self,
            in_chans,
            out_chans,
            stride=2,
            padding=1,
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_chans, out_chans, 3, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_chans)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class ConvStream(nn.Module):
    """
    Simple ConvStream containing a series of basic conv3x3 layers to extract detail features.
    """

    def __init__(
            self,
            in_chans=3,
            out_chans=[48, 96, 192],
    ):
        super().__init__()
        self.convs = nn.ModuleList()

        self.conv_chans = out_chans.copy()
        self.conv_chans.insert(0, in_chans)

        for i in range(len(self.conv_chans) - 1):
            in_chan_ = self.conv_chans[i]
            out_chan_ = self.conv_chans[i + 1]
            self.convs.append(
                Basic_Conv3x3(in_chan_, out_chan_)
            )

    def forward(self, x):
        out_dict = {'D0': x}
        for i in range(len(self.convs)):
            x = self.convs[i](x)
            name_ = 'D' + str(i + 1)
            out_dict[name_] = x

        return out_dict


class Fusion_Block(nn.Module):
    """
    Simple fusion block to fuse feature from ConvStream and Plain Vision Transformer.
    """

    def __init__(
            self,
            in_chans,
            out_chans,
    ):
        super().__init__()
        self.conv = Basic_Conv3x3(in_chans, out_chans, stride=1, padding=1)

    def forward(self, x, D):
        F_up = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        out = torch.cat([D, F_up], dim=1)
        out = self.conv(out)

        return out


class Matting_Head(nn.Module):
    """
    Simple Matting Head, containing only conv3x3 and conv1x1 layers.
    """

    def __init__(
            self,
            in_chans=32,
            mid_chans=16,
    ):
        super().__init__()
        self.matting_convs = nn.Sequential(
            nn.Conv2d(in_chans, mid_chans, 3, 1, 1),
            nn.BatchNorm2d(mid_chans),
            nn.ReLU(True),
            nn.Conv2d(mid_chans, 1, 1, 1, 0)
        )

    def forward(self, x):
        x = self.matting_convs(x)

        return x


class Detail_Capture(nn.Module):
    def __init__(
            self,
            in_chans=640,
            img_chans=3,
            convstream_out=[48, 96, 192],
            fusion_out=[256, 128, 64, 32],
            use_dift=True,
            dift_config=None,
    ):
        super().__init__()
        assert len(fusion_out) == len(convstream_out) + 1

        self.convstream = ConvStream(in_chans=img_chans)
        self.conv_chans = self.convstream.conv_chans

        self.fusion_blks = nn.ModuleList()
        self.fus_channs = fusion_out.copy()
        self.fus_channs.insert(0, in_chans)
        for i in range(len(self.fus_channs) - 1):
            self.fusion_blks.append(
                Fusion_Block(
                    in_chans=self.fus_channs[i] + self.conv_chans[-(i + 1)],
                    out_chans=self.fus_channs[i + 1],
                )
            )

        self.matting_head = Matting_Head(
            in_chans=fusion_out[-1],
        )
        


    def inject_anchor_points(self, pred_alpha, trimap):
        if trimap.shape != pred_alpha.shape:
             trimap = F.interpolate(trimap, size=pred_alpha.shape[2:], mode='nearest')
        is_known = (trimap < 0.1) | (trimap > 0.9)
        hybrid_alpha = torch.where(is_known, trimap, pred_alpha)
        
        return hybrid_alpha

    def forward(self, features, images, detail_image ,targets):
        pixel_mean=[123.675 / 255., 116.280 / 255., 103.530 / 255.],
        pixel_std=[58.395 / 255., 57.120 / 255., 57.375 / 255.],
        detail_features = self.convstream(images)

        for i in range(len(self.fusion_blks)):
            d_name_ = 'D' + str(len(self.fusion_blks) - i - 1)
            features = self.fusion_blks[i](features, detail_features[d_name_])

        phas = self.matting_head(features).sigmoid()

        if self.training:
            images = images[:, :3, :, :]
            guide_image = images * torch.tensor(pixel_std, device=images.device).view(1, 3, 1, 1) + torch.tensor(pixel_mean, device=images.device).view(1, 3, 1, 1)

            f_x_list = []
            B, C, H, W = images.shape
            for i in range(B):
                f_x_item = compute_eikonal_from_tensor(images[i], targets['mask'][i], normalize=True, return_gradients=False)
                f_x_list.append(f_x_item)
            f_xs = torch.stack(f_x_list, dim=0) # [B, 1, H, W]
            images_dist, phas_dist, indices = consistency_txs(images, phas, f_xs)
            
            if targets is not None and 'trimap' in targets:
                trimap = targets['trimap']
                phas_Com = self.inject_anchor_points(phas, trimap)
            
        else:
            images_dist = None
            phas_dist = None
            dift_feat_dist = None
            
        out = {
            'phas': phas, 
            'images_dist': images_dist, 
            'phas_dist': phas_dist, 
            'images':images,
        }

        return out
