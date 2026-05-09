import torch
import torch.nn as nn


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

    def forward(self, batched_inputs):
        images, targets, H, W = self.preprocess_inputs(batched_inputs)

        features = self.backbone(images)
        outputs = self.decoder(features, images, targets)

        if self.training:
            assert targets is not None
            losses = self.criterion(outputs, targets)
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
        
        '''
        if "alpha" in batched_inputs:
            alpha = batched_inputs["alpha"].to(self.device)
            if len(alpha.shape) == 3:
                alpha = alpha.unsqueeze(1)
            
            low = 0.01
            high = 1.0
            # 为整个 Batch 生成一个随机阈值，或者每个样本一个 (这里简化为统一一个)
            thresh = low + torch.rand(1, device=self.device) * (high - low)
            mask = (alpha > thresh).float()

            # 2. 训练阶段加入扰动 (PyTorch 版腐蚀膨胀)
            if self.training:
                import torch.nn.functional as F
                
                # 随机决定操作：0:腐蚀, 1:膨胀, 2:开运算(先腐蚀后膨胀), 3:闭运算(先膨胀后腐蚀)
                random_num = 2
                
                # 随机核大小 (对应 range(1, 30))
                # PyTorch max_pool 核大小通常是奇数，这里模拟 1~30 像素
                k_size = int(torch.randint(1, 15, (1,)).item()) * 2 + 1
                padding = k_size // 2
                
                # 定义两个基础操作函数
                def dilate(m, k, p):
                    # 膨胀 = MaxPool
                    return F.max_pool2d(m, kernel_size=k, stride=1, padding=p)
                
                def erode(m, k, p):
                    # 腐蚀 = 负值的 MaxPool
                    return -F.max_pool2d(-m, kernel_size=k, stride=1, padding=p)

                # 执行逻辑 (模拟你的 cv2 逻辑)
                if random_num == 0: # 腐蚀
                    mask = erode(mask, k_size, padding)
                elif random_num == 1: # 膨胀
                    mask = dilate(mask, k_size, padding)
                elif random_num == 2: # 先腐蚀后膨胀 (开运算)
                    # 第二步可以换一个随机核，或者用同一个
                    k2 = int(torch.randint(1, 15, (1,)).item()) * 2 + 1
                    p2 = k2 // 2
                    mask = erode(mask, k_size, padding)
                    mask = dilate(mask, k2, p2)
                elif random_num == 3: # 先膨胀后腐蚀 (闭运算)
                    k2 = int(torch.randint(1, 15, (1,)).item()) * 2 + 1
                    p2 = k2 // 2
                    mask = dilate(mask, k_size, padding)
                    mask = erode(mask, k2, p2)
            #腐蚀 5 p
            # =================【调试代码开始】=================
        # 仅在调试时打开，确认无误后可注释掉
        
        try:
            import cv2
            import numpy as np
            import os
            
            # 1. 创建一个保存目录，以免弄乱根目录
            save_dir = "debug_vis"
            os.makedirs(save_dir, exist_ok=True)
            
            # 2. 取出 Batch 中的第一张图 (batch_index=0, channel=0)
            # mask shape: [B, 1, H, W] -> [H, W]
            vis_mask = mask[0, 0].detach().cpu().numpy()
            
            # 3. 如果有 alpha 原图，也可以一起保存对比
            # alpha shape: [B, 1, H, W] -> [H, W]
            vis_alpha = alpha[0, 0].detach().cpu().numpy()
            
            # 4. 转换数值范围: 0-1 (float) -> 0-255 (uint8)
            vis_mask_uint8 = (vis_mask * 255).astype(np.uint8)
            vis_alpha_uint8 = (vis_alpha * 255).astype(np.uint8)
            
            # 5. 保存图片
            # 建议加上随机数或时间戳，或者直接覆盖 'latest.png'
            cv2.imwrite(f"{save_dir}/debug_mask_latest.png", vis_mask_uint8)
            cv2.imwrite(f"{save_dir}/debug_alpha_latest.png", vis_alpha_uint8)
            
            print(f"DEBUG: Mask saved to {save_dir}/debug_mask_latest.png")
            
        except Exception as e:
            print(f"DEBUG SAVE FAILED: {e}")
        '''
        if "alpha" in batched_inputs:
            mask = batched_inputs["alpha"].to(self.device)
            if len(mask.shape) == 3:
                mask = mask.unsqueeze(1)
                
        images = torch.cat([images, mask], dim=1) 

        targets = dict()

        if self.training:
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

        return images, targets, H, W
