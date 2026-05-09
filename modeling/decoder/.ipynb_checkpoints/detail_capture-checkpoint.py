import torch
from torch import nn
from torch.nn import functional as F
import os #修改的添加
import sys

sys.path.insert(0, '/root/autodl-tmp/home/joey/wy/dift-main')


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

'''
def consistency(image, alpha, kernel_size=11, k=None):
    b, c, h, w = image.shape
    mean = image.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = image.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    image = image * std + mean
    unfold_image = F.unfold(image, kernel_size=kernel_size, padding=kernel_size // 2).view(b, c, kernel_size ** 2, h, w)
    image_dist = torch.norm(image.view(b, c, 1, h, w) - unfold_image, 2, dim=1)
    image_dist, indices = torch.topk(image_dist, k=kernel_size, dim=1, largest=False)
    unfold_alpha = F.unfold(alpha, kernel_size=kernel_size, padding=kernel_size // 2).view(b, kernel_size ** 2, h, w)
    alpha_dist = torch.gather(alpha - unfold_alpha, dim=1, index=indices)

    gray = (0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3])
    # 展开灰度窗口: [B, K^2, H, W]
    unfold_gray = F.unfold(gray, kernel_size=kernel_size, padding=kernel_size // 2).view(
        b, kernel_size ** 2, h, w
    )

    total_win = kernel_size ** 2
    k = kernel_size if k is None else min(k, total_win)

    # Top-k / Bottom-k 灰度值并求均值 → F/B: [B,1,H,W]
    topk_vals, _ = torch.topk(unfold_gray, k=k, dim=1, largest=True)
    botk_vals, _ = torch.topk(unfold_gray, k=k, dim=1, largest=False)
    fg_value = topk_vals.mean(dim=1, keepdim=True)  # [B,1,H,W]
    bg_value = botk_vals.mean(dim=1, keepdim=True)  # [B,1,H,W]

    return image_dist, alpha_dist, fg_value, bg_value, indices
'''
def dift_consistency(dift_features_upsampled, indices, kernel_size=11, feat_chunk_size=128):
    """
    计算DIFT特征一致性 - 在特征维度分块，进一步降低显存
    
    Args:
        dift_features_upsampled: [B, C_feat, H, W] DIFT特征
        indices: [B, K, H, W] 颜色相似像素的索引(0-120)
        kernel_size: 窗口大小
        feat_chunk_size: 每次处理的特征维度数（控制显存）
        
    Returns:
        dift_feat_dist: [B, K, H, W] DIFT特征距离（最终输出很小！）
    """
    b, c_feat, h, w = dift_features_upsampled.shape
    _, k_size, _, _ = indices.shape
    pad = kernel_size // 2
    
    # 1. 归一化并padding
    dift_norm = F.normalize(dift_features_upsampled, p=2, dim=1)
    dift_padded = F.pad(dift_norm, (pad, pad, pad, pad), mode='replicate')
    
    # 2. 准备采样坐标（只需计算一次）
    indices_y = indices // kernel_size - pad
    indices_x = indices % kernel_size - pad
    
    y_center = torch.arange(h, device=dift_norm.device).view(h, 1).expand(h, w) + pad
    x_center = torch.arange(w, device=dift_norm.device).view(1, w).expand(h, w) + pad
    
    sample_y = y_center.unsqueeze(0).unsqueeze(0) + indices_y
    sample_x = x_center.unsqueeze(0).unsqueeze(0) + indices_x
    
    h_padded, w_padded = h + 2 * pad, w + 2 * pad
    sample_y_norm = 2.0 * sample_y / (h_padded - 1) - 1.0
    sample_x_norm = 2.0 * sample_x / (w_padded - 1) - 1.0
    
    grid = torch.stack([sample_x_norm, sample_y_norm], dim=-1)
    grid = grid.view(b, k_size * h * w, 1, 2)
    
    # 3. 在特征维度上分块计算余弦相似度
    num_feat_chunks = (c_feat + feat_chunk_size - 1) // feat_chunk_size
    cosine_sim = torch.zeros(b, k_size, h, w, device=dift_norm.device)
    
    for feat_idx in range(num_feat_chunks):
        start_c = feat_idx * feat_chunk_size
        end_c = min((feat_idx + 1) * feat_chunk_size, c_feat)
        chunk_c = end_c - start_c
        
        # 提取当前特征块
        dift_chunk = dift_padded[:, start_c:end_c, :, :]  # [B, chunk_c, H+10, W+10]
        
        # Grid sample当前块
        selected_chunk = F.grid_sample(
            dift_chunk, grid,
            mode='bilinear', padding_mode='border', align_corners=True
        )  # [B, chunk_c, K*H*W, 1]
        
        selected_chunk = selected_chunk.squeeze(-1).view(b, chunk_c, k_size, h, w)
        # [B, chunk_c, K, H, W]  ← 只有 128×11×512×512 ≈ 1.5GB
        
        # 当前块的中心特征
        center_chunk = dift_norm[:, start_c:end_c, :, :].unsqueeze(2)
        # [B, chunk_c, 1, H, W]
        
        # 计算当前块的余弦相似度贡献（在chunk_c维度上求和）
        chunk_similarity = torch.sum(center_chunk * selected_chunk, dim=1)
        # [B, K, H, W]
        
        # 累加到总的相似度
        cosine_sim += chunk_similarity
        
        # 释放当前块
        del dift_chunk, selected_chunk, center_chunk, chunk_similarity
    
    # 4. 转换为距离
    dift_feat_dist = (1 - cosine_sim) / 2.0  # [B, K, H, W] ≈ 46MB
    
    return dift_feat_dist

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
            in_chans=4,
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

#修改的的Detail_Capture
class Detail_Capture(nn.Module):
    """
    Simple and Lightweight Detail Capture Module for ViT Matting.
    增强版：在线提取DIFT特征
    """

    def __init__(
            self,
            in_chans=384,
            img_chans=4,
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
        
        # 初始化DIFT模型
        self.use_dift = use_dift
        if self.use_dift:
            print("Initializing DIFT model for online feature extraction...")
            try:
                from src.models.dift_sd import SDFeaturizer
                
                # DIFT配置
                if dift_config is None:
                    dift_config = {
                        'sd_id': 'stabilityai/stable-diffusion-2-1',
                        'null_prompt': ''
                    }
                
                self.dift_model = SDFeaturizer(**dift_config)
                
                # 冻结DIFT模型参数
                for param in self.dift_model.pipe.unet.parameters():
                    param.requires_grad = False
                for param in self.dift_model.pipe.vae.parameters():
                    param.requires_grad = False
                    
                # DIFT提取参数
                self.dift_t = 261
                self.dift_up_ft_index = 2
                self.dift_ensemble_size = 4
                
                print("DIFT model initialized successfully!")
            except Exception as e:
                print(f"Warning: Failed to initialize DIFT model: {e}")
                print("Training will continue without DIFT features.")
                self.use_dift = False

    @torch.no_grad()
    def extract_dift_features(self, images):
        """
        在线提取DIFT特征
        Args:
            images: [B, 3, 512, 512] 归一化后的图像
        Returns:
            dift_features: [B, C, H, W] DIFT特征
        """
        # 反归一化到[0, 1]
        mean = images.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = images.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        images_denorm = images * std + mean  # [B, 3, 512, 512]
        
        # 转换到[-1, 1]用于DIFT
        images_for_dift = (images_denorm - 0.5) * 2
        
        # 逐个样本提取特征（避免显存不足）
        batch_features = []
        for i in range(images.shape[0]):
            img = images_for_dift[i:i+1]  # [1, 3, 512, 512]
            
            # 提取DIFT特征
            feat = self.dift_model.forward(
                img_tensor=img,
                prompt='',
                t=self.dift_t,
                up_ft_index=self.dift_up_ft_index,
                ensemble_size=self.dift_ensemble_size
            )  # [1, C, H_feat, W_feat]
            
            batch_features.append(feat)
        
        # 合并batch
        dift_features = torch.cat(batch_features, dim=0)  # [B, C, H_feat, W_feat]
        
        return dift_features

    def box_filter(self, x, radius):
        """
        Box filter for guided filter computation
        Args:
            x: [B, C, H, W] tensor
            radius: filter radius
        Returns:
            filtered: [B, C, H, W] tensor
        """
        k_size = 2 * radius + 1
        kernel = torch.ones(x.shape[1], 1, k_size, k_size, device=x.device, dtype=x.dtype)
        return F.conv2d(x, kernel, padding=radius, groups=x.shape[1])

    def guided_filter(self, I, p, r, eps):
        """
        彩色导向滤波 (Color Guided Filter) - PyTorch实现
        Args:
            I: [B, 3, H, W] 导向图（RGB图像）
            p: [B, 1, H, W] 输入图像（alpha matte）
            r: int, 滤波半径
            eps: float, 正则化参数
        Returns:
            q: [B, 1, H, W] 滤波后的输出
        """
        '''
        I = I.float()
        p = p.float()
        if I.max() > 1.0 + 1e-5:
            I = I / 255.0
        if p.max() > 1.0 + 1e-5:
            p = p / 255.0
        '''
        I = I.float()
        p = p.float()
        compute_dtype = torch.float32

        B, C, H, W = I.shape
        
        # 1. 计算局部均值
        N = self.box_filter(torch.ones((B, 1, H, W), device=I.device, dtype=compute_dtype), r)
        mean_I = self.box_filter(I, r) / N  # [B, 3, H, W]
        mean_p = self.box_filter(p, r) / N  # [B, 1, H, W]
        
        # 2. 计算协方差矩阵
        # I * I^T: [B, 3, 3, H, W]
        I_expanded = I.view(B, 3, 1, H, W)
        I_T = I.view(B, 1, 3, H, W)
        II = I_expanded * I_T  # [B, 3, 3, H, W]
        II_flat = II.view(B, 9, H, W)
        
        mean_II_flat = self.box_filter(II_flat, r) / N
        mean_II = mean_II_flat.view(B, 3, 3, H, W)
        
        # I*p 的均值
        Ip = I * p  # [B, 3, H, W]
        mean_Ip = self.box_filter(Ip, r) / N
        
        # 协方差
        cov_II = mean_II - mean_I.view(B, 3, 1, H, W) * mean_I.view(B, 1, 3, H, W)
        cov_Ip = (mean_Ip - mean_I * mean_p).unsqueeze(2)  # [B, 3, 1, H, W]
        
        # 添加正则化
        cov_II = cov_II + eps * torch.eye(3, device=I.device, dtype=compute_dtype).view(1, 3, 3, 1, 1)
        
        # 3. 求解线性方程 (转换维度以适应 linalg.solve)
        cov_II_permuted = cov_II.permute(0, 3, 4, 1, 2)  # [B, H, W, 3, 3]
        cov_Ip_permuted = cov_Ip.permute(0, 3, 4, 1, 2)  # [B, H, W, 3, 1]
        
        # 在 float32 下求解（避免 AMP 导致的数值问题）
        #a_permuted = torch.linalg.solve(cov_II_permuted.float(), cov_Ip_permuted.float())
        a_permuted = torch.linalg.solve(
            cov_II_permuted.to(torch.float64),
            cov_Ip_permuted.to(torch.float64)
        ).to(torch.float32)  # 立刻转回 float32
        a = a_permuted.squeeze(-1).permute(0, 3, 1, 2).to(I.dtype)  # [B, 3, H, W]
        
        # 4. 计算截距
        b = mean_p - torch.sum(a * mean_I, dim=1, keepdim=True)  # [B, 1, H, W]
        
        # 5. 均值滤波
        mean_a = self.box_filter(a, r) / N
        mean_b = self.box_filter(b, r) / N
        
        # 6. 输出
        q = torch.sum(mean_a * I, dim=1, keepdim=True) + mean_b  # [B, 1, H, W]
        q = q.clamp(0, 1)
        return q

    def inject_anchor_points(self, pred_alpha, trimap):
        """
        核心逻辑：强制锚点注入 (PyTorch 版)
        pred_alpha: [B, 1, H, W] 预测的 alpha
        trimap: [B, 1, H, W] 这里的 trimap 已经是 0, 0.5, 1 的数值了
        把预测alpha蒙版与trimap组合,使引导滤波输入图更加精确
        """
        # 1. 确保 trimap 和 pred_alpha 对齐
        if trimap.shape != pred_alpha.shape:
             # 有时 trimap 需要 resize 才能和 pred_alpha 对齐，防止报错
             trimap = F.interpolate(trimap, size=pred_alpha.shape[2:], mode='nearest')

        # 2. 定义已知区域 (Known Regions)
        # trimap < 0.1 (背景) 或 trimap > 0.9 (前景)
        # 你的 preprocess 已经把 trimap 变成了 0, 0.5, 1，所以这个阈值是安全的
        is_known = (trimap < 0.1) | (trimap > 0.9)
        
        # 3. 构建混合图 (Hybrid)
        # 逻辑：如果 is_known 为 True，取 trimap 的值；否则取 pred_alpha 的值
        # 这里的 trimap 本身就是 0 和 1，所以直接用 trimap 替换即可
        hybrid_alpha = torch.where(is_known, trimap, pred_alpha)
        
        return hybrid_alpha

    #def forward(self, features, images):
    def forward(self, features, images, targets):
        """
        Args:
            features: [B, 384, 32, 32] - backbone特征
            images: [B, 3, 512, 512] - 归一化后的图像
        """
        pixel_mean=[123.675 / 255., 116.280 / 255., 103.530 / 255.],
        pixel_std=[58.395 / 255., 57.120 / 255., 57.375 / 255.],
        detail_features = self.convstream(images)
        r = 7
        eps = 1e-6

        for i in range(len(self.fusion_blks)):
            d_name_ = 'D' + str(len(self.fusion_blks) - i - 1)
            features = self.fusion_blks[i](features, detail_features[d_name_])

        phas = self.matting_head(features).sigmoid()

        if self.training:
            #guide_image = images
            images = images[:, :3, :, :]
            guide_image = images * torch.tensor(pixel_std, device=images.device).view(1, 3, 1, 1) + torch.tensor(pixel_mean, device=images.device).view(1, 3, 1, 1)
            # 原有的颜色一致性
            #images_dist, phas_dist, fg_value, bg_value, indices = consistency(images, phas)
            images_dist, phas_dist, indices = consistency(images, phas)
            
            #phas_Com = phas
            if targets is not None and 'trimap' in targets:
                trimap = targets['trimap']
                # 调用注入函数
                phas_Com = self.inject_anchor_points(phas, trimap)
            # 在线提取DIFT特征
            dift_feat_dist = None
            #dift_alpha_dist = None
            
            guided_alpha = self.guided_filter(guide_image, phas_Com, r, eps)
            guided_alpha = self.guided_filter(guide_image, guided_alpha, r, eps)
            
            #images_dist, phas_dist, indices = consistency(images, guided_alpha)
            if self.use_dift:
                try:
                    # 提取DIFT特征
                    dift_features = self.extract_dift_features(images)  # [B, C, H_feat, W_feat]
                    
                    # 上采样到512×512
                    b, c, h, w = images.shape
                    dift_features_upsampled = F.interpolate(
                        dift_features,
                        size=(h, w),
                        mode='bilinear',
                        align_corners=False
                    )  # [B, C, 512, 512]
                    
                    # 计算DIFT特征一致性
                    dift_feat_dist = dift_consistency(dift_features_upsampled, indices, kernel_size=11, feat_chunk_size=128)
                    
                except Exception as e:
                    print(f"Warning: DIFT feature extraction failed: {e}")
                    dift_feat_dist = None
                    #dift_alpha_dist = None
        else:
            images_dist = None
            phas_dist = None
            #fg_value = None
            #bg_value = None
            guided_alpha = None
            dift_feat_dist = None
            #dift_alpha_dist = None
            
        out = {
            'phas': phas, 
            'images_dist': images_dist, 
            'phas_dist': phas_dist, 
            'images':images,
            'guided_alpha':guided_alpha,
            #'fg_value': fg_value, 
            #'bg_value': bg_value,
            #'dift_feat_dist': dift_feat_dist,
            #'dift_alpha_dist': dift_alpha_dist,
        }

        return out
'''
class Detail_Capture(nn.Module):
    """
    Simple and Lightweight Detail Capture Module for ViT Matting.
    """

    def __init__(
            self,
            in_chans=384,
            img_chans=4,
            convstream_out=[48, 96, 192],
            fusion_out=[256, 128, 64, 32],
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

    def forward(self, features, images):
        detail_features = self.convstream(images)

        for i in range(len(self.fusion_blks)):
            d_name_ = 'D' + str(len(self.fusion_blks) - i - 1)
            features = self.fusion_blks[i](features, detail_features[d_name_])

        phas = self.matting_head(features).sigmoid()
        if self.training:
            images_dist, phas_dist, fg_value, bg_value = consistency(images, phas)
        else:
            images_dist = None
            phas_dist = None
            fg_value = None
            bg_value = None
        out = {'phas': phas, 'images_dist': images_dist, 'phas_dist': phas_dist, 'fg_value':fg_value, 'bg_value':bg_value}
        # out = {'phas': phas}

        return out
'''
'''
    def forward(self, features, images):
        detail_features = self.convstream(images)

        for i in range(len(self.fusion_blks)):
            d_name_ = 'D' + str(len(self.fusion_blks) - i - 1)
            features = self.fusion_blks[i](features, detail_features[d_name_])

        phas = self.matting_head(features).sigmoid()
        if self.training:
            images_dist, phas_dist = consistency(images, phas)
        else:
            images_dist = None
            phas_dist = None
        out = {'phas': phas, 'images_dist': images_dist, 'phas_dist': phas_dist}
        # out = {'phas': phas}

        return out
'''
