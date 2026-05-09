import torch
import torch.nn as nn
import torch.nn.functional as F


class MattingCriterion(nn.Module):
    def __init__(self,
                 *,
                 losses,
                 aux_losses=None,
                 ):
        super(MattingCriterion, self).__init__()
        self.losses = losses
        self.aux_losses = aux_losses
        # 添加像素归一化参数（与训练时使用的相同）
        #self.register_buffer("pixel_mean", torch.tensor([123.675 / 255., 116.280 / 255., 103.530 / 255.]).view(1, 3, 1, 1))
        #self.register_buffer("pixel_std", torch.tensor([58.395 / 255., 57.120 / 255., 57.375 / 255.]).view(1, 3, 1, 1))
    '''
    def box_filter(self, x, radius):
        k_size = 2 * radius + 1
        # 修复：为每个通道创建独立的 kernel
        kernel = torch.ones(x.shape[1], 1, k_size, k_size, device=x.device)
        return F.conv2d(x, kernel, padding=radius, groups=x.shape[1])

    def guided_filter(self, image, alpha, radius=5, eps=1e-4):
        """
        使用 PyTorch 矩阵运算的正确彩色引导滤波 (修正维度版)
        """
        image = image.float()
        alpha = alpha.float()
    
        B, C, H, W = image.shape
        
        # 1. 计算局部均值
        # box_filter 只能处理 [B, C, H, W]，所以计算 N 时只需传入单通道
        N = self.box_filter(torch.ones((B, 1, H, W), device=image.device, dtype=torch.float), radius)
        mean_I = self.box_filter(image, radius) / N  # [B, 3, H, W]
        mean_p = self.box_filter(alpha, radius) / N # [B, 1, H, W]
        
        # 2. 计算协方差矩阵 (向量化计算)
        
        # --- 关键修正 A: 处理 I*I 的 5D 张量问题 ---
        # 目标: 计算 I * I^T，形状原为 [B, 3, 3, H, W]
        # 为了放入 box_filter，我们需要将其 reshape 为 [B, 9, H, W]
        
        guide_expanded = image.view(B, 3, 1, H, W)
        guide_T = image.view(B, 1, 3, H, W)
        # 逐元素相乘得到外积，结果为 [B, 3, 3, H, W]
        II = guide_expanded * guide_T 
        # 展平为 [B, 9, H, W] 以进行卷积
        II_flat = II.view(B, 9, H, W) 
        
        mean_II_flat = self.box_filter(II_flat, radius) / N
        mean_II = mean_II_flat.view(B, 3, 3, H, W) # 恢复为 [B, 3, 3, H, W]
        
        # 计算 I*p 的均值
        Ip = image * alpha  # [B, 3, H, W] (利用广播)
        mean_Ip = self.box_filter(Ip, radius) / N
        
        # 协方差计算
        # cov_II: [B, 3, 3, H, W]
        # mean_I * mean_I^T 也需要广播处理
        cov_II = mean_II - mean_I.view(B, 3, 1, H, W) * mean_I.view(B, 1, 3, H, W)
        
        # cov_Ip: [B, 3, H, W] -> 调整为 [B, 3, 1, H, W] 以便后续矩阵运算
        cov_Ip = mean_Ip - mean_I * mean_p
        cov_Ip = cov_Ip.unsqueeze(2) 
        
        # 添加正则化
        # eye 形状: [1, 3, 3, 1, 1] -> 广播到 [B, 3, 3, H, W]
        cov_II = cov_II + eps * torch.eye(3, device=image.device, dtype=torch.float).view(1, 3, 3, 1, 1)
        
        # --- 关键修正 B: 处理 linalg.solve 的维度问题 ---
        # linalg.solve 要求矩阵在最后两维: [..., 3, 3]
        # 当前 cov_II: [B, 3, 3, H, W] -> 需要变为 [B, H, W, 3, 3]
        # 当前 cov_Ip: [B, 3, 1, H, W] -> 需要变为 [B, H, W, 3, 1]
        
        cov_II_permuted = cov_II.permute(0, 3, 4, 1, 2)
        cov_Ip_permuted = cov_Ip.permute(0, 3, 4, 1, 2)
        
        # 3. 求解线性方程 Σ * a = cov_Ip
        # result shape: [B, H, W, 3, 1]
        import torch.cuda.amp as amp # 需要导入 amp 模块

        with amp.autocast(enabled=False):
        # 确保输入张量在进入这个块时已经是 float32
        # 或者在块内再次显式转换为 float32
            a_permuted = torch.linalg.solve(cov_II_permuted.float(), cov_Ip_permuted.float())
        #a_permuted = torch.linalg.solve(cov_II_permuted, cov_Ip_permuted)
        
        # 将结果 permute 回 [B, 3, H, W] (去除最后的维度1)
        a = a_permuted.squeeze(-1).permute(0, 3, 1, 2)
        
        # 4. 计算截距
        # a: [B, 3, H, W], mean_I: [B, 3, H, W]
        # sum(dim=1) 相当于点积
        b = mean_p - torch.sum(a * mean_I, dim=1, keepdim=True)
        
        # 5. 均值滤波
        mean_a = self.box_filter(a, radius) / N
        mean_b = self.box_filter(b, radius) / N
        
        # 6. 输出
        guide_phas = torch.sum(mean_a * image, dim=1, keepdim=True) + mean_b
        
        return guide_phas
    '''

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

    def loss_pha_laplacian(self, preds, targets):
        assert 'phas' in preds and 'phas' in targets
        loss = laplacian_loss(preds['phas'], targets['phas'])

        return dict(loss_pha_laplacian=loss)

    def unknown_l1_loss(self, sample_map, preds, targets):
        
        scale = sample_map.shape[0]*512*512/torch.sum(sample_map)

        loss = F.l1_loss(preds['phas']*sample_map, targets['phas']*sample_map)*scale
        return dict(unknown_l1_loss=loss)

    def known_l1_loss(self, sample_map, outputs, targets):
        new_sample_map = torch.zeros_like(sample_map)
        new_sample_map[sample_map==0] = 1
        
        if torch.sum(new_sample_map) == 0:
            scale = 0
        else:
            scale = new_sample_map.shape[0]*512*512/torch.sum(new_sample_map)

        # loss = F.l1_loss(preds['phas']*new_sample_map, targets['phas']*new_sample_map)*scale
        #loss = F.l1_loss(outputs['phas']*new_sample_map, targets['trimap']*new_sample_map)*scale
        loss = F.l1_loss(outputs['phas']*new_sample_map, targets['trimap']*new_sample_map)*scale
        return dict(known_l1_loss=loss)

    def l1_loss(self, sample_map, outputs, targets):
        loss = F.l1_loss(outputs['phas'], targets['trimap'])
        return dict(l1_loss=loss)

    def dift_consistency_loss(self, sample_map, outputs, targets):
        """
        基于DIFT特征的一致性损失
        """
        #if outputs.get('dift_feat_dist') is None or outputs.get('dift_alpha_dist') is None:
        if outputs.get('dift_feat_dist') is None:
            # 如果DIFT特征不存在，返回0损失
            return dict(dift_consistency_loss=torch.tensor(0.0, device=sample_map.device))
        
        #feat_dist = outputs['dift_feat_dist']
        #phas_dist = outputs['phas_dist']
        '''
        # 特征相似 → alpha相似
        loss = (
            torch.pow(1 - feat_dist, 4) * torch.pow(alpha_dist, 4) +
            torch.pow(feat_dist, 4) * torch.pow(1 - alpha_dist, 4)
        )
        loss = torch.pow(torch.mean(loss), 1/4) * 10
        '''
        loss = F.l1_loss(outputs['dift_feat_dist'], outputs['phas_dist']) * 5

        return dict(dift_consistency_loss=loss)
    
    def consistency_loss(self, sample_map, outputs, targets):
        #loss = F.l1_loss(outputs['images_dist'], outputs['phas_dist'] * (outputs['fg_value'] - outputs['bg_value'])) * 10.
        """
        image_dist = outputs['images_dist']
        phas_dist = outputs['phas_dist']
        diff = image_dist - image_dist * (phas_dist - image_dist)
        mean_diff = torch.mean(torch.pow(torch.abs(diff), 1))
        P_norm = torch.pow(mean_diff, 1)
        loss = P_norm * 10
        """
        #相似和不相似逻辑损失
        #image_dist = outputs['images_dist']
        #phas_dist = outputs['phas_dist']
        #loss = torch.exp(-image_dist * 4) * torch.pow(phas_dist,4) + torch.pow(image_dist,4) * torch.pow(1 - phas_dist,4)
        #loss = torch.pow(1-image_dist, 4) * torch.pow(phas_dist,4) + torch.pow(image_dist,4) * torch.pow(1 - phas_dist,4)
        #loss = torch.pow(torch.mean(loss),1/4) * 3
    
        loss = F.l1_loss(outputs['images_dist'], outputs['phas_dist']) * 10
        return dict(consistency_loss=loss)

    def guided_filter_loss(self, sample_map, outputs, targets):
        """
        计算引导滤波损失：guided_alpha 与 alpha_pred 的 L1 差异。
        """
        if 'guided_alpha' not in outputs:
            return torch.tensor(0.0, device=sample_map.device)
        
        guided_alpha = outputs['guided_alpha']  # [B, 1, H, W]
        alpha_pred = outputs['phas']  # 或正确的 alpha 键
        
        # 计算 L1 损失
        #loss = F.l1_loss(guided_alpha, alpha_pred, reduction='none')
        loss = F.l1_loss(guided_alpha.detach(), alpha_pred) * 10

        return dict(guided_loss=loss)

    def gradient_loss(self, sample_map, outputs, targets):
        '''
        区域感知的梯度损失
        - 过渡区域：alpha梯度应与原图梯度相关
        - 前景/背景已知区域：alpha应保持平滑
        '''
        alpha_pred = outputs['phas']  # [B, 1, H, W]
        input_image = outputs['images']  # [B, 3, H, W]
        trimap = targets['trimap']  # [B, 1, H, W]
    
        # 1. 计算梯度
        # 原图梯度（灰度化）
        gray_image = 0.299 * input_image[:, 0] + 0.587 * input_image[:, 1] + 0.114 * input_image[:, 2]
        
        sobel_x = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], 
                            dtype=alpha_pred.dtype, device=alpha_pred.device)
        sobel_y = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], 
                            dtype=alpha_pred.dtype, device=alpha_pred.device)
        
        image_grad_x = F.conv2d(gray_image.unsqueeze(1), sobel_x, padding=1)
        image_grad_y = F.conv2d(gray_image.unsqueeze(1), sobel_y, padding=1)
        image_grad = torch.sqrt(image_grad_x**2 + image_grad_y**2 + 1e-8)
        
        alpha_grad_x = F.conv2d(alpha_pred, sobel_x, padding=1)
        alpha_grad_y = F.conv2d(alpha_pred, sobel_y, padding=1)
        alpha_grad = torch.sqrt(alpha_grad_x**2 + alpha_grad_y**2 + 1e-8)
    
        # 2. 创建区域掩码
        transition_mask = (trimap == 0.5).float()  # 过渡区域
        known_mask = (trimap != 0.5).float()       # 已知区域（前景+背景）
        unknown_mask = sample_map  # 训练时的未知区域掩码
        
        # 3. 在不同区域应用不同的梯度策略
    
        # 策略A: 过渡区域 - alpha梯度应与原图梯度正相关
        # 即：原图梯度大处，alpha梯度也应较大
        transition_weight = transition_mask * unknown_mask
        
        # 计算相关性损失（只在过渡区域）
        if torch.sum(transition_weight) > 0:
            # 归一化梯度到相同尺度
            image_grad_norm = (image_grad - image_grad.min()) / (image_grad.max() - image_grad.min() + 1e-8)
            
            # alpha梯度应该与原图梯度成正比，但不要求完全相等
            # 使用Huber损失，更好地处理异常值
            grad_diff = torch.abs(alpha_grad - image_grad_norm)
            transition_loss = torch.where(
                grad_diff < 1.0,
                0.5 * grad_diff**2,
                grad_diff - 0.5
            )
            transition_loss = torch.mean(transition_loss * transition_weight)
        else:
            transition_loss = torch.tensor(0.0, device=alpha_pred.device)
    
        # 策略B: 已知区域 - alpha应保持平滑（梯度尽量小）
        # 惩罚大的alpha梯度变化
        smooth_weight = known_mask * unknown_mask
        
        if torch.sum(smooth_weight) > 0:
            # L2惩罚大梯度
            smooth_loss = torch.mean(alpha_grad**2 * smooth_weight)
        else:
            smooth_loss = torch.tensor(0.0, device=alpha_pred.device)
        
        # 4. 综合损失
        total_loss = transition_loss * 5.0 + smooth_loss * 2.0
        
        scale = sample_map.shape[0] * 512 * 512 / torch.sum(sample_map)
        total_loss = total_loss * scale
    
        return dict(gradient_loss=total_loss)

    '''
    def get_loss(self, loss, sample_map, outputs, targets):
        loss_map = {
            'l1': self.l1_loss,
            'known': self.known_l1_loss,
            'consistency': self.consistency_loss,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](sample_map, outputs, targets)
    '''

    def get_loss(self, loss, sample_map, outputs, targets):
        loss_map = {
            'l1': self.l1_loss,
            'known': self.known_l1_loss,
            'consistency': self.consistency_loss,
            #'dift_consistency': self.dift_consistency_loss,  # 添加
            #'gradient': self.gradient_loss,
            'guided_filter': self.guided_filter_loss,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](sample_map, outputs, targets)

    def forward(self, outputs, targets):
        sample_map = torch.zeros_like(targets['trimap'])
        sample_map[targets['trimap'] == 0.5] = 1  # 0.5!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        losses = dict()
        for loss in self.losses:
            losses.update(self.get_loss(loss, sample_map, outputs, targets))
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                # sample_map = torch.zeros_like(aux_targets['trimap'])
                # sample_map[aux_targets['trimap'] == 0.5] = 1
                for loss in self.aux_losses:
                    l_dict = self.get_loss(loss, sample_map, aux_outputs, targets)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        return losses


#-----------------Laplacian Loss-------------------------#
def laplacian_loss(pred, true, max_levels=5):
    kernel = gauss_kernel(device=pred.device, dtype=pred.dtype)
    pred_pyramid = laplacian_pyramid(pred, kernel, max_levels)
    true_pyramid = laplacian_pyramid(true, kernel, max_levels)
    loss = 0
    for level in range(max_levels):
        loss += (2 ** level) * F.l1_loss(pred_pyramid[level], true_pyramid[level])
    return loss / max_levels

def laplacian_pyramid(img, kernel, max_levels):
    current = img
    pyramid = []
    for _ in range(max_levels):
        current = crop_to_even_size(current)
        down = downsample(current, kernel)
        up = upsample(down, kernel)
        diff = current - up
        pyramid.append(diff)
        current = down
    return pyramid

def gauss_kernel(device='cpu', dtype=torch.float32):
    kernel = torch.tensor([[1,  4,  6,  4, 1],
                        [4, 16, 24, 16, 4],
                        [6, 24, 36, 24, 6],
                        [4, 16, 24, 16, 4],
                        [1,  4,  6,  4, 1]], device=device, dtype=dtype)
    kernel /= 256
    kernel = kernel[None, None, :, :]
    return kernel

def gauss_convolution(img, kernel):
    B, C, H, W = img.shape
    img = img.reshape(B * C, 1, H, W)
    img = F.pad(img, (2, 2, 2, 2), mode='reflect')
    img = F.conv2d(img, kernel)
    img = img.reshape(B, C, H, W)
    return img

def downsample(img, kernel):
    img = gauss_convolution(img, kernel)
    img = img[:, :, ::2, ::2]
    return img

def upsample(img, kernel):
    B, C, H, W = img.shape
    out = torch.zeros((B, C, H * 2, W * 2), device=img.device, dtype=img.dtype)
    out[:, :, ::2, ::2] = img * 4
    out = gauss_convolution(out, kernel)
    return out

def crop_to_even_size(img):
    H, W = img.shape[2:]
    H = H - H % 2
    W = W - W % 2
    return img[:, :, :H, :W]