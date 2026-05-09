import cv2
import numpy as np
import torch
import skfmm
from scipy.ndimage import distance_transform_edt


pb_mode = "all"

# Hyperparameters for resistance sensitivity analysis
lambda_resistance = 10.0
color_weight = 0.8

# Hyperparameters for spatial modulation and sparse supervision
spatial_ratio_global = 0.1
high_thresh_global = 0.8
def build_resistance(
    color_diff,
    grad_term,
    D_norm,
    W,
    beta=2.0,
):
    """
    Main setting:
        f = 1 + W * lambda_resistance *
            (color_weight * color_diff + (1 - color_weight) * grad_term)
    """
    if pb_mode == "all":
        f = 1.0 + W * lambda_resistance * (
            color_weight * color_diff
            + (1.0 - color_weight) * grad_term
        )

    elif pb_mode == "only_color":
        f = 1.0 + lambda_resistance * color_diff

    elif pb_mode == "only_gradient":
        f = 1.0 + lambda_resistance * grad_term
    
    elif pb_mode == "only_w":
        f = 1.0 + W

    elif pb_mode == "none":
        f = 1.0

    else:
        raise ValueError(f"Unknown pb_mode: {pb_mode}")

    return f.astype(np.float32)



def compute_eikonal_from_tensor(
    img_tensor,
    mask_tensor,
    beta=2.0,
    spatial_ratio=None,
    eps=1e-8,
    normalize=False,
    return_gradients=False,
):
    device = img_tensor.device
    if spatial_ratio is None:
        spatial_ratio = spatial_ratio_global
    else:
        print("[spatial_ratio] ERROR, USE INPUT PARAM")

    img_np = img_tensor.detach().float().cpu().squeeze().numpy()
    mask_np = mask_tensor.detach().float().cpu().squeeze().numpy()

    if img_np.ndim != 3 or img_np.shape[0] != 3:
        raise ValueError(
            f"img_tensor after squeeze should have shape [3, H, W], got {img_np.shape}"
        )
    if mask_np.ndim != 2:
        raise ValueError(
            f"mask_tensor after squeeze should have shape [H, W], got {mask_np.shape}"
        )

    if img_np.max() > 1.0 or img_np.min() < 0.0:
        img_np = np.clip(img_np, 0.0, 255.0)
        if img_np.max() > 1.0:
            img_np = img_np / 255.0

    fg = mask_np > 0.5

    _, h, w = img_np.shape
    if not np.any(fg) or np.all(fg):
        zero_tensor = torch.zeros((1, h, w), dtype=torch.float32, device=device)
        if return_gradients:
            return zero_tensor, zero_tensor, zero_tensor, zero_tensor, zero_tensor
        return zero_tensor

    gray = (
        0.2989 * img_np[0] +
        0.5870 * img_np[1] +
        0.1140 * img_np[2]
    ).astype(np.float32)

    fg_colors = img_np[:, fg]  # shape: (3, N)
    mu_fg = np.mean(fg_colors, axis=1).reshape(3, 1, 1)         # (3,1,1)
    sigma_fg = np.std(fg_colors, axis=1).reshape(3, 1, 1) + eps # (3,1,1)

    diff = (img_np - mu_fg) / sigma_fg
    color_diff = np.sqrt(np.sum(diff ** 2, axis=0)).astype(np.float32)  # (H,W)
    color_diff = color_diff / (color_diff.max() + eps)

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx**2 + gy**2).astype(np.float32)
    grad_mag = grad_mag / (grad_mag.max() + eps)

    sigma_g_adaptive = np.mean(grad_mag) + 1e-5
    grad_term = np.exp(-(grad_mag ** 2) / (sigma_g_adaptive ** 2 + eps)).astype(np.float32)

    bg = ~fg
    D = distance_transform_edt(bg).astype(np.float32)
    D_norm = D / (D.max() + eps)

    sigma_adaptive = max(D.max() * spatial_ratio, 1.0)
    W = np.exp(-(D ** 2) / (2.0 * sigma_adaptive ** 2 + eps)).astype(np.float32)

    f = build_resistance(
        color_diff=color_diff,
        grad_term=grad_term,
        D_norm=D_norm,
        W=W,
        beta=beta,
    )

    speed = (1.0 / (f + eps)).astype(np.float32)

    phi = np.where(fg, 0.0, 1.0).astype(np.float32)
    T = skfmm.travel_time(phi, speed).astype(np.float32)

    if normalize:
        T = (T - T.min()) / (T.max() - T.min() + eps)

    T_tensor = torch.from_numpy(T).float().unsqueeze(0).to(device)

    if return_gradients:
        Ty, Tx = np.gradient(T)
        Ty = Ty.astype(np.float32)
        Tx = Tx.astype(np.float32)
        grad_T = np.sqrt(Tx ** 2 + Ty ** 2).astype(np.float32)

        Tx_tensor = torch.from_numpy(Tx).float().unsqueeze(0).to(device)
        Ty_tensor = torch.from_numpy(Ty).float().unsqueeze(0).to(device)
        grad_T_tensor = torch.from_numpy(grad_T).float().unsqueeze(0).to(device)
        f_tensor = torch.from_numpy(f).float().unsqueeze(0).to(device)

        return T_tensor, Tx_tensor, Ty_tensor, grad_T_tensor, f_tensor

    return T_tensor



def compute_absolute_bg_mask(u_x, full_bg_mask, high_thresh=None):
    if high_thresh is None:
        high_thresh = high_thresh_global
    else:
        print("[high_thresh] ERROR, USE INPUT PARAM")
    
    B, C, H, W = full_bg_mask.shape
    
    u_x_flat = u_x.view(B, -1)
    u_max = u_x_flat.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
    
    T_norm = u_x / (u_max + 1e-8)
    P_indep = torch.clamp(T_norm / (high_thresh + 1e-8), min=0.0, max=1.0)
    P_indep = P_indep * full_bg_mask.float()
    
    random_matrix = torch.rand_like(P_indep)
    sampled_mask = (random_matrix < P_indep).float()
    return sampled_mask
