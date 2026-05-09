from detectron2.engine import AMPTrainer
import torch
import time
import os
from PIL import Image
from torchvision.transforms import functional as F

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

class MattingTrainer(AMPTrainer):
    def __init__(self, model, data_loader, optimizer, cfg=None, grad_scaler=None):
        super().__init__(model, data_loader, optimizer, grad_scaler=None)
        self.data_loader_iter = iter(cycle(self.data_loader))
        self.gf_start_iter = cfg.train.gf_start_iter if cfg else 5700

        self.device = cfg.train.device if cfg else "cuda"
        
        # 定义保存可视化结果的文件夹
        if cfg:
            self.vis_output_dir = os.path.join(cfg.train.output_dir, "visualizations")
        else:
            self.vis_output_dir = './output_of_train/vitmatte_s_am2k_guided_DDC_test3_100ep'

        self.test_img_path = 'datasets/validation/original/m_7be04a91.jpg'
        self.test_alpha_path = 'datasets/validation/sam_mask/m_7be04a91.png' # 假设 Trimap 在这里

    def run_step(self):
        """
        Implement the AMP training logic.
        """
        assert self.model.training, "[AMPTrainer] model was changed to eval mode!"
        assert torch.cuda.is_available(), "[AMPTrainer] CUDA is required for AMP training!"
        from torch.cuda.amp import autocast

        #matting pass
        start = time.perf_counter()        
        data = next(self.data_loader_iter)
        data_time = time.perf_counter() - start
       
        # with autocast():
        with torch.autocast(device_type="cuda", dtype=torch.float32):
            loss_dict = self.model(data)

            if isinstance(loss_dict, torch.Tensor):
                losses = loss_dict
                loss_dict = {"total_loss": loss_dict}
            else:
                
                if self.iter < self.gf_start_iter:
                    loss_dict["guided_loss"] = loss_dict["guided_loss"] * 0.0
                else:
                    loss_dict["guided_loss"] = loss_dict["guided_loss"]
                    #loss_dict["known_l1_loss"] = loss_dict["known_l1_loss"]
                
                losses = sum(loss_dict.values())
            '''
            self.model.eval()
            test_img = cv2.imread('datasets/train/original/m_0a6c2018.jpg')
            pred = self.model(test_img)
            path_name = path + "/时间戳_" + name
            cv2.imwrite(pred, path_name)
            self.model.train()
            '''

        self.optimizer.zero_grad()
        self.grad_scaler.scale(losses).backward()

        self._write_metrics(loss_dict, data_time)

        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()

        if self.iter % 200 == 0:
            self.run_visualization()

    def run_visualization(self):
        """
        结合了 run_one_image.py 的 PIL 处理风格，
        同时修复了 Trimap 缺失和 Resize 问题。
        """
        self.model.eval()
        
        try:
            with torch.no_grad():
                # 1. 准备路径
                if not os.path.exists(self.test_img_path) :
                    return

                # 2. 加载图片 (使用 PIL，保持和 run_one_image.py 一致的风格)
                # PIL 读取的是 RGB，不需要像 cv2 那样转通道
                original_pil_image = Image.open(self.test_img_path).convert('RGB')
                original_pil_alpha = Image.open(self.test_alpha_path).convert('L')
                #trimap_pil = Image.open(self.test_tri_path).convert('L') # Trimap 是灰度

                # 3. 预处理 (转 Tensor)
                
                # 不强制 resize 到 256，保持原图或训练尺寸，这样看的清楚
                # 如果显存不够，可以在这里 resize 到 512x512
                # original_pil = original_pil.resize((512, 512), Image.LANCZOS)
                # trimap_pil = trimap_pil.resize((512, 512), Image.NEAREST)

                img_tensor = F.to_tensor(original_pil_image).unsqueeze(0).to(self.device)
                alpha_tensor = F.to_tensor(original_pil_alpha).unsqueeze(0).to(self.device)
                #tri_tensor = F.to_tensor(trimap_pil).unsqueeze(0).to(self.device)

                # 4. 构造输入
                inputs = {
                    "image": img_tensor, 
                    "alpha": alpha_tensor, 
                    #"trimap": tri_tensor
                }

                # 5. 推理
                outputs = self.model(inputs)

                # 6. 获取 Alpha (兼容字典或Tensor返回)
                if isinstance(outputs, list):
                    if len(outputs) > 0 and isinstance(outputs[0], dict):
                        # 取出第一张图的结果
                        pred_alpha = outputs[0].get("phas", outputs[0].get("pred_alpha"))
                    elif len(outputs) > 0 and isinstance(outputs[0], torch.Tensor):
                        pred_alpha = outputs[0]

                # 情况 B: 有些模型直接返回 Dict
                elif isinstance(outputs, dict):
                    pred_alpha = outputs.get("phas", outputs.get("pred_alpha"))
                
                # 情况 C: 直接返回 Tensor
                elif isinstance(outputs, torch.Tensor):
                    pred_alpha = outputs

                # 7. 保存结果 (使用 run_one_image.py 的风格)
                if pred_alpha is not None:
                    # 取出 Tensor: [1, 1, H, W] -> [1, H, W]
                    if pred_alpha.dim() == 4:
                        pred_alpha = pred_alpha.squeeze(0)
                    
                    # 转为 PIL
                    pred_pil = F.to_pil_image(pred_alpha.cpu())
                    
                    # 保存
                    save_name = f"iter_{self.iter:06d}.png"
                    save_path = os.path.join(self.vis_output_dir, save_name)
                    pred_pil.save(save_path)
                    # print(f"Saved: {save_path}")

        except Exception as e:
            print(f"Visualization Error: {e}")
        
        finally:
            self.model.train()