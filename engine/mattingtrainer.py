from detectron2.engine import AMPTrainer
from detectron2.checkpoint import DetectionCheckpointer
import torch
import time
import copy
import os
import logging
from PIL import Image
from torchvision.transforms import functional as F
from configs.base.base import exp_name

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

        self.need_load_prev_model = False
        self.prev_model = None
        
        if cfg:
            self.vis_output_dir = os.path.join(cfg.train.output_dir, "visualizations")
        else:
            self.vis_output_dir = './output_of_train/' + exp_name

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

        original_level = logging.getLogger("detectron2").level
       
        with torch.autocast(device_type="cuda", dtype=torch.float32):
            loss_dict = self.model(data, self.iter, self.max_iter, self.prev_model)

            if isinstance(loss_dict, torch.Tensor):
                losses = loss_dict
                loss_dict = {"total_loss": loss_dict}
            else:
                losses = sum(loss_dict.values())

        self.optimizer.zero_grad()
        self.grad_scaler.scale(losses).backward()

        self._write_metrics(loss_dict, data_time)

        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()
