from .common.train import train
from .common.model import model
from .common.optimizer import optimizer
from .common.scheduler import lr_multiplier
from .common.dataloader import dataloader
from .base.base import exp_name

train.max_iter = int(1800 / 16 / 2 * 125)
train.checkpointer.period = int(1800 / 16 / 2 * 10)

optimizer.lr=5e-4
lr_multiplier.scheduler.values=[1.0, 0.1, 0.01]
lr_multiplier.scheduler.milestones=[int(1800 / 16 / 2 * 60), int(1800 / 16 / 2 * 110)]
lr_multiplier.scheduler.num_updates = train.max_iter
lr_multiplier.warmup_length = 250 / train.max_iter

train.init_checkpoint = '../pretrained/dino_vit_s_fna_3_channels.pth'
train.output_dir = './output_of_train/' + exp_name

dataloader.train.batch_size=8
dataloader.train.num_workers=0

train.eval_period = 200
