from .common.train import train
from .common.model import model
from .common.optimizer import optimizer
from .common.scheduler import lr_multiplier
from .common.dataloader import dataloader

train.max_iter = int(1800 / 16 / 2 * 125)
train.checkpointer.period = int(1800 / 16 / 2 * 10)   # 从10改为20，减少检查点保存频率

optimizer.lr=5e-4
lr_multiplier.scheduler.values=[1.0, 0.1, 0.01]
lr_multiplier.scheduler.milestones=[int(1800 / 16 / 2 * 60), int(1800 / 16 / 2 * 110)]
lr_multiplier.scheduler.num_updates = train.max_iter
lr_multiplier.warmup_length = 250 / train.max_iter

train.init_checkpoint = './pretrained/dino_vit_s_fna_3_channels.pth'
train.output_dir = './output_of_train/vitmatte_s_am2k_guided_DDC_test3_100ep'

dataloader.train.batch_size=12  # 改为1，与数据加载器配置保持一致
dataloader.train.num_workers=0  # 改为0，减少内存使用

train.eval_period = 200  # 从50改为200，减少评估频率
