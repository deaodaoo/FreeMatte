from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from detectron2.config import LazyCall as L
from torch.utils.data.distributed import DistributedSampler

from data import ImageFileTrainAM2K, DataGeneratorAM2K

# Dataloader
train_dataset = DataGeneratorAM2K(
    data=ImageFileTrainAM2K(
        trimap_dir='datasets/train/trimap',
        image_dir='datasets/train/original',
        alpha_dir='datasets/train/sam_train_masks',
        alpha_ext='.png',
        trimap_ext='.png',
        #image_ext='.png',
        image_ext='.jpg',
    ),
    phase='train'
)

dataloader = OmegaConf.create()
dataloader.train = L(DataLoader)(
    dataset=train_dataset,
    batch_size=12,
    shuffle=True,  # 改为True，单GPU训练时不需要DistributedSampler
    num_workers=0,
    pin_memory=False,
    # 移除强制使用的DistributedSampler
    drop_last=True
)

