from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from detectron2.config import LazyCall as L
from torch.utils.data.distributed import DistributedSampler

from data import ImageFileTrainAM2K, DataGeneratorAM2K

# AM-2K
train_dataset = DataGeneratorAM2K(
    data=ImageFileTrainAM2K(
        trimap_dir='../datasets/train/trimap',
        image_dir='../datasets/train/original',
        # 
        alpha_dir='../datasets/codes/batch_run_sam1B/AM2K_training/point',
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
    shuffle=True,
    num_workers=0,
    pin_memory=False,
    drop_last=True
)

