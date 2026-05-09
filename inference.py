import os
import torch
import typing
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as torch_F
from torchvision.transforms import functional as F
from os.path import join as opj
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.engine import default_argument_parser

from configs.base.base import exp_name

import warnings
warnings.filterwarnings('ignore')

# Dataset and Dataloader
def collate_fn(batched_inputs):
    rets = dict()
    for k in batched_inputs[0].keys():
        rets[k] = torch.stack([_[k] for _ in batched_inputs])
    return rets


class Composition_1k(Dataset):
    def __init__(self, data_dir, mask_dir):
        self.data_dir = data_dir
        self.mask_dir = mask_dir

        def filter_image_files(files):
            image_extensions = {'.png', '.jpg', '.jpeg'}
            return [f for f in files if os.path.splitext(f)[1].lower() in image_extensions]
        
        self.original_names = sorted(filter_image_files(os.listdir(opj(self.data_dir, 'original'))))
        self.phas_names = sorted(filter_image_files(os.listdir(self.mask_dir)))
        self.trimap_names = sorted(filter_image_files(os.listdir(opj(self.data_dir, 'trimap'))))

    def __len__(self):
        return len(self.original_names)

    def __getitem__(self, idx):
        tris = Image.open(opj(self.data_dir, 'trimap', self.trimap_names[idx]))
        imgs = Image.open(opj(self.data_dir, 'original', self.original_names[idx]))
        mask = Image.open(opj(self.mask_dir, self.phas_names[idx])).convert('L')

        sample = {}

        sample['trimap'] = F.to_tensor(tris) * 255.
        sample['image'] = F.to_tensor(imgs)
        sample['alpha'] = F.to_tensor(mask)
        sample['image_name'] = self.original_names[idx]

        return sample


def matting_inference(
        config_dir='',
        checkpoint_dir='',
        inference_dir='',
        data_dir='',
        mask_dir = '',
):
    # initializing model
    cfg = LazyConfig.load(config_dir)
    model = instantiate(cfg.model)
    model.to(cfg.train.device)
    model.eval()

    checkpoint = torch.load(checkpoint_dir, map_location="cpu", weights_only=False)
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict, strict=False)

    # initializing dataset
    composition_1k_dataloader = DataLoader(
        dataset=Composition_1k(
            data_dir=data_dir,
            mask_dir=mask_dir,
        ),
        shuffle=False,
        batch_size=1,
    )

    print("inference_dir: ", inference_dir)
    os.makedirs(inference_dir, exist_ok=True)

    for data in tqdm(composition_1k_dataloader):
        with torch.no_grad():
            for k in data.keys():
                if k == 'image_name':
                    continue
                else:
                    data[k].to(model.device)
            output = model(data)['phas'].flatten(0, 2)
            output = F.to_pil_image(output)
            output.save(opj(inference_dir, data['image_name'][0].split('.')[0] + '.png'))
            torch.cuda.empty_cache()


if __name__ == '__main__':
    # add argument we need:
    parser = default_argument_parser()
    parser.add_argument('--config-dir', type=str, required=False, default='configs/vitmatte_s_am2k_100ep.py')
    parser.add_argument('--checkpoint-dir', type=str, required=False,
                    default=str(Path(__file__).resolve().parent / 'your model path'))
    parser.add_argument('--inference-dir', type=str, required=False,
                    default=str(Path(__file__).resolve().parent / 'out_inference'))
    parser.add_argument('--data-dir', type=str, required=False,
                    default=str(Path(__file__).resolve().parent / '../datasets/validation'))
    parser.add_argument('--mask-dir', type=str, required=False,
                    default=str(Path(__file__).resolve().parent / '../datasets/codes/batch_run_sam1B/AM2K_validation'))
    args = parser.parse_args()
    matting_inference(
        config_dir=args.config_dir,
        checkpoint_dir=args.checkpoint_dir,
        inference_dir=args.inference_dir,
        data_dir=args.data_dir,
        mask_dir=args.mask_dir
    )