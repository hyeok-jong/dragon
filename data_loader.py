import random
import pathlib
import torch
import numpy as np

from PIL import Image
from torchvision import transforms


class lee_dragon(torch.utils.data.Dataset):
    def __init__(self, image_dirs, gt_dirs):
        self.image_dirs = image_dirs
        self.gt_dirs = gt_dirs
        self.train_t = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Resize([2048 // 3, 1024 //3]),
            transforms.RandomResizedCrop(size = [2048 // 3, 1024 //3], scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333),)
            transforms.ColorJitter(brightness = 0.5, contrast = 1, saturation = 0.1, hue = 0.5),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        self.gt_t = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Resize([2048 // 3, 1024 //3], interpolation = transforms.InterpolationMode.NEAREST),
            transforms.RandomResizedCrop(size = [2048 // 3, 1024 //3], scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333),)
            ])
        
        assert len(self.gt_dirs) == len(self.image_dirs), 'fuck wrong'
        
    def __len__(self):
        return len(self.image_dirs)

    def __getitem__(self, index):
        seed = np.random.randint(2147483647)
        random.seed(seed)
        torch.manual_seed(seed)
        
        dir = self.image_dirs[index]
        image = Image.open(dir)
        if self.train_t:
            image = self.train_t(image)
            
        dir = self.gt_dirs[index]
        gt = Image.open(dir)
        if self.gt_t:
            gt = self.gt_t(gt)
            gt[gt == 255] = 13
    
        return image, gt.to(torch.long)
    
train_image_dirs = sorted([str(i) for i in pathlib.Path('./open/train_source_image').rglob('*.png')])
train_gt_dirs = sorted([str(i) for i in pathlib.Path('./open/train_source_gt').rglob('*.png')])

val_image_dirs = sorted([str(i) for i in pathlib.Path('./open/val_source_image').rglob('*.png')])
val_gt_dirs = sorted([str(i) for i in pathlib.Path('./open/val_source_gt').rglob('*.png')])


def set_loader(batch_size):
    train_loader = torch.utils.data.DataLoader(
        dataset = lee_dragon(train_image_dirs, train_gt_dirs),
        batch_size = batch_size,
        num_workers = 4,
        pin_memory = True,
        shuffle = True
    )
    
    val_loader = torch.utils.data.DataLoader(
        dataset = lee_dragon(val_image_dirs, val_gt_dirs),
        batch_size = batch_size,
        num_workers = 4,
        pin_memory = True,
        shuffle = True
    )
    print(len(train_loader.dataset), len(val_loader.dataset))    
    return train_loader, val_loader