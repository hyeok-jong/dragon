import random
import pathlib
import torch
import numpy as np

from PIL import Image
from torchvision import transforms




class lee_dragon(torch.utils.data.Dataset):
    def __init__(self, image_dirs, gt_dirs, train_flag = True):
        self.image_dirs = image_dirs
        self.gt_dirs = gt_dirs

        self.train_flag = train_flag
        self.train_image_t = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Resize([2048 // 3, 1024 //3]),
            transforms.RandomResizedCrop(size = [1024 // 3, 2048 //3], scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333),),
            transforms.ColorJitter(brightness = 0.5, contrast = 1, saturation = 0.1, hue = 0.5),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        self.train_gt_t = transforms.Compose([
            transforms.PILToTensor(),
            # transforms.Resize([2048 // 3, 1024 //3], interpolation = transforms.InterpolationMode.NEAREST),
            transforms.RandomResizedCrop(
                size = [1024 // 3, 2048 //3], 
                scale=(0.08, 1.0), 
                ratio=(0.75, 1.3333333333333333),
                interpolation = transforms.InterpolationMode.NEAREST),
            ])

        self.valid_image_t = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([1024 // 3, 2048 //3]),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        self.valid_gt_t = transforms.Compose([
            transforms.PILToTensor(),
            transforms.Resize([1024 // 3, 2048 //3], interpolation = transforms.InterpolationMode.NEAREST),
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

        dir = self.gt_dirs[index]
        gt = Image.open(dir)

        if self.train_flag:
            image = self.train_image_t(image)
            gt = self.train_gt_t(gt)
        elif not self.train_flag:
            image = self.valid_image_t(image)
            gt = self.valid_gt_t(gt)

        
        gt[gt == 255] = 12
    
        return image, gt.to(torch.long)
    
train_image_dirs = sorted([str(i) for i in pathlib.Path('./open/train_source_image').rglob('*.png')])
train_gt_dirs = sorted([str(i) for i in pathlib.Path('./open/train_source_gt').rglob('*.png')])

val_image_dirs = sorted([str(i) for i in pathlib.Path('./open/val_source_image').rglob('*.png')])
val_gt_dirs = sorted([str(i) for i in pathlib.Path('./open/val_source_gt').rglob('*.png')])


def set_loader(batch_size):
    train_loader = torch.utils.data.DataLoader(
        dataset = lee_dragon(train_image_dirs, train_gt_dirs, True),
        batch_size = batch_size,
        num_workers = 4,
        pin_memory = True,
        shuffle = True
    )
    
    val_loader = torch.utils.data.DataLoader(
        dataset = lee_dragon(val_image_dirs, val_gt_dirs, False),
        batch_size = batch_size,
        num_workers = 4,
        pin_memory = True,
        shuffle = True
    )
    print(len(train_loader.dataset), len(val_loader.dataset))    
    return train_loader, val_loader