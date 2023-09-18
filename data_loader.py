import pathlib
import torch

from PIL import Image
from torchvision import transforms

class lee_dragon(torch.utils.data.Dataset):
    def __init__(self, image_dirs, gt_dirs):
        self.image_dirs = image_dirs
        self.gt_dirs = gt_dirs
        self.train_t = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([2048 // 3, 1024 //3]),
            transforms.ColorJitter(brightness = 0.5, contrast = 1, saturation = 0.1, hue = 0.5),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        self.test_t = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([2048 // 3, 1024 //3]),
            ])
        
        assert len(self.gt_dirs) == len(self.image_dirs), 'fuck wrong'
        
    def __len__(self):
        return len(self.image_dirs)

    def __getitem__(self, index):
        dir = self.image_dirs[index]
        image = Image.open(dir)
        if self.train_t:
            image = self.train_t(image)
            
        dir = self.gt_dirs[index]
        gt = Image.open(dir)
        if self.test_t:
            gt = self.test_t(gt)
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