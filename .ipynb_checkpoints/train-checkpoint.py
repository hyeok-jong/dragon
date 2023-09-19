import torch
import numpy as np

from tqdm import tqdm
from data_loader import set_loader
from models import set_model


def inv_save(image, gt, pred):
    colormap = torch.tensor([
    [255, 255, 255],
    [255, 0, 0],   
    [0, 255, 0], 
    [0, 0, 255],  
    [255, 255, 0], 
    [255, 0, 255], 
    [0, 255, 255], 
    [128, 128, 0], 
    [128, 0, 128],
    [0, 128, 128],
    [255, 165, 0],
    [128, 0, 0],
    [0, 128, 0],
    [0, 0, 128],  
    ], dtype=torch.float32) / 255.0 
    gt = colormap[gt].squeeze().permute(2,0,1)
    pred = colormap[pred].squeeze().permute(2,0,1)
    
    mean = torch.tensor([0.485, 0.456, 0.406]).unsqueeze(dim = 1).unsqueeze(dim = 2)
    std = torch.tensor([0.229, 0.224, 0.225]).unsqueeze(dim = 1).unsqueeze(dim = 2)
    image = image*std+mean
    
    return [
        transforms.functional.to_pil_image(image), 
        transforms.functional.to_pil_image(gt), 
        transforms.functional.to_pil_image(image*.5+gt*.5),
        transforms.functional.to_pil_image(pred), 
        transforms.functional.to_pil_image(image*.5+pred*.5),
    ]




def metric_function(output, target, num_classes = 14):
    
    preds = torch.argmax(output, dim=1)
    intersection = torch.zeros(num_classes).float().to(output.device)
    union = torch.zeros(num_classes).float().to(output.device)
    
    for cls in range(num_classes):
        intersection[cls] = torch.sum((preds == cls) & (target == cls))
        union[cls] = torch.sum((preds == cls) | (target == cls))
    
    iou = intersection / (union + 1e-6)
    miou = torch.mean(iou)
    
    return miou.item()


def train(loader, optimizer, loss_function, model, metric_function):
    loss_value = 0
    metric_value = 0
    total_batch = 0
    
    model.train()
    for data in loader:
        image, gt = data
        batch_size = image.shape[0]
        output = model(image.cuda())
        loss = loss_function(output, gt.cuda().squeeze(dim = 1))
        metric = metric_function(output, gt.cuda())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_value += loss.detach().cpu().item()*batch_size
        metric_value += metric*batch_size
        
        total_batch += batch_size
        
        
    return loss_value / total_batch, metric_value / total_batch


@torch.no_grad()
def evaluate(loader, loss_function, model, metric_function):
    loss_value = 0
    metric_value = 0
    total_batch = 0
    model.eval()
    for data in loader:
        image, gt = data
        batch_size = image.shape[0]
        output = model(image.cuda())
        loss = loss_function(output, gt.cuda().squeeze(dim = 1))
        metric = metric_function(output, gt.cuda())
        
        loss_value += loss.detach().cpu().item()*batch_size
        metric_value += metric*batch_size
        
        total_batch += batch_size
        
    return loss_value / total_batch, metric_value / total_batch, image, gt, torch.argmax(output, dim=1).detach().cpu()

        
if __name__ == '__main__':
    batch_size = 16
    epochs = 100
    train_loader, val_loader = set_loader(batch_size)
    
    
    model = set_model().cuda()
    loss_function = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr = 1)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer = optimizer, 
        milestones = [20*(i+1) for i in range(epochs // 10 + 1)],
        gamma = 0.1, 
        last_epoch = -1,
        verbose = False)
    
    result_dict = {
        'train metric' : [],
        'train loss' : [],
        'val metric' : [],
        'val loss' : []
    }
    
    for epoch in tqdm(range(epochs)):
        train_loss, train_metric = train(train_loader, optimizer, loss_function, model, metric_function)
        val_loss, val_metric, image, gt, pred = evaluate(val_loader, loss_function, model, metric_function)
        image, gt, image_gt, pred, image_pred = inv_save(image, gt, pred)
        image.save(f'{epoch}_image.png')
        gt.save(f'{epoch}_gt.png')
        image_gt.save(f'{epoch}_image_gt.png')
        pred.save(f'{epoch}_pred.png')
        image_pred.save(f'{epoch}_image_pred.png')
        
        result_dict['train metric'].append(train_metric)
        result_dict['val metric'].append(val_metric)
        result_dict['train loss'].append(train_loss)
        result_dict['val loss'].append(val_loss)

        for k, v in result_dict.items():
            print(k, ':', v[-1])
        
    