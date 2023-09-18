import torch
import numpy as np

from tqdm import tqdm
from data_loader import set_loader
from models import set_model



def metric_function(output, target, num_classes = 13):
    
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
    for data in tqdm(loader):
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
    for data in tqdm(loader):
        image, gt = data
        batch_size = image.shape[0]
        output = model(image.cuda())
        loss = loss_function(output, gt.cuda().squeeze(dim = 1))
        metric = metric_function(output, gt.cuda())
        
        loss_value += loss.detach().cpu().item()*batch_size
        metric_value += metric*batch_size
        
        total_batch += batch_size
        
    return loss_value / total_batch, metric_value / total_batch

        
if __name__ == '__main__':
    batch_size = 32
    epochs = 50
    train_loader, val_loader = set_loader(batch_size)
    
    
    model = set_model().cuda()
    loss_function = torch.nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr = 1e-1)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer = optimizer, 
        milestones = [10*(i+1) for i in range(epochs // 10 + 1)],
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
        val_loss, val_metric = evaluate(train_loader, loss_function, model, metric_function)
        
        result_dict['train metric'].append(train_metric)
        result_dict['val metric'].append(val_metric)
        result_dict['train loss'].append(train_loss)
        result_dict['val loss'].append(val_loss)

        for k, v in result_dict.items():
            print(k, ':', v[-1])
        
    