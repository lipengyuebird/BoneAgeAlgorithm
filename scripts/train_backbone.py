#!venv/Scripts/python
# -*- coding: utf-8 -*-
# @file     : train_backbone.py
# @author   : lipengyu @CAICT
# @datetime : 2021/7/9 16:41 
# @software : PyCharm

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Lambda, Compose, Resize, RandomCrop
from torchvision.models import resnet50

from util.config_loader import config
from dataset import BackBoneDataSet

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

train_transform = Compose([Resize(int(config['train']['backbone']['input_size'])), RandomCrop(448), ToTensor()])
resNet50 = resnet50(pretrained=False, progress=False, zero_init_residual=True, num_classes=1001).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(resNet50.parameters(), lr=float(config['train']['backbone']['init_lr']))
train_data_loader = DataLoader(BackBoneDataSet(transform=train_transform, train_or_test='train'),
                               batch_size=int(config['train']['backbone']['batch_size']))
val_data_loader = DataLoader(BackBoneDataSet(transform=train_transform, train_or_test='val'),
                             batch_size=int(config['train']['backbone']['batch_size']))
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=10,
                                                       verbose=False,
                                                       threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0,
                                                       eps=1e-08)


def train(data_loader, model, loss_fn1, optimizer1):
    size = len(data_loader.dataset)
    for batch, (x, y) in enumerate(data_loader):
        x, y = x.to(device), y.to(device)

        # Compute prediction error
        pred = model(x)
        loss = loss_fn1(pred, y)

        # Backpropagation
        optimizer1.zero_grad()
        loss.backward()
        optimizer1.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(x)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def val(data_loader, model, loss_fn1):
    size = len(data_loader.dataset)
    num_batches = len(data_loader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            test_loss += loss_fn1(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


for t in range(int(config['train']['backbone']['max_epochs'])):
    print(f"Epoch {t + 1}\n-------------------------------")
    train(train_data_loader, resNet50, loss_fn, optimizer)
    val(val_data_loader, resNet50, loss_fn)
