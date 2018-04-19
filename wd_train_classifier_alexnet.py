# Code taken from here : http://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

from config import domainData
from config import num_classes as NUM_CLASSES
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from logger import Logger
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.autograd import Function
import torch.nn.functional as F
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
from logit import flatten
import time
import os
import copy
import itertools
from tqdm import *

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(224),
        # transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(224),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

use_gpu = True and torch.cuda.is_available()
train_dir = domainData['amazon'] # 'amazon', 'dslr', 'webcam'
val_dir = domainData['webcam']
num_classes = NUM_CLASSES['office']
EPOCHS = 50
gamma = 0.001
power = 0.75
LR = 2e-4
W_Decay = 1e-7
BATCH_SIZE = 64
log = False
exp_name = 'wd_jstCls_Rsnt2blk'

print("use gpu: ", use_gpu)

torch.manual_seed(7)
if use_gpu:
    torch.cuda.manual_seed(7)

image_datasets = {'train' : datasets.ImageFolder(train_dir,
                                          data_transforms['train']),
                  'val' : datasets.ImageFolder(val_dir,
                                          data_transforms['val'])
                 }

dataloaders = {'train' : torch.utils.data.DataLoader(image_datasets['train'], batch_size=BATCH_SIZE, shuffle=True, num_workers=4, drop_last=False),
                'val' : torch.utils.data.DataLoader(image_datasets['val'], batch_size=BATCH_SIZE, shuffle=False, num_workers=4, drop_last=False)
              }
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

def test_model(model_ft, criterion, save_model=False, save_name=None):
    data_iter = iter(dataloaders['val'])
    model_ft.eval()
    acc_val = 0
    step = 0
    for data in data_iter:
        img, lbl = data
        if use_gpu:
            img = img.cuda()
            lbl = lbl.cuda()
        img = Variable(img)
        lbl = Variable(lbl)

        out = model_ft(img)
        _, preds = torch.max(nn.functional.softmax(out, dim=1).data, 1)
        loss = criterion(out, lbl)
        # acc_val += (torch.sum(preds == lbl.data)/BATCH_SIZE)
        acc_val += torch.sum(preds == lbl.data)
        step = step+1
    acc = acc_val / dataset_sizes['val']
    # acc = acc_val / step
    print("validation accuracy: {:.4f}".format(acc))
    if save_model:
        torch.save(model_ft.state_dict(), save_name)
    return

def train_model(model, clscriterion, optimizer, lr_sch=None, num_epochs=15):
    since = time.time()

    print("start training")

    for epoch in range(num_epochs):
        srcdata = iter(dataloaders['train'])
                
        running_clsloss = 0.0
        running_corrects = 0
        step = 0

        for data in srcdata:
            srcinps, srclbls = data

            if use_gpu:
                srcinps = Variable(srcinps.cuda())
                srclbls = Variable(srclbls.cuda())
            else:
                srcinps, srclbls = Variable(srcinps), Variable(srclbls)

            optimizer.zero_grad()

            srcoutput = model(srcinps)

            _, preds = torch.max(nn.functional.softmax(srcoutput,dim=1).data, 1)
            clsloss = clscriterion(srcoutput, srclbls)         

            clsloss.backward()
            optimizer.step()
            step = step + 1

            running_clsloss += clsloss.data[0]
            running_corrects += (torch.sum(preds == srclbls.data)/BATCH_SIZE)

        if lr_sch is not None:
        	if lr_sch=='custom':
        		# p = (epoch+1)/max_epoch_lr
        		lr_decay = pow((1. + (epoch+1) * gamma), (-1 * power))
        		new_lr = lr_decay * LR
        		for param_group in optimizer.param_groups:
        			param_group['lr'] = new_lr
        	else:
        		lr_sch.step()

        epoch_clsloss = running_clsloss / step
        epoch_acc = running_corrects / step

        if epoch % 10 == 0:
            test_model(model_ft, clscriterion)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    return


class _Features(nn.Module):
    def __init__(self):
        super(_Features, self).__init__()
        alexnet = models.alexnet(pretrained=True)
        alexnet.classifier = nn.Sequential(*list(alexnet.classifier.children())[:-1])
        self.basenet = alexnet
        self.extra = nn.Linear(4096,256)

        nn.init.xavier_uniform(self.extra.weight)
        nn.init.constant(self.extra.bias, 0)


    def forward(self, x):
        x = self.basenet(x)
        x = self.extra(x)
        return x

class GRLModel(nn.Module):
    def __init__(self):
        super(GRLModel, self).__init__()
        self.features = _Features()
        self.relu = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(256, num_classes)
        
        nn.init.xavier_uniform(self.classifier.weight)
        nn.init.constant(self.classifier.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.relu(x)
        out = self.classifier(x.view(x.size(0), -1))
        return out
        
model_ft = GRLModel()

if use_gpu:
    model_ft = model_ft.cuda()

clscriterion = nn.CrossEntropyLoss()

params = None
ct = 0

for child in model_ft.features.basenet.features.children():
	if ct<=7:
		ct = ct+1
		for params in child.parameters():
			params.requires_grad=False

print(model_ft.features)
print(model_ft.classifier)

for name, params in model_ft.features.named_parameters():
	print(name, params.requires_grad)

weight_params = []
bias_params = []
for name, params in model_ft.features.basenet.features.named_parameters():
    if name=='8.weight' or name=='10.weight':
        weight_params += [params]
    elif name=='8.bias' or name=='10.bias':
        bias_params += [params]
for name, params in model_ft.features.basenet.classifier.named_parameters():
    if 'weight' in name:
        weight_params += [params]
    elif 'bias' in name:
        bias_params += [params]
print("weight params: ", len(weight_params))
print("bias params: ", len(bias_params))

params=[
{'params': weight_params, 'lr': LR, 'weight_decay': W_Decay},
{'params': bias_params, 'lr': 2*LR, 'weight_decay': 0.},
{'params': model_ft.features.extra.weight, 'lr': 10*LR, 'weight_decay': 10*W_Decay},
{'params': model_ft.features.extra.bias, 'lr': 20*LR, 'weight_decay': 0.},
{'params': model_ft.classifier.weight, 'lr': 10*LR, 'weight_decay': 10*W_Decay},
{'params': model_ft.classifier.bias, 'lr': 20*LR, 'weight_decay': 0.}
]

optimizer = optim.SGD(params, momentum=0.9)

# lr_sch = None
# lr_sch = lr_scheduler.ExponentialLR(optimizer, 0.1)
lr_sch = lr_scheduler.StepLR(optimizer, 20, 0.1)
# lr_sch = 'custom'

train_model(model_ft, clscriterion, optimizer, lr_sch, num_epochs=EPOCHS)

test_model(model_ft, clscriterion)
