# Code taken from here : http://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

from config import domainData
from config import num_classes as NUM_CLASSES
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.autograd import Function
import torch.nn.functional as F
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import itertools

class GRL(Function):
    @staticmethod
    def forward(ctx, x):
        x = 1 * x
        return x
    @staticmethod
    def backward(ctx, grad_output):
        grad_output = -1 * grad_output
        return grad_output
grl = GRL.apply # create alias

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

use_gpu = True and torch.cuda.is_available()
train_dir = domainData['webcam'] # 'amazon', 'dslr', 'webcam'
val_dir = domainData['dslr']
num_classes = NUM_CLASSES['office']
print("use gpu: ", use_gpu)

torch.manual_seed(7)
if use_gpu:
    torch.cuda.manual_seed(7)

image_datasets = {'train' : datasets.ImageFolder(train_dir,
                                          data_transforms['train']),
                  'val' : datasets.ImageFolder(val_dir,
                                          data_transforms['val'])
                 }
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

def train_model(model, clscriterion, dmncriterion, srcoptimizer, taroptimizer,
                srcscheduler, tarscheduler, num_epochs=25):
    since = time.time()

    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # 0 - source 1 - target
        srcdata = iter(dataloaders['train'])
        tardata = iter(dataloaders['val'])
        tardata = itertools.cycle(tardata) # generating target data without hassle

        model.train() # training mode
        srcscheduler.step()
        tarscheduler.step()

        running_clsloss = 0.0
        running_dmnloss = 0.0
        running_corrects = 0

        # Iterate over data.
        for data in srcdata:
            # get the inputs
            srcinps, srclbls = data
            tarinps, tarlbls = next(tardata)

            # wrap them in Variable

            if use_gpu:
                srcinps = Variable(srcinps.cuda())
                srclbls = Variable(srclbls.cuda())
                tarinps = Variable(tarinps.cuda())
                tarlbls = Variable(tarlbls.cuda())
            else:
                srcinps, srclbls = Variable(srcinps), Variable(srclbls)
                tarinps, tarlbls = Variable(tarinps), Variable(tarlbls)

            # zero the parameter gradients
            srcoptimizer.zero_grad()

            # source data
            model.is_source = True
            srcoutput = model(srcinps)
            dmnlbls = torch.LongTensor(srcoutput[0].size(0)).zero_()
            if use_gpu:
                dmnlbls = dmnlbls.cuda()
            dmnlbls = Variable(dmnlbls, requires_grad=False)
            _, preds = torch.max(srcoutput[1].data, 1)
            clsloss = clscriterion(srcoutput[1], srclbls)
#             print("lblb: ", srcoutput[0])
#             print("srclbls: ", srclbls)
            dmnloss = dmncriterion(srcoutput[0], dmnlbls)
            loss = clsloss + dmnloss

            loss.backward()
            srcoptimizer.step()

            taroptimizer.zero_grad()

            # target data
            model.is_source = False
            taroutput = model(tarinps)
            dmnlbls = torch.ones(taroutput.size(0)).long()
            if use_gpu:
                dmnlbls = dmnlbls.cuda()
            dmnlbls = Variable(dmnlbls, requires_grad=False)
            dmnloss2 = dmncriterion(taroutput, dmnlbls)
            dmnloss2.backward()
            taroptimizer.step()

            # statistics
            running_clsloss += clsloss.data[0] * srcinps.size(0)
            running_dmnloss += dmnloss.data[0] * srcinps.size(0)
            running_dmnloss += dmnloss2.data[0] * tarinps.size(0)
            running_corrects += torch.sum(preds == srclbls.data)


        epoch_clsloss = running_clsloss / dataset_sizes['train']
        epoch_dmnloss = running_dmnloss / (dataset_sizes['train'] + dataset_sizes['val'])
        epoch_acc = running_corrects / dataset_sizes['train']

        print('Classification Loss: {:.4f} Domain Loss: {:.4f} Acc: {:.4f}'.format(
        epoch_clsloss, epoch_dmnloss, epoch_acc))

        if best_acc < epoch_acc:
            best_acc = epoch_acc
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    return

class GRLModel(nn.Module):
    def __init__(self):
        super(GRLModel, self).__init__()
        resnet18 = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(resnet18.children())[:-1]) # get the feature extractor
        self.tDmn = nn.Sequential(nn.Linear(512,64), nn.ReLU(inplace=True),
                                       nn.Linear(64,512), nn.ReLU(inplace=True))
        self.tCls = nn.Sequential(nn.Linear(512,64), nn.ReLU(inplace=True),
                                       nn.Linear(64,512), nn.ReLU(inplace=True))
        self.is_source = True
        self.grl = nn.Sequential(
            nn.Linear(512,2), nn.ReLU(inplace=True)
        )
        self.classifier = nn.Linear(512,num_classes)
        def weight_init(gen):
            for x in gen:
                if isinstance(x, nn.Conv2d) or isinstance(x, nn.ConvTranspose2d):
                    init.xavier_uniform(x.weight, gain=np.sqrt(2))
                    init.constant(x.bias, 0.1)
                elif isinstance(x, nn.Linear):
                    init.xavier_uniform(x.weight)
                    init.constant(x.bias, 0.0)

        weight_init(self.tDmn.modules())
        weight_init(self.tCls.modules())
        weight_init(self.grl.modules())
        weight_init(self.classifier.modules())

    def forward(self, x):
        if self.training:
            x = self.features(x)
            x = x.view(x.size(0), -1)
            if self.is_source:
                x = self.tDmn(x)
                x_ = x.clone()
                x_ = self.tCls(x_)
                x = grl(x) # position 3
                out1 = self.grl(x)
                out2 = self.classifier(x_)
                return out1, out2
            else:
                # x_ = x_.view(x_.size(0), -1)
                x = grl(x) # position 3
                out = self.grl(x)
                return out
        else:
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.tCls(x)
            out = self.classifier(x)
            return out



model_ft = GRLModel()

if use_gpu:
    model_ft = model_ft.cuda()

clscriterion = nn.CrossEntropyLoss()
dmncriterion = nn.CrossEntropyLoss()

# src_params = []
# src_params += list(model_ft.features.parameters())
# src_params += list(model_ft.classifier.parameters())
# src_params += list(model_ft.grl.parameters())
srcoptimizer = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9) # optimize all parameters
param_group = []
param_group += list(model_ft.features.parameters())
# param_group += list(model_ft.transform.parameters())
param_group += list(model_ft.grl.parameters())
taroptimizer = optim.SGD(param_group, lr=0.01, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
src_lr_scheduler = lr_scheduler.StepLR(srcoptimizer, step_size=7, gamma=0.1)
tar_lr_scheduler = lr_scheduler.StepLR(taroptimizer, step_size=7, gamma=0.1)

train_model(model_ft, clscriterion, dmncriterion, srcoptimizer, taroptimizer, src_lr_scheduler, tar_lr_scheduler,
                       num_epochs=15)

def test_model(model_ft, criterion, save_model=False, save_name=None):
    data_iter = iter(dataloaders['val'])
    model_ft.eval()
    acc_val = 0
    for data in data_iter:
        img, lbl = data
        if use_gpu:
            img = img.cuda()
            lbl = lbl.cuda()
        img = Variable(img)
        lbl = Variable(lbl)

        out = model_ft(img)
        _, preds = torch.max(out.data, 1)
        loss = criterion(out, lbl)
        acc_val += torch.sum(preds == lbl.data)
    acc = acc_val / dataset_sizes['val']
    print("validation accuracy: {:.4f}".format(acc))
    if save_model:
        torch.save(model_ft.state_dict(), save_name)
    return

save_name = "grl_model_with_transform.pth"
test_model(model_ft, clscriterion, False, save_name)
