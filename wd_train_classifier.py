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
from logit import flatten
import time
import os
import copy
import itertools
from tqdm import *

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
train_dir = domainData['amazon'] # 'amazon', 'dslr', 'webcam'
val_dir = domainData['webcam']
num_classes = NUM_CLASSES['office']

image_datasets = {'train' : datasets.ImageFolder(train_dir,
                                          data_transforms['train']),
                  'val' : datasets.ImageFolder(val_dir,
                                          data_transforms['val'])
                 }
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32,
                                             shuffle=True, num_workers=4)
              for x in ['train']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train']}
class_names = image_datasets['train'].classes

def train_model(model, clscriterion, optimizer, num_epochs=15):
    since = time.time()

    best_acc = 0.0
    all_acc = []

    for epoch in tqdm(range(num_epochs)):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # 0 - source 1 - target
        srcdata = iter(dataloaders['train'])
        
        model.train() # training mode
        
        running_clsloss = 0.0
        running_corrects = 0

        # Iterate over data.
        for data in srcdata:
            # get the inputs
            srcinps, srclbls = data

            # wrap them in Variable

            if use_gpu:
                srcinps = Variable(srcinps.cuda())
                srclbls = Variable(srclbls.cuda())
            else:
                srcinps, srclbls = Variable(srcinps), Variable(srclbls)

            # zero the parameter gradients
            optimizer.zero_grad()

            # source data
            srcoutput = model(srcinps)
            _, preds = torch.max(srcoutput.data, 1)
            clsloss = clscriterion(srcoutput, srclbls)
#             print("lblb: ", srcoutput[0])
#             print("srclbls: ", srclbls)
            

            clsloss.backward()
            optimizer.step()

            running_clsloss += clsloss.data[0] * srcinps.size(0)
            running_corrects += torch.sum(preds == srclbls.data)


        epoch_clsloss = running_clsloss / dataset_sizes['train']
        epoch_acc = running_corrects / dataset_sizes['train']
        all_acc += [epoch_acc]

        print('Classification Loss: {:.4f} Acc: {:.4f}'.format(
        epoch_clsloss, epoch_acc))

        if best_acc < epoch_acc:
            best_acc = epoch_acc
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    print('Mean Acc: {:4f}'.format(np.mean(all_acc)))

    return


class _Features(nn.Module):
    def __init__(self):
        super(_Features, self).__init__()
        self.basenet = None
        self.extra = None
        self.config = 3
        if self.config == 1:
            self.basenet = nn.Sequential(nn.Conv2d(3,16,4,stride=2), nn.ReLU(inplace=True), nn.Conv2d(16,24,3,2), nn.ReLU(inplace=True), nn.Conv2d(24,32,3,2), nn.ReLU(inplace=True), nn.Conv2d(32,64,5,2), nn.ReLU(inplace=True), nn.Conv2d(64,128,5,2), nn.ReLU(inplace=True), nn.Conv2d(128,512,4,2), nn.ReLU(inplace=True))
            # train: 0.57, val: 0.40
        elif self.config == 2:
            resnet = models.resnet18(pretrained=True)
            self.basenet = nn.Sequential(*list(resnet.children())[:-4])
            self.extra = nn.Sequential(nn.AvgPool2d(7,7), flatten(), nn.ReLU(inplace=True), nn.Linear(2048, 512), nn.ReLU(inplace=True))
            #train: 0.80, val: 0.67
        elif self.config == 3:
            self.basenet = nn.Sequential(flatten(), nn.Linear(3*224*224, 512), nn.ReLU(inplace=True))
            # train: 0.04
        elif self.config == 4:
            self.basenet = nn.Sequential(nn.Conv2d(3,16,4,2), nn.ReLU(inplace=True),
                            flatten(), nn.Linear(16*111*111, 512), nn.ReLU(inplace=True)
                )
            # train: 0.28
        elif self.config == 5:
            self.basenet = nn.Sequential(nn.Conv2d(3,16,4,2), nn.ReLU(inplace=True),
                            nn.AvgPool2d(7,7), flatten(), nn.Linear(16*15*15, 512), nn.ReLU(inplace=True)
                )


    def forward(self, x):
        x = self.basenet(x)
        if self.extra:
            x = self.extra(x)
        return x

class GRLModel(nn.Module):
    def __init__(self):
        super(GRLModel, self).__init__()
        activation_fn = nn.ReLU(inplace=True)
        self.features = _Features()
        self.classifier = nn.Sequential(
            nn.Linear(512, num_classes)
            )
        #weight initialization
        def weight_init(gen):
            for x in gen:
                if isinstance(x, nn.Conv2d) or isinstance(x, nn.ConvTranspose2d):
                    init.xavier_uniform(x.weight, gain=np.sqrt(2))
                    if x.bias is not None:
                        init.constant(x.bias, 0.1)
                elif isinstance(x, nn.Linear):
                    init.xavier_uniform(x.weight)
                    if x.bias is not None:
                        init.constant(x.bias, 0.0)
        weight_init(self.classifier.modules())

    def save_dict(self):
        cls_name = "classifier_dump.pth"
        ft_name = "features_dump.pth"
        torch.save(self.features.state_dict(), ft_name)
        torch.save(self.classifier.state_dict(), cls_name)

    def forward(self, x):
        if self.training:
            x = self.features(x)
            # x = x.view(x.size(0), -1)
            out = self.classifier(x.view(x.size(0), -1))
            return out
        else:
            x = self.features(x)
            out = self.classifier(x.view(x.size(0), -1))
            return out


model_ft = GRLModel()

if use_gpu:
    model_ft = model_ft.cuda()

clscriterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model_ft.parameters(), lr=1e-3, weight_decay=1e-4) # optimize all parameters
# optimizer = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

train_model(model_ft, clscriterion, optimizer, num_epochs=50)
# model_ft.save_dict()
