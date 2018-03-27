# Code taken from here : http://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

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
from logit import flatten
import time
import os
import copy
import itertools
from tqdm import *

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

def get_log_dir(path, log_dir):
    path = path + '/' + log_dir
    os.makedirs(path, exist_ok=True)
    return path

use_gpu = True and torch.cuda.is_available()
train_dir = domainData['amazon'] # 'amazon', 'dslr', 'webcam'
val_dir = domainData['webcam']
num_classes = NUM_CLASSES['office']
EPOCHS = 50
CONFIG = 11
log = False
exp_name = 'wd_jstCls_Rsnt2blk'
if log:
    log_dir = get_log_dir('./logs', exp_name)
    logger = Logger(log_dir)
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

def train_model(model, clscriterion, optimizer, lr_scheduler=None, num_epochs=15):
    since = time.time()

    print("start training")

    best_acc = 0.0
    all_acc = []

    for epoch in range(num_epochs):
        # print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        # print('-' * 10)

        # 0 - source 1 - target
        srcdata = iter(dataloaders['train'])
        
        model.train() # training mode

        if lr_scheduler:
            lr_scheduler.step()
        
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

            clsloss.backward()
            optimizer.step()

            running_clsloss += clsloss.data[0] * srcinps.size(0)
            running_corrects += torch.sum(preds == srclbls.data)


        epoch_clsloss = running_clsloss / dataset_sizes['train']
        epoch_acc = running_corrects / dataset_sizes['train']
        all_acc += [epoch_acc]

        # print('Classification Loss: {:.4f} Acc: {:.4f}'.format(
        # epoch_clsloss, epoch_acc))

        if log:
            logger.scalar_summary("loss", epoch_clsloss, epoch)
            logger.scalar_summary("accuracy", epoch_acc, epoch)

        if best_acc < epoch_acc:
            best_acc = epoch_acc

        if epoch % 10 == 0:
            test_model(model_ft, clscriterion)

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
        self.extra1 = None
        self.config = CONFIG
        if self.config == 1:
            self.basenet = nn.Sequential(nn.Conv2d(3,16,4,stride=2), nn.ReLU(inplace=True), nn.Conv2d(16,24,3,2), nn.ReLU(inplace=True), nn.Conv2d(24,32,3,2), nn.ReLU(inplace=True), nn.Conv2d(32,64,5,2), nn.ReLU(inplace=True), nn.Conv2d(64,128,5,2), nn.ReLU(inplace=True), nn.Conv2d(128,512,4,2), nn.ReLU(inplace=True))
            # train: 0.57
        elif self.config == 2:
            resnet = models.resnet18(pretrained=True)
            self.basenet = nn.Sequential(*list(resnet.children())[:-4])
            self.extra = nn.Sequential(nn.AvgPool2d(7,7), flatten(), nn.ReLU(inplace=True), nn.Linear(2048, 512), nn.ReLU(inplace=True))
            #train: 0.80, val: 0.20 (Amazon -> Webcam)
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
        elif self.config == 6:
            resnet = models.resnet18(pretrained=True)
            self.basenet = nn.Sequential(*list(resnet.children())[:-1])
            self.extra = nn.Sequential(flatten())

        elif self.config == 7:
            resnet = models.resnet18(pretrained=True)
            params = []
            params += list(resnet.conv1.parameters())
            params += list(resnet.bn1.parameters())
            params += list(resnet.layer1.parameters())
            params += list(resnet.layer2.parameters())
            # params += list(resnet.layer3.parameters())
            # params += list(resnet.layer4.parameters())
            for x in params:
                x.requires_grad=False
            resnet.fc = flatten()
            self.basenet = resnet
            # self.extra = nn.Sequential(flatten())
            # train: 0.92 val: 0.28 (Amazon -> Webcam, 50 epoch)

        elif self.config == 8:
            alexnet = models.alexnet(pretrained=True)
            for x in alexnet.features.parameters():
                x.requires_grad = False
            self.basenet = nn.Sequential(*list(alexnet.features.children()))
            self.extra = nn.Sequential(flatten(), nn.Linear(9216, 4096), nn.ReLU(inplace=True), nn.Linear(4096, 512), nn.ReLU(inplace=True))
            # train: 0.89 val: 0.36

        elif self.config == 9:
            alexnet = models.alexnet(pretrained=True)
            self.basenet = nn.Sequential(*list(alexnet.features.children()))
            self.extra = nn.Sequential(flatten(), nn.Linear(9216, 4096), nn.ReLU(inplace=True), nn.Linear(4096, 512), nn.ReLU(inplace=True))
            f_layers = [0,1,2,3,4,5,6]
            for x in f_layers:
                for y in self.basenet[x].parameters():
                    y.requires_grad=False
            print("train params: ")
            for x in self.basenet.parameters():
                print(x.requires_grad)

        elif self.config == 10:
            resnet = models.resnet34(pretrained=True)
            params = []
            params += list(resnet.conv1.parameters())
            params += list(resnet.bn1.parameters())
            params += list(resnet.layer1.parameters())
            params += list(resnet.layer2.parameters())
            params += list(resnet.layer3.parameters())
            # params += list(resnet.layer4.parameters())
            for x in params:
                x.requires_grad=False
            resnet.fc = flatten()
            self.basenet = resnet
            # self.extra = nn.Sequential(flatten())
            # train: 0.86 val: 0.68 (Amazon -> Webcam, 50 epoch)

        elif self.config == 11:
            resnet = models.resnet50(pretrained=True)
            params = []
            params += list(resnet.conv1.parameters())
            params += list(resnet.bn1.parameters())
            params += list(resnet.layer1.parameters())
            params += list(resnet.layer2.parameters())
            params += list(resnet.layer3.parameters())
            # params += list(resnet.layer4.parameters())
            for x in params:
                x.requires_grad=False
            resnet.fc = nn.Sequential(flatten(), nn.Linear(2048, 512))
            self.basenet = resnet
            self.extra = None
            # train: 0.86 val: 0.68 (Amazon -> Webcam, 50 epoch)


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
        if self.extra:
            weight_init(self.extra.modules())


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

params = None

if CONFIG == 9:
    params = [
    { 'params' : [x for x in model_ft.features.basenet[8].parameters() if x.requires_grad == True], 'lr' : 2e-6},
    { 'params' : [x for x in model_ft.features.basenet[10].parameters() if x.requires_grad == True], 'lr' : 2e-6},
    { 'params' : [x for x in model_ft.features.extra.parameters() if x.requires_grad == True], 'lr' : 1e-3, 'weight_decay' : 1e-4},
    { 'params' : [x for x in model_ft.classifier.parameters() if x.requires_grad == True], 'lr' : 1e-3, 'weight_decay' : 1e-4}
    ]
    # best with Adamax
elif CONFIG in [7, 10, 11]:
    params = [
    # {'params' : model_ft.features.basenet.layer3.parameters(), 'lr' : 2e-4 },
    {'params' : model_ft.features.basenet.layer4.parameters(), 'lr' : 2e-4 },
    {'params' : model_ft.classifier.parameters(), 'lr' : 1e-3, 'weight_decay' : 1e-4}
    ]
    # best with Adam -> train: 0.837416 val: 0.6214
else:
    params=[
    {'params' : model_ft.parameters(), 'lr' : 1e-3, 'weight_decay' : 1e-4}

    ]
optimizer = optim.Adam(params) # optimize all parameters
# optimizer = optim.SGD([ x for x in model_ft.parameters() if x.requires_grad == True], lr=0.001, momentum=0.9)

lr_scheduler = lr_scheduler.ExponentialLR(optimizer, 0.1)

train_model(model_ft, clscriterion, optimizer, lr_scheduler, num_epochs=EPOCHS)
# model_ft.save_dict()

test_model(model_ft, clscriterion)
