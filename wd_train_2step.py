# Code taken from here : http://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

from config import domainData
from config import num_classes as NUM_CLASSES
from wdDomain import WDDomain
from logger import Logger
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.init as init
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.autograd import Function
import torch.nn.functional as F
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
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
gp_lambda = 10
batch_size = 64
EPOCHS = 450
load_cls = False
log = False
exp_name = 'wd_tr_2step_Rsnt2blk_r3'
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
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                             shuffle=True, num_workers=4, drop_last=True)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

model_ft = WDDomain(num_classes)

if load_cls:
    model_ft.load_cls()

if use_gpu:
    model_ft = model_ft.cuda()

clscriterion = nn.CrossEntropyLoss()

param_group1 = [
# {'params' : model_ft.features.parameters(), 'lr' : 1e-4, 'betas' : (0.5, 0.9)},
{'params' : model_ft._discriminator.parameters(), 'lr' : 1e-4, 'betas' : (0.5, 0.9), 'weight_decay' : 1e-5}
]
disc_opt = optim.Adam(param_group1)

param_group2 = [
{'params' : model_ft.features.basenet.layer4.parameters(), 'lr' : 1e-4, 'betas' : (0.5, 0.9), 'weight_decay' : 1e-5},
{'params' : model_ft.features.basenet.fc.parameters(), 'lr' : 1e-4, 'betas' : (0.5, 0.9), 'weight_decay' : 1e-5}
]
gen_opt = optim.Adam(param_group2)

param_group3 = [
{'params' : model_ft.features.basenet.layer4.parameters(), 'lr' : 2e-4, 'betas' : (0.5, 0.9), 'weight_decay' : 1e-4},
{'params' : model_ft.features.basenet.fc.parameters(), 'lr' : 1e-4, 'betas' : (0.5, 0.9), 'weight_decay' : 1e-5},
{'params' : model_ft.classifier.parameters(), 'lr' : 1e-4, 'weight_decay' : 1e-4}
]

cls_opt = optim.Adam(param_group3)

# Decay LR by a factor of 0.1 every 7 epochs
# tar_lr_scheduler = lr_scheduler.StepLR(taroptimizer, step_size=7, gamma=0.1)

cls_scheduler = lr_scheduler.ExponentialLR(cls_opt, 0.1)
gen_scheduler = lr_scheduler.ExponentialLR(gen_opt, 0.1)
disc_scheduler = lr_scheduler.ExponentialLR(disc_opt, 0.1)

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

        out = model_ft(img, None)
        _, preds = torch.max(out.data, 1)
        loss = criterion(out, lbl)
        acc_val += torch.sum(preds == lbl.data)
    acc = acc_val / dataset_sizes['val']
    print("validation accuracy: {:.4f}".format(acc))
    if save_model:
        torch.save(model_ft.state_dict(), save_name)
    return


def train_model(model, clscriterion, disc_opt, gen_opt, cls_opt, lr_schedulers=None, num_epochs=25):
    since = time.time()

    best_acc = 0.0
    all_acc = []

    for epoch in range(num_epochs):
        
        # 0 - source 1 - target
        srcdata = iter(dataloaders['train'])
        tardata = iter(dataloaders['val'])
        tardata = itertools.cycle(tardata) # generating target data without hassle

        if lr_schedulers:
            for x in lr_schedulers:
                x.step()

        running_clsloss = 0.0
        running_gloss = 0.0
        running_gploss = 0.0
        running_dloss = 0.0
        running_corrects = 0
        running_dgrads = 0.0

        P = epoch / num_epochs
        P = torch.Tensor([P])
        lmbda = 2. / (1. + torch.exp(-10. * P)) - 1
        if use_gpu:
            lmbda = lmbda.cuda() 
        lmbda = Variable(lmbda, requires_grad=False)
        # lmbda = 1.

        model.train() # training mode

        # Iterate over data.
        for data in srcdata:

            # get the inputs
            srcinps, srclbls = data
            # print("min val: ", torch.min(srcinps))
            # print("max val: ", torch.max(srcinps))
            tarinps, _ = next(tardata)

            if use_gpu:
                srcinps = Variable(srcinps.cuda(), requires_grad=False)
                srclbls = Variable(srclbls.cuda(), requires_grad=False)
                tarinps = Variable(tarinps.cuda(), requires_grad=False)
            else:
                srcinps, srclbls = Variable(srcinps, requires_grad=False), Variable(srclbls, requires_grad=False)
                tarinps = Variable(tarinps, requires_grad=False)

            model.cls_train = False
            
            X, G, D_, D = model(srcinps, tarinps)
            
            eps = torch.Tensor(X.size(0),1).uniform_(0.1,0.99)
            if X.is_cuda:
                eps = eps.cuda()
            eps = Variable(eps)
            X_inter = X + eps * (G - X)
            D_inter = model._discriminator(X_inter)
            
            ones = torch.ones(D_inter.size())
            if use_gpu:
                ones = ones.cuda()
            X_inter_grads = autograd.grad(D_inter, X_inter, grad_outputs=ones,
                                            retain_graph=True)[0]
            X_inter_grads.volatile = False
            X_inter_grads.requires_grad = True
            X_inter_grads = X_inter_grads ** 2
            slopes = torch.sqrt((X_inter_grads.view(X_inter_grads.size(0), -1)).sum(-1))
            gp_loss = ((slopes - 1.) ** 2).mean()

            d_l = (torch.sigmoid(D_)**2) + ((1. - torch.sigmoid(D))**2)
            d_l = d_l.mean()

            # TODO: use D_inter in d_l
            # TODO: d_l = d_l * -1 ?

            d_loss = (d_l + gp_loss * gp_lambda)
            running_dloss += d_l.data[0] * srcinps.size(0)
            running_gploss += gp_loss.data[0] * srcinps.size(0)

            model.zero_grad()

            d_loss.backward(retain_graph=True)
            disc_opt.step()

            model.zero_grad()
            # loss = nn.L1Loss()
            # g_l = loss(D_, D.detach())
            g_l = (D - D_).mean()
            running_gloss += g_l.data[0] * srcinps.size(0)
            g_l.backward()
            gen_opt.step()

            model.cls_train = True
            _, _, cls_out = model(srcinps, tarinps)
            
            model.zero_grad()

            _, preds = torch.max(cls_out.data, 1)
            clsloss = clscriterion(cls_out, srclbls)
            total_loss = clsloss
            total_loss.backward()
            cls_opt.step()
            
            # statistics
            running_clsloss += clsloss.data[0] * srcinps.size(0)
            running_corrects += torch.sum(preds == srclbls.data)

        epoch_clsloss = running_clsloss / dataset_sizes['train']
        epoch_gloss = running_gloss / dataset_sizes['train']
        # epoch_dmnloss = running_dmnloss / (dataset_sizes['train'] + dataset_sizes['val'])
        epoch_dloss = running_dloss / (dataset_sizes['train'])
        epoch_gploss = running_gploss / (dataset_sizes['train'])
        epoch_acc = running_corrects / dataset_sizes['train']
        # epoch_dgrads = running_dgrads / dataset_sizes['train']

        print('Classification Loss: {:.4f} Generator Loss: {:.4f} Critic Loss: {:.4f} GP Loss: {:.4f} Acc: {:.4f} '.format(
        epoch_clsloss, epoch_gloss, epoch_dloss, epoch_gploss, epoch_acc))

        if log:
            logger.scalar_summary("loss", epoch_clsloss, epoch)
            logger.scalar_summary("accuracy", epoch_acc, epoch)
            logger.scalar_summary("gloss", epoch_gloss, epoch)
            logger.scalar_summary("dloss", epoch_dloss, epoch)
            logger.scalar_summary("gploss", epoch_gploss, epoch)

        # all_acc += [epoch_acc]

        if epoch % 5 == 0:
            test_model(model, clscriterion, False, None)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    # print('Mean Acc: {:4f}'.format(np.mean(all_acc)))

    return

lr_schedulers = [cls_scheduler, gen_scheduler, disc_scheduler]
train_model(model_ft, clscriterion, disc_opt, gen_opt, cls_opt, lr_schedulers, num_epochs=EPOCHS)

save_name = "grl_model_with_transform.pth"
test_model(model_ft, clscriterion, False, save_name)
