# Code taken from here : http://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"

from config import domainData
from config import num_classes as NUM_CLASSES
from wdStackDomain1 import WDDomain
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
import logit
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
l2_param = 1e-5
g_loss_param = 0.1
batch_size = 32 # batch_size for each of source and target samples
EPOCHS = 50
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

print("Dataset: ")
print(type(image_datasets['train'][0][0]))
print(image_datasets['train'][2][1])

if load_cls:
    model_ft.load_cls()

if use_gpu:
    model_ft = model_ft.cuda()

# clscriterion = nn.CrossEntropyLoss()
clscriterion = logit.softmax_cross_entropy_with_logits(num_classes)

param_group1 = [
# {'params' : model_ft.features.parameters(), 'lr' : 1e-4, 'betas' : (0.5, 0.9)},
{'params' : model_ft._discriminator.parameters(), 'lr' : 1e-4} # , 'betas' : (0.5, 0.9)
] # , 'weight_decay' : 1e-5
disc_opt = optim.Adam(param_group1)

param_group2 = [
# {'params' : model_ft.features.basenet.layer4.parameters(), 'lr' : 1e-4, 'betas' : (0.5, 0.9), 'weight_decay' : 1e-5},
{'params' : model_ft.features.basenet.fc.parameters(), 'lr' : 1e-4} # , 'betas' : (0.5, 0.9
] # , 'weight_decay' : 1e-5
gen_opt = optim.Adam(param_group2)

param_group3 = [
# {'params' : model_ft.features.basenet.layer4.parameters(), 'lr' : 2e-4, 'betas' : (0.5, 0.9), 'weight_decay' : 1e-4},
{'params' : model_ft.features.basenet.fc.parameters(), 'lr' : 1e-4}, # , 'betas' : (0.5, 0.9), 'weight_decay' : 1e-5
{'params' : model_ft.classifier.parameters(), 'lr' : 1e-4} # , 'weight_decay' : 1e-4
]

cls_opt = optim.Adam(param_group3)

# Decay LR by a factor of 0.1 every 7 epochs
# tar_lr_scheduler = lr_scheduler.StepLR(taroptimizer, step_size=7, gamma=0.1)

# cls_scheduler = lr_scheduler.ExponentialLR(cls_opt, 0.1)
# gen_scheduler = lr_scheduler.ExponentialLR(gen_opt, 0.1)
# disc_scheduler = lr_scheduler.ExponentialLR(disc_opt, 0.1)

# lr_schedulers = [cls_scheduler, gen_scheduler, disc_scheduler]
lr_schedulers = None

softmax = nn.Softmax(dim=1)
l2_reg = logit.L2_Loss()

def test_model(model_ft, criterion, save_model=False, save_name=None):
    data_iter = iter(dataloaders['val'])

    model_ft.features.eval()
    model_ft.classifier.eval()
    model_ft._discriminator.eval()
    
    acc_val = 0
    steps = 0.
    softmax = nn.Softmax(dim=1)
    for data in data_iter:
        img, lbl = data
        if use_gpu:
            img = img.cuda()
            lbl = lbl.cuda()
        img = Variable(img, volatile=True)
        lbl = Variable(lbl, requires_grad=False)

        feat_out = model_ft.features(img)
        out = model_ft.classifier(feat_out)

        loss = criterion(out, lbl)
        
        out1 = softmax(out)
        _, preds = torch.max(out1.data, 1)
        acc_val += torch.eq(preds, lbl.data).float().mean()
        steps = steps + 1
    # acc = acc_val / dataset_sizes['val']
    acc = acc_val / steps
    print("validation accuracy: {:.4f}".format(acc))
    if save_model:
        torch.save(model_ft.state_dict(), save_name)
    return


def train_model(model, clscriterion, disc_opt, gen_opt, cls_opt, lr_schedulers=None, num_epochs=25):
    since = time.time()

    for epoch in range(num_epochs):
        
        # 0 - source 1 - target
        srcdata = iter(dataloaders['train'])
        tardata = iter(dataloaders['val'])

        # https://docs.python.org/2/library/itertools.html#itertools.cycle
        # NOTE: This requires significant memory if target dataset is large
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
        steps = 0.

        P = epoch / num_epochs
        P = torch.Tensor([P])
        lmbda = 2. / (1. + torch.exp(-10. * P)) - 1
        if use_gpu:
            lmbda = lmbda.cuda() 
        lmbda = Variable(lmbda, requires_grad=False)
        # lmbda = 1.

        model.classifier.train()
        model._discriminator.train()
        model.features.train()

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

            inps = torch.cat([srcinps, tarinps])
            
            feat_out = model.features(inps)
            b_s = feat_out.size(0) // 2

            X = feat_out[:b_s]
            G = feat_out[b_s:]
            
            cls_out = model.classifier(X)

            eps = torch.Tensor(X.size()).uniform_(0,1)
            if X.is_cuda:
                eps = eps.cuda()
            # eps = Variable(eps, requires_grad=False)
            difference = X.data - G.data
            interpolates = X.data + (eps * difference)
            X_whole = torch.cat([feat_out.data, interpolates])
            X_whole = Variable(X_whole, requires_grad=True)

            disc_out = model._discriminator(X_whole)
            
            ones = torch.ones(disc_out.size())
            if use_gpu:
                ones = ones.cuda()
            grads = autograd.grad(disc_out, X_whole, grad_outputs=ones,
                                            retain_graph=True, create_graph=True)[0]
            
            slopes = torch.sqrt(torch.sum(grads ** 2, dim=1))
            gp_loss = ((slopes - 1.) ** 2).mean()

            D = disc_out[:b_s]
            D_ = disc_out[feat_out.size(0)//2:feat_out.size(0)]

            d_l = (torch.sigmoid(D_)**2) + ((1. - torch.sigmoid(D))**2)
            d_l = d_l.mean()

            # TODO: use D_inter in d_l
            # TODO: d_l = d_l * -1 ?

            d_loss = (d_l + gp_loss * gp_lambda) * g_loss_param
            # running_dloss += d_l.data[0] * srcinps.size(0)
            running_dloss += d_l.data[0]
            # running_gploss += gp_loss.data[0] * srcinps.size(0)
            running_gploss += gp_loss.data[0]

            disc_opt.zero_grad()
            d_loss.backward(retain_graph=True)
            disc_opt.step()

            cls_opt.zero_grad()
            # loss = nn.L1Loss()
            # g_l = loss(D_, D.detach())
            # g_l = (D - D_).mean()
            # running_gloss += g_l.data[0] * srcinps.size(0)
            # gen_opt.step()
            
            g_l = (D - D_).mean()
            running_gloss += g_l.data[0]
            # g_l.backward(retain_graph=True)
            
            # model.zero_grad()

            clsloss = clscriterion(cls_out, srclbls)
            cls_out = softmax(cls_out)
            _, preds = torch.max(cls_out.data, 1)
            running_clsloss += clsloss.data[0]
            running_corrects += torch.eq(preds, srclbls.data).float().mean()
            steps = steps + 1

            l2_loss = None
            for name, param in model.classifier.named_parameters():
                if 'weight' in name:
                    if l2_loss is None:
                        l2_loss = l2_reg(param) * l2_param
                    else:
                        l2_loss += l2_reg(param)
            for name, param in model.features.basenet.fc.named_parameters():
                if 'weight' in name:
                    l2_loss += l2_reg(param) * l2_param

            total_loss = clsloss + l2_loss + g_l * g_loss_param
            # total_loss = clsloss + l2_loss + (g_l + d_loss) * g_loss_param
            total_loss.backward()
            cls_opt.step()
            
            # statistics
            # running_clsloss += clsloss.data[0] * srcinps.size(0)
            # running_corrects+= torch.sum(preds == srclbls.data)

        epoch_clsloss = running_clsloss / steps
        epoch_gloss = running_gloss / steps
        epoch_dloss = running_dloss / steps
        epoch_gploss = running_gploss / steps
        epoch_acc = running_corrects / steps
        # epoch_clsloss = running_clsloss / dataset_sizes['train']
        # epoch_gloss = running_gloss / dataset_sizes['train']
        # epoch_dloss = running_dloss / (dataset_sizes['train'])
        # epoch_gploss = running_gploss / (dataset_sizes['train'])
        # epoch_acc = running_corrects / dataset_sizes['train']

        print('Classification Loss: {:.4f} Generator Loss: {:.4f} Critic Loss: {:.4f} GP Loss: {:.4f} Acc: {:.4f} '.format(
        epoch_clsloss, epoch_gloss, epoch_dloss, epoch_gploss, epoch_acc))

        if log:
            logger.scalar_summary("loss", epoch_clsloss, epoch)
            logger.scalar_summary("accuracy", epoch_acc, epoch)
            logger.scalar_summary("gloss", epoch_gloss, epoch)
            logger.scalar_summary("dloss", epoch_dloss, epoch)
            logger.scalar_summary("gploss", epoch_gploss, epoch)

        if epoch % 5 == 0:
            test_model(model, clscriterion, False, None)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    
    return

train_model(model_ft, clscriterion, disc_opt, gen_opt, cls_opt, lr_schedulers, num_epochs=EPOCHS)

save_name = "grl_model_with_transform.pth"
test_model(model_ft, clscriterion, False, save_name)
