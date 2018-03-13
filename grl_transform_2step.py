# Code taken from here : http://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

from config import domainData
from config import num_classes as NUM_CLASSES
from advDomain import ADVDomain
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

def get_log_dir(path, log_dir):
    path = path + '/' + log_dir
    os.makedirs(path, exist_ok=True)
    return path

use_gpu = True and torch.cuda.is_available()
train_dir = domainData['amazon'] # 'amazon', 'dslr', 'webcam'
val_dir = domainData['webcam']
num_classes = NUM_CLASSES['office']
gp_lambda = 0.1
batch_size = 32
load_cls = True
log = False
exp_name = 'cls_ft_train_2step_adv_wt_schedule_150_epoch'
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
                                             shuffle=True, num_workers=0, drop_last=True)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

model_ft = ADVDomain(num_classes)

if load_cls:
    model_ft.load_cls()

if use_gpu:
    model_ft = model_ft.cuda()

clscriterion = nn.CrossEntropyLoss()

param_group = [
{'params' : model_ft._discriminator.parameters(), 'lr' : 1e-4, 'betas' : (0.5, 0.9)}
]
disc_opt = optim.Adam(param_group)

param_group = [
{'params' : model_ft.features.parameters(), 'lr' : 1e-4, 'betas' : (0.5, 0.9)}
]
gen_opt = optim.Adam(param_group)

param_group = [
{'params' : model_ft.features.parameters(), 'lr' : 1e-4, 'betas' : (0.5, 0.9)},
{'params' : model_ft.classifier.parameters(), 'lr' : 1e-4, 'betas' : (0.5, 0.9)}
]
cls_opt = optim.Adam(param_group)
# opt = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
# tar_lr_scheduler = lr_scheduler.StepLR(taroptimizer, step_size=7, gamma=0.1)


def train_model(model, clscriterion, disc_opt, gen_opt, cls_opt, lr_scheduler=None, num_epochs=25):
    since = time.time()

    best_acc = 0.0
    all_acc = []

    # gen_params = list(model.features.parameters())
    # cls_params = list(model.classifier.parameters())
    # dis_params = list(model._discriminator.parameters())


    for epoch in tqdm(range(num_epochs)):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # 0 - source 1 - target
        srcdata = iter(dataloaders['train'])
        tardata = iter(dataloaders['val'])
        tardata = itertools.cycle(tardata) # generating target data without hassle

        model.train() # training mode
        if lr_scheduler:
            lr_scheduler.step()

        running_clsloss = 0.0
        running_gloss = 0.0
        running_dloss = 0.0
        running_corrects = 0

        P = epoch / num_epochs
        lmbd = 2. / (1. + np.exp(-10. * P)) - 1

        # Iterate over data.
        for data in srcdata:
            # get the inputs
            srcinps, srclbls = data
            tarinps, _ = next(tardata)

            # wrap them in Variable

            if use_gpu:
                srcinps = Variable(srcinps.cuda())
                srclbls = Variable(srclbls.cuda())
                tarinps = Variable(tarinps.cuda())
            else:
                srcinps, srclbls = Variable(srcinps), Variable(srclbls)
                tarinps = Variable(tarinps)

            model.cls_train = False
            x_inter, d_inter, D_, D = model(srcinps, tarinps)

            d_l = (torch.sigmoid(D_)**2) + ((1. - torch.sigmoid(D))**2)
            
            g_l = D - D_

            # max_D = D.clone()
            # max_D[max_D < 0] = 0
            # D[D>0] = 0

            # max_D_ = D_.clone()
            # max_D_[max_D_ < 0] = 0
            # D_[D_>0] = 0

            # g_l = max_D ** 2 - D ** 2 + D_ ** 2 - max_D_ ** 2

            assert d_inter.dim() == 2, "d_inter dim: %s" % str(d_inter.dim())
            ones = torch.ones(d_inter.size())
            if use_gpu:
                ones = ones.cuda()
            x_inter_grads = autograd.grad(d_inter, x_inter, grad_outputs=ones, retain_graph=True)[0]

            x_inter_grads.volatile = False
            x_inter_grads.requires_grad = True
            x_inter_grads = x_inter_grads ** 2
            slopes = torch.sqrt((x_inter_grads.view(x_inter_grads.size(0), -1)).sum(-1))
            gp_loss = ((slopes - 1.) ** 2).mean() * gp_lambda

            # TODO: gp_loss should remain part of graph?
            d_loss = d_l + gp_loss

            model.zero_grad()

            d_loss = d_loss.mean() * lmbd
            running_dloss += d_loss.data[0] * srcinps.size(0)
            d_loss.backward(retain_graph=True)
            disc_opt.step()

            model.zero_grad()

            g_l = g_l.mean() * lmbd
            running_gloss += g_l.data[0] * srcinps.size(0)
            g_l.backward()
            gen_opt.step()

            model.cls_train = True
            cls_out = model(srcinps, None)

            _, preds = torch.max(cls_out.data, 1)
            clsloss = clscriterion(cls_out, srclbls)
#             print("lblb: ", srcoutput[0])
#             print("srclbls: ", srclbls)

            model.zero_grad() # zero all grads
            clsloss.backward()
            cls_opt.step()

            # statistics
            running_clsloss += clsloss.data[0] * srcinps.size(0)
            # running_dmnloss += dmnloss.data[0] * srcinps.size(0)
            # running_dmnloss += dmnloss2.data[0] * tarinps.size(0)
            running_corrects += torch.sum(preds == srclbls.data)


        epoch_clsloss = running_clsloss / dataset_sizes['train']
        epoch_gloss = running_gloss / dataset_sizes['train']
        epoch_dloss = running_dloss / dataset_sizes['train']
        epoch_acc = running_corrects / dataset_sizes['train']

        print('Classification Loss: {:.4f} Acc: {:.4f}'.format(
        epoch_clsloss, epoch_acc))
        print('Generator Loss: {:.4f}'.format(
        epoch_gloss))
        if log:
            logger.scalar_summary("loss", epoch_clsloss, epoch)
            logger.scalar_summary("accuracy", epoch_acc, epoch)
            logger.scalar_summary("gloss", epoch_gloss, epoch)
            logger.scalar_summary("dloss", epoch_dloss, epoch)

        all_acc += [epoch_acc]

        if best_acc < epoch_acc:
            best_acc = epoch_acc
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    print('Mean Acc: {:4f}'.format(np.mean(all_acc)))

    return

train_model(model_ft, clscriterion, disc_opt, gen_opt, cls_opt, num_epochs=1)

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

save_name = "grl_model_with_transform.pth"
test_model(model_ft, clscriterion, False, save_name)