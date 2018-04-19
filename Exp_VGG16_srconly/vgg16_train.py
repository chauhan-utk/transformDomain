
# coding: utf-8

import sys
sys.path.append('../')
from config import domainData
from config import num_classes as NUM_CLASSES
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
import torchvision
import utils
from torchvision import datasets, transforms
import numpy as np
import itertools
import matplotlib
matplotlib.pyplot.switch_backend('agg') # for running the script on gpu servers
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import os
import argparse

def str2bool(v):
	return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(
description='Experiment Script for Domain Adaptation Source Only')
parser.add_argument('--src', default='amazon', choices=['amazon', 'webcam', 'dslr'],
	type=str, help='source domain: amazon or webcam or dslr')
parser.add_argument('--tar', default='webcam', choices=['amazon', 'webcam', 'dslr'],
	type=str, help='target domain: amazon or webcam or dslr')
parser.add_argument('--manual_seed', default=1, type=int,
	help='For deterministic results')
parser.add_argument('--batch_size', default=32, type=int,
	help='batch size')
parser.add_argument('--use_gpu', default=True, type=str2bool,
	help='use gpu')
parser.add_argument('--gpu', default=0, type=int,
    help='specify which gpu to use')
parser.add_argument('--epochs', default=120, type=int,
	help='number of epochs for source data')
parser.add_argument('--momentum', default=0.9, type=float,
	help='Momentum value for optim that require momentum')
parser.add_argument('--lr', default=2e-4, type=float,
	help='Learning rate for all layers. Individual rates can be changed using learning rate multipliers.')
parser.add_argument('--lr_sch', default='No', choices=['No', 'exponential', 'step'],
	type=str, help='Learning rate scheduler')
parser.add_argument('--lr_sch_gamma', default=0.1, type=float, help='Learning rate scheduler multiplicative factor')
parser.add_argument('--p_lr_decay', default=-1, type=int, help='Different epoch size for learning rate decay calculation')
parser.add_argument('--net_wtDcy', default=1e-5, type=float, help='Weight decay for basenet weights')
parser.add_argument('--net_biasDcy', default=1e-5, type=float, help='Weight decay for basenet bias')
parser.add_argument('--btl_wtDcy', default=1e-5, type=float, help='Weight decay for classifier weights')
parser.add_argument('--btl_biasDcy', default=1e-5, type=float, help='Weight decay for classifier bias')
parser.add_argument('--net_wtLR', default=1, type=int, help='LR multiplier for basenet weights')
parser.add_argument('--net_biasLR', default=2, type=int, help='LR multiplier for basenet bias')
parser.add_argument('--btl_wtLR', default=10, type=int, help='LR multiplier for classifier/bottleneck weights')
parser.add_argument('--btl_biasLR', default=20, type=int, help='LR multiplier for classifier/bottleneck bias')
parser.add_argument('--alexnet_path', default='../alexnet.pth', type=str, help='path to alexnet pth')
parser.add_argument('--exp_name', default=None, type=str, help='optional experiment name')
parser.add_argument('--exp_name_suffix', default=None, type=str, help='optional experiment name suffix')

args = parser.parse_args()

src = domainData[args.src]
tar = domainData[args.tar]

src_transforms = transforms.Compose([
    transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.RandomVerticalFlip(),
    transforms.RandomResizedCrop(224, scale=(0.50,1.0), ratio=(1.,1.)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])    
])
tar_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
#     transforms.RandomResizedCrop(224, scale=(0.25,1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]) 
])

src_dataset = datasets.ImageFolder(src, transform=src_transforms)
tar_dataset = datasets.ImageFolder(tar, transform=tar_transforms)
srcSampler = torch.utils.data.sampler.RandomSampler(src_dataset)
tarSampler = torch.utils.data.sampler.RandomSampler(tar_dataset)
srcDataLen = len(src_dataset)
tarDataLen = len(tar_dataset)
use_gpu = args.use_gpu and torch.cuda.is_available()
gpu = args.gpu

opt = {
    'src': args.src,
    'tar': args.tar,
    'manual_seed':args.manual_seed,
    'batchSize':args.batch_size,
    'use_gpu': use_gpu,
    'num_classes': 31,
    'epochs': args.epochs,
    'momentum': args.momentum,
    'lr': args.lr,
    'lr_sch': args.lr_sch,
    'lr_sch_gamma': args.lr_sch_gamma,
    'p_lr_decay': args.p_lr_decay,
    'n0': 1.,
    'alpha': 10,
    'beta': 0.75,
    'betas': (0.5,0.99),
    'net_wtDcy': args.net_wtDcy,
    'net_biasDcy': args.net_biasDcy,
    'net_wtLR': args.net_wtLR,
    'net_biasLR': args.net_biasLR,
    'btl_wtDcy': args.btl_wtDcy,
    'btl_biasDcy': args.btl_biasDcy,
    'btl_wtLR': args.btl_wtLR,
    'btl_biasLR': args.btl_biasLR,
    'srcDataLen': srcDataLen,
    'tarDataLen': tarDataLen
}

if opt['p_lr_decay'] == -1: opt['p_lr_decay'] = opt['epochs']

torch.manual_seed(opt['manual_seed'])
if opt['use_gpu']: torch.cuda.manual_seed(opt['manual_seed'])

net = models.vgg16(pretrained=True)
net.classifier = nn.Sequential(*list(net.classifier.children())[:-1])
# print(net)

print("use_gpu: ", opt['use_gpu'])

src_dataloader = torch.utils.data.DataLoader(src_dataset, batch_size=opt['batchSize'], 
                                             shuffle=(srcSampler is None), sampler=srcSampler,
                                            num_workers=2, pin_memory=False, drop_last=False)
tar_dataloader = torch.utils.data.DataLoader(tar_dataset, batch_size=opt['batchSize'],
                                            shuffle=(tarSampler is None), sampler=tarSampler,
                                            num_workers=2, pin_memory=False, drop_last=False)


def init_weights(m):
#     print(m)
    if isinstance(m, nn.Linear):
        init.xavier_normal(m.weight)
        init.constant(m.bias, 0.1)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.classifier = nn.Sequential(nn.Linear(4096,256),
                                       nn.ReLU(inplace=True), nn.Linear(256, opt['num_classes']))
        self.classifier.apply(init_weights)
    
    def forward(self, x):
        x = self.classifier(x)
        return x


net2 = Model()

if opt['use_gpu']:
    net = net.cuda(gpu)
    net2 = net2.cuda(gpu)
    torch.backends.cudnn.enabled=True
    torch.backends.cudnn.benchmark=True

criterion = nn.CrossEntropyLoss()
softmax = nn.Softmax(dim=1)

netfeatwt, netfeatbias = list(), list()
for name, param in net.features.named_parameters():
    if name in ['19.weight', '21.weight', '24.weight', '26.weight', '28.weight']: netfeatwt.append(param)
    elif name in ['19.bias', '21.bias', '24.bias', '26.bias', '28.bias']: netfeatbias.append(param)
    else: param.requires_grad=False

for name, param in net.classifier.named_parameters():
    if 'weight' in name: netfeatwt.append(param)
    elif 'bias' in name: netfeatbias.append(param)

for name, param in net.named_parameters():
    print(name, param.requires_grad)

net2_weight, net2_bias = list(), list()
for name, param in net2.named_parameters():
    if 'weight' in name: net2_weight.append(param)
    elif 'bias' in name: net2_bias.append(param)

sgd_params = [
    {'params': netfeatwt, 'lr': opt['lr'], 'momentum': opt['momentum'], 'weight_decay': opt['net_wtDcy'], 'lr_mul': opt['net_wtLR']},
    {'params': netfeatbias, 'lr': opt['lr'], 'momentum': opt['momentum'], 'weight_decay': opt['net_biasDcy'], 'lr_mul': opt['net_biasLR']},
    {'params': net2_weight, 'lr': opt['lr'], 'momentum': opt['momentum'], 'weight_decay': opt['btl_wtDcy'], 'lr_mul': opt['btl_wtLR']},
    {'params': net2_bias, 'lr': opt['lr'], 'momentum': opt['momentum'], 'weight_decay': opt['btl_biasDcy'], 'lr_mul': opt['btl_biasLR']}
]

optimizer_2 = optim.SGD(sgd_params)

lr_sch = None
if opt['lr_sch']=='step': lr_sch = optim.lr_scheduler.StepLR(optimizer_2, 20, opt['lr_sch_gamma'])
if opt['lr_sch']=='exponential': lr_sch = optim.lr_scheduler.ExponentialLR(optimizer_2, opt['lr_sch_gamma'])

def get_validation_acc(cm=False):
    tarData = iter(tar_dataloader)
    totalCorrects = 0.
    if cm: y_preds, y_true = list(), list()
    for tarimgs, tarlbls in tarData:
        tarimgs = tarimgs.cuda(gpu) if opt['use_gpu'] else tarimgs
        tarlbls = tarlbls.cuda(gpu) if opt['use_gpu'] else tarlbls      
        tarimgs = Variable(tarimgs, volatile=True)

        feat_ = net(tarimgs)
        logits = net2(feat_)

        _, preds = torch.max(softmax(logits).data, 1)
        totalCorrects += torch.eq(preds, tarlbls).float().sum()
        if cm: y_preds.append(preds.cpu().numpy()), y_true.append(tarlbls.cpu().numpy())
    valAcc = totalCorrects / opt['tarDataLen']
    if cm: return valAcc, y_preds, y_true
    return valAcc

srcClsLoss_plt, srcAcc_plt, valAcc_plt = list(), list(), list()

p = np.linspace(float(1./opt['p_lr_decay']),1,opt['p_lr_decay'])

for epoch in range(opt['epochs']):
    srcData = iter(src_dataloader)
    totalCorrects = 0.
    totalClsLoss = 0.
    
    n_p = opt['n0'] / pow((1. + opt['alpha'] * p[epoch]), (opt['beta']))
    print("n_p: ", n_p)
    for param_group in optimizer_2.param_groups:
        param_group['lr'] = opt['lr'] * param_group['lr_mul'] * n_p
        
    for srcimgs, srclbls in srcData:
        srcimgs = srcimgs.cuda(gpu) if opt['use_gpu'] else srcimgs
        srclbls = srclbls.cuda(gpu) if opt['use_gpu'] else srclbls
        srcimgs, srclbls = Variable(srcimgs), Variable(srclbls)
        
        feat_ = net(srcimgs)
        logits = net2(feat_)
        
        clsloss = criterion(logits, srclbls)
        totalClsLoss += clsloss.data[0] * opt['batchSize']
        
        _, preds = torch.max(softmax(logits).data, 1)
        totalCorrects += torch.eq(preds, srclbls.data).float().sum()
               
        optimizer_2.zero_grad()
        clsloss.backward()
        optimizer_2.step()
        
        if lr_sch: lr_sch.step()
        
    srcAcc = totalCorrects / opt['srcDataLen']
    srcLoss = totalClsLoss / opt['srcDataLen']
    valAcc = get_validation_acc()
    print("Epoch: {}, train loss: {:.4f}, train acc: {:.4f}, validation acc: {:.4f}".
          format(epoch+1, srcLoss, srcAcc, valAcc))
    srcClsLoss_plt.append(srcLoss), srcAcc_plt.append(srcAcc), valAcc_plt.append(valAcc)

if args.exp_name: exp_dir = args.exp_name
else: exp_dir = 'lr_'+str(opt['lr']).replace('.','_')+'_bs_'+str(opt['batchSize'])+'_epochs_'+str(opt['epochs'])+'_'
dir_cnt = 0
while(os.path.exists(exp_dir+str(dir_cnt))):
    dir_cnt = dir_cnt+1
exp_dir = exp_dir + str(dir_cnt)
if args.exp_name_suffix:
    exp_dir = exp_dir + args.exp_name_suffix
os.makedirs(exp_dir)

plt.figure(figsize=(20,10))
plt.title('Src and Tar accuracy')
# plt.ylim([0,1])
plt.plot(srcAcc_plt, label='srcAcc')
plt.plot(valAcc_plt, label='tarAcc')
plt.legend(loc=(0.80,0.10), scatterpoints=1)
plt.grid()
plt.savefig(exp_dir+'/acc1.png')
plt.close()

with open(exp_dir+'/report.html', 'w+') as file:
    file.write(
        '''<!DOCTYPE html>
        <html>
        <style>
        img {
            float: right;
        }
        img.one {
            height: auto;
            width: auto;
        }
        </style>
        <body>
        <h4>HyperParams</h4>
        <table><img src="./acc1.png" alt="Accuracy plots" width="80%" height="auto" style="margin-left:15px;">
        <tr>
            <th>Param</th><th>Value</th>
        </tr>'''
    )
    for k,v in opt.items():
        file.write('<tr><td>%s </td><td> %s</td></tr>' % (k,v))
    file.write('</table>\n')
    file.write('<h4>Classifier Network</h4>')
    file.write(str(net2)+'\n')
    file.write('<h4>Base Network - Alexnet</h4>')
    file.write(str(net)+'\n')
    file.write('<h4>Layers frozen</h4>\n')
    for n,param in net.named_parameters():
        file.write('<span style="font-weight:bold">%s</span> %s\n' % (n, param.requires_grad))
    file.write('</body></html>')

with open(exp_dir+'/experiment.log', 'w+') as file:
    file.write('Experiment log\nEpoch\tClsLoss\tSrcAcc\tTarAcc\n')
    epoch = 1.
    for l,s,t in zip(srcClsLoss_plt, srcAcc_plt, valAcc_plt):
        file.write("Epoch:{}\t{:.4f}\t{:.4f}\t{:.4f}\n".format(epoch,l,s,t))
        epoch = epoch + 1.

