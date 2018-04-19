
# coding: utf-8

get_ipython().magic('set_env CUDA_DEVICE_ORDER=PCI_BUS_ID')
get_ipython().magic('set_env CUDA_VISIBLE_DEVICES=1')

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
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from alexnet import alexnet


net = alexnet
net.load_state_dict(torch.load('alexnet.pth'))
# print(net)

net = nn.Sequential(*list(net.children())[:-2])
print(net)


# In[5]:


src = domainData['amazon']
tar = domainData['webcam']


# In[6]:


tmp_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(227, scale=(0.50,1.0), ratio=(1.,1.)),
    transforms.RandomHorizontalFlip()
])
tmp_dataset = datasets.ImageFolder(src, transform=tmp_transforms)


# In[7]:


import random
tmp_len = len(tmp_dataset)


# In[8]:


tmp_dataset[random.randint(0,tmp_len-1)][0]


# In[9]:


src_transforms = transforms.Compose([
    transforms.Resize(256),
#     transforms.CenterCrop(227),
#     transforms.RandomVerticalFlip(),
    transforms.RandomResizedCrop(227, scale=(0.50,1.0), ratio=(1.,1.)),
#     transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[1.,1.,1.]),
    transforms.Lambda(lambda x: torch.stack([x[2],x[1],x[0]])), # RGB -> BGR
    transforms.Lambda(lambda x: x * 255.)
])
tar_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(227),
#     transforms.RandomResizedCrop(224, scale=(0.25,1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[1.,1.,1.]),
    transforms.Lambda(lambda x: torch.stack([x[2],x[1],x[0]])),
    transforms.Lambda(lambda x: x * 255.)
])


# In[10]:


src_dataset = datasets.ImageFolder(src, transform=src_transforms)
tar_dataset = datasets.ImageFolder(tar, transform=tar_transforms)
srcSampler = torch.utils.data.sampler.RandomSampler(src_dataset)
tarSampler = torch.utils.data.sampler.RandomSampler(tar_dataset)
srcDataLen = len(src_dataset)
tarDataLen = len(tar_dataset)
use_gpu = True and torch.cuda.is_available()


# In[11]:


opt = {
    'src': 'Amazon',
    'tar': 'Webcam',
    'manual_seed':1,
    'batchSize':64,
    'use_gpu': use_gpu,
    'num_classes': 31,
    'epochs': 250,
    'momentum': 0.9,
    'lr': 2e-4,
    'lr_sch': 0,
    'lr_sch_gamma': 0.1,
    'p_lr_decay': 250,
    'n0': 1.,
    'alpha': 10,
    'beta': 0.75,
    'betas': (0.5,0.99),
    'net_wtDcy': 0.001,
    'btl_wtDcy': 0.001,
    'srcDataLen': srcDataLen,
    'tarDataLen': tarDataLen
}


# experiment.log_multiple_params(opt)

# In[12]:


torch.manual_seed(opt['manual_seed'])
if opt['use_gpu']: torch.cuda.manual_seed(opt['manual_seed'])


# In[13]:


print("use_gpu: ", opt['use_gpu'])


# In[14]:


src_dataloader = torch.utils.data.DataLoader(src_dataset, batch_size=opt['batchSize'], 
                                             shuffle=(srcSampler is None), sampler=srcSampler,
                                            num_workers=2, pin_memory=True, drop_last=False)
tar_dataloader = torch.utils.data.DataLoader(tar_dataset, batch_size=opt['batchSize'],
                                            shuffle=(tarSampler is None), sampler=tarSampler,
                                            num_workers=2, pin_memory=True, drop_last=False)


# In[15]:


def init_weights(m):
#     print(m)
    if isinstance(m, nn.Linear):
        init.xavier_normal(m.weight)
        init.constant(m.bias, 0.1)


# In[16]:


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
    net = net.cuda()
    net2 = net2.cuda()
    torch.backends.cudnn.enabled=True
    torch.backends.cudnn.benchmark=True


# To computer norm of various parameters
# 
# ```python
# for name, param in net2.named_parameters():
#     nrm = torch.norm(param, 2)
#     zero = param.eq(0.).float().sum()
#     nele = torch.numel(param)
#     print(name, nrm.data[0], nele, zero.data[0])
# ```

# In[19]:


criterion = nn.CrossEntropyLoss()
softmax = nn.Softmax(dim=1)

net_featWtNm, net_featBiasNm = list(), list()
for name, _ in net.named_parameters():
    if 'weight' in name: net_featWtNm.append(name)
    elif 'bias' in name: net_featBiasNm.append(name)
    else: print("some error in saving trainable layer names")

net_featWtNm = net_featWtNm[st_lyr:]
net_featBiasNm = net_featBiasNm[st_lyr:]

netfeatwt, netfeatbias = list(), list()
for name, param in net.named_parameters():
    if name in net_featWtNm: netfeatwt.append(param)
    elif name in net_featBiasNm: netfeatbias.append(param)
    else: param.requires_grad=False


# In[21]:


for name, param in net.named_parameters():
    print(name, param.requires_grad)


# In[22]:


net2_weight, net2_bias = list(), list()
for name, param in net2.named_parameters():
    if 'weight' in name: net2_weight.append(param)
    elif 'bias' in name: net2_bias.append(param)


# In[23]:


sgd_params = [
    {'params': netfeatwt, 'lr': opt['lr'], 'momentum': opt['momentum'], 'weight_decay': opt['net_wtDcy'], 'lr_mul': 1., 'name': 'netfeatwt'},
    {'params': netfeatbias, 'lr': opt['lr'], 'momentum': opt['momentum'], 'weight_decay': 0*opt['net_wtDcy'], 'lr_mul': 2., 'name': 'netfeatbias'},
    {'params': net2_weight, 'lr': opt['lr'], 'momentum': opt['momentum'], 'weight_decay': opt['btl_wtDcy'], 'lr_mul': 10., 'name': 'net2wt'},
    {'params': net2_bias, 'lr': opt['lr'], 'momentum': opt['momentum'], 'weight_decay': 0*opt['btl_wtDcy'], 'lr_mul': 20., 'name': 'net2bias'}
]

# optimizer = optim.Adam(net2.parameters(), lr=opt['lr'], betas=opt['betas'], weight_decay=opt['weight_decay'])
# optimizer = optim.Adam(sgd_params, betas=opt['betas'])

optimizer_2 = optim.SGD(sgd_params)
# optimizer_2 = optim.SGD(net2.parameters(), lr=opt['lr'], momentum=opt['momentum'], weight_decay=opt['btl_wtDcy'])

lr_sch = None
if opt['lr_sch']=='step': lr_sch = optim.lr_scheduler.StepLR(optimizer, 20, opt['lr_sch_gamma'])
if opt['lr_sch']=='exponential': lr_sch = optim.lr_scheduler.ExponentialLR(optimizer_2, opt['lr_sch_gamma'])


# In[24]:


def get_validation_acc(cm=False):
    tarData = iter(tar_dataloader)
    totalCorrects = 0.
    if cm: y_preds, y_true = list(), list()
    for tarimgs, tarlbls in tarData:
        tarimgs = tarimgs.cuda() if opt['use_gpu'] else tarimgs
        tarlbls = tarlbls.cuda() if opt['use_gpu'] else tarlbls      
        tarimgs = Variable(tarimgs, volatile=True)

        feat_ = net(tarimgs)
        logits = net2(feat_)

        _, preds = torch.max(softmax(logits).data, 1)
        totalCorrects += torch.eq(preds, tarlbls).float().sum()
        if cm: y_preds.append(preds.cpu().numpy()), y_true.append(tarlbls.cpu().numpy())
    valAcc = totalCorrects / opt['tarDataLen']
    if cm: return valAcc, y_preds, y_true
    return valAcc


# In[25]:


p = np.linspace(float(1./opt['p_lr_decay']),1,opt['p_lr_decay'])

for epoch in range(opt['epochs']):
    srcData = iter(src_dataloader)
    totalCorrects = 0.
    totalClsLoss = 0.
#     experiment.log_current_epoch(epoch)
    
    n_p = opt['n0'] / pow((1. + opt['alpha'] * p[epoch]), (opt['beta']))
    print("n_p: ", n_p)
    for param_group in optimizer_2.param_groups:
        param_group['lr'] = opt['lr'] * param_group['lr_mul'] * n_p
        
    for srcimgs, srclbls in srcData:
        srcimgs = srcimgs.cuda() if opt['use_gpu'] else srcimgs
        srclbls = srclbls.cuda() if opt['use_gpu'] else srclbls
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
#     experiment.log_metric("src_clsLoss", srcLoss, step=epoch)
#     experiment.log_metric("src_clsAcc", srcAcc, step=epoch)
#     experiment.log_metric('val_accuracy', valAcc, step=epoch)
#     experiment.log_epoch_end(epoch)
    print("Epoch: {}, train loss: {:.4f}, train acc: {:.4f}, validation acc: {:.4f}".
          format(epoch+1, srcLoss, srcAcc, valAcc))


# ```
# valAcc, y_preds, y_true = get_validation_acc(cm=True)
# 
# y_preds = [d for sublist in y_preds for d in sublist]
# y_true = [d for sublist in y_true for d in sublist]
# 
# cm = confusion_matrix(y_true, y_preds)
# 
# np.set_printoptions(precision=2)
# plt.figure(figsize=(20,20))
# utils.plot_confusion_matrix(cm, classes=src_dataset.classes, normalize=False)
# plt.show()
# ```
