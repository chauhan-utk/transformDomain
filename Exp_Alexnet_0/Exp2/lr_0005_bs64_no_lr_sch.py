
# coding: utf-8

# In[26]:


gpu=0


# In[72]:


import sys
sys.path.append('../../')
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
from alexnet import alexnet


# In[28]:


net = alexnet
net.load_state_dict(torch.load('../../alexnet.pth'))
# print(net)


# In[29]:


net = nn.Sequential(*list(net.children())[:-2])
# print(net)


# In[30]:


src = domainData['amazon']
tar = domainData['webcam']


# In[31]:


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


# In[32]:


src_dataset = datasets.ImageFolder(src, transform=src_transforms)
tar_dataset = datasets.ImageFolder(tar, transform=tar_transforms)
srcSampler = torch.utils.data.sampler.RandomSampler(src_dataset)
tarSampler = torch.utils.data.sampler.RandomSampler(tar_dataset)
srcDataLen = len(src_dataset)
tarDataLen = len(tar_dataset)
use_gpu = True and torch.cuda.is_available()


# In[33]:


opt = {
    'src': 'Amazon',
    'tar': 'Webcam',
    'manual_seed':1,
    'batchSize':64,
    'use_gpu': use_gpu,
    'num_classes': 31,
    'epochs': 2,
    'momentum': 0.9,
    'lr': 5e-4,
    'lr_sch': 0,
    'lr_sch_gamma': 0.1,
    'p_lr_decay': 2,
    'n0': 1.,
    'alpha': 10,
    'beta': 0.75,
    'betas': (0.5,0.99),
    'net_wtDcy': 1e-5,
    'btl_wtDcy': 1e-5,
    'srcDataLen': srcDataLen,
    'tarDataLen': tarDataLen
}


# In[34]:


torch.manual_seed(opt['manual_seed'])
if opt['use_gpu']: torch.cuda.manual_seed(opt['manual_seed'])


# In[35]:


print("use_gpu: ", opt['use_gpu'])


# In[36]:


src_dataloader = torch.utils.data.DataLoader(src_dataset, batch_size=opt['batchSize'], 
                                             shuffle=(srcSampler is None), sampler=srcSampler,
                                            num_workers=2, pin_memory=True, drop_last=False)
tar_dataloader = torch.utils.data.DataLoader(tar_dataset, batch_size=opt['batchSize'],
                                            shuffle=(tarSampler is None), sampler=tarSampler,
                                            num_workers=2, pin_memory=True, drop_last=False)


# In[37]:


def init_weights(m):
#     print(m)
    if isinstance(m, nn.Linear):
        init.xavier_normal(m.weight)
        init.constant(m.bias, 0.1)


# In[38]:


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.classifier = nn.Sequential(nn.Linear(4096,256),
                                       nn.ReLU(inplace=True), nn.Linear(256, opt['num_classes']))
        self.classifier.apply(init_weights)
    
    def forward(self, x):
        x = self.classifier(x)
        return x


# In[39]:


net2 = Model()


# In[40]:


if opt['use_gpu']:
    net = net.cuda(gpu)
    net2 = net2.cuda(gpu)
    torch.backends.cudnn.enabled=True
    torch.backends.cudnn.benchmark=True


# In[41]:


criterion = nn.CrossEntropyLoss()
softmax = nn.Softmax(dim=1)


# In[42]:


netfeatwt, netfeatbias = list(), list()
for name, param in net.named_parameters():
    if name in ['10.weight', '12.weight', '16.1.weight', '19.1.weight']: netfeatwt.append(param)
    elif name in ['10.bias', '12.bias', '16.1.bias', '19.1.bias']: netfeatbias.append(param)
    else: param.requires_grad=False


# In[43]:


for name, param in net.named_parameters():
    print(name, param.requires_grad)


# In[44]:


net2_weight, net2_bias = list(), list()
for name, param in net2.named_parameters():
    if 'weight' in name: net2_weight.append(param)
    elif 'bias' in name: net2_bias.append(param)


# In[45]:


sgd_params = [
    {'params': netfeatwt, 'lr': opt['lr'], 'momentum': opt['momentum'], 'weight_decay': opt['net_wtDcy'], 'lr_mul': 1., 'name': 'netfeatwt'},
    {'params': netfeatbias, 'lr': opt['lr'], 'momentum': opt['momentum'], 'weight_decay': opt['net_wtDcy'], 'lr_mul': 2., 'name': 'netfeatbias'},
    {'params': net2_weight, 'lr': opt['lr'], 'momentum': opt['momentum'], 'weight_decay': opt['btl_wtDcy'], 'lr_mul': 10., 'name': 'net2wt'},
    {'params': net2_bias, 'lr': opt['lr'], 'momentum': opt['momentum'], 'weight_decay': opt['btl_wtDcy'], 'lr_mul': 20., 'name': 'net2bias'}
]

# optimizer = optim.Adam(net2.parameters(), lr=opt['lr'], betas=opt['betas'], weight_decay=opt['weight_decay'])
# optimizer = optim.Adam(sgd_params, betas=opt['betas'])

optimizer_2 = optim.SGD(sgd_params)
# optimizer_2 = optim.SGD(net2.parameters(), lr=opt['lr'], momentum=opt['momentum'], weight_decay=opt['btl_wtDcy'])

lr_sch = None
if opt['lr_sch']=='step': lr_sch = optim.lr_scheduler.StepLR(optimizer, 20, opt['lr_sch_gamma'])
if opt['lr_sch']=='exponential': lr_sch = optim.lr_scheduler.ExponentialLR(optimizer_2, opt['lr_sch_gamma'])


# In[46]:


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


# In[47]:


srcClsLoss_plt, srcAcc_plt, valAcc_plt = list(), list(), list()


# In[48]:

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
#     experiment.log_metric("src_clsLoss", srcLoss, step=epoch)
#     experiment.log_metric("src_clsAcc", srcAcc, step=epoch)
#     experiment.log_metric('val_accuracy', valAcc, step=epoch)
#     experiment.log_epoch_end(epoch)
    print("Epoch: {}, train loss: {:.4f}, train acc: {:.4f}, validation acc: {:.4f}".
          format(epoch+1, srcLoss, srcAcc, valAcc))
    srcClsLoss_plt.append(srcLoss), srcAcc_plt.append(srcAcc), valAcc_plt.append(valAcc)


# In[70]:


plt.figure(figsize=(20,10))
plt.title('Src and Tar accuracy')
# plt.ylim([0,1])
plt.plot(srcAcc_plt, label='srcAcc')
plt.plot(valAcc_plt, label='tarAcc')
plt.legend(loc=(0.80,0.10), scatterpoints=1)
plt.grid()
plt.savefig('acc1.png')
plt.close()


# In[ ]:


with open('report.html', 'w+') as file:
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


# In[82]:


with open('experiment.log', 'w+') as file:
    file.write('Experiment log\nEpoch\tClsLoss\tSrcAcc\tTarAcc\n')
    epoch = 1.
    for l,s,t in zip(srcClsLoss_plt, srcAcc_plt, valAcc_plt):
        file.write("Epoch:{}\t{:.4f}\t{:.4f}\t{:.4f}\n".format(epoch,l,s,t))
        epoch = epoch + 1.

