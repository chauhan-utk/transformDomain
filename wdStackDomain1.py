'''
Extract features from a base network and provide generator, critic and a classifier for those features.
'''

import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable
from torchvision import models
import torch.nn.init as init
import numpy as np
from logit import *

class _Discriminator(nn.Module):
    '''
    forward() expects shape [batch_size, 512, 7, 7]
    '''
    def __init__(self, activation_fn = nn.LeakyReLU(), bn=False):
        super(_Discriminator, self).__init__()
        self.net = None
        if bn:
            self.net = nn.Sequential(
                nn.Linear(512, 256, bias=False), 
                nn.BatchNorm2d(), activation_fn,
                nn.Linear(256,128, bias=False), 
                nn.BatchNorm2d(), activation_fn,
                nn.Linear(128,1)
            )
        else:
            self.net = nn.Sequential(
                nn.Linear(512, 256), activation_fn,
                nn.Linear(256, 100), activation_fn,
                nn.Linear(100, 1)
            )

    def forward(self, x):
        out = self.net(x)
        return out

class _Features(nn.Module):
    """
    Input size: [batch_size, 3, 224, 224]
    """
    def __init__(self, activation_fn=nn.LeakyReLU()):
        super(_Features, self).__init__()
        resnet = models.resnet50(pretrained=True)
        # params = []
        # params += list(resnet.conv1.parameters())
        # params += list(resnet.bn1.parameters())
        # params += list(resnet.layer1.parameters())
        # params += list(resnet.layer2.parameters())
        # params += list(resnet.layer3.parameters())
        # params += list(resnet.layer4.parameters())
        for x in resnet.parameters():
            x.requires_grad=False
        resnet.fc = nn.Sequential(flatten(), nn.Linear(2048, 1024), activation_fn,
            nn.Linear(1024, 512), activation_fn)
        self.basenet = resnet
        self.extra = None
        
    def forward(self, x):
        x = self.basenet(x)
        if self.extra:
            x = self.extra(x)
        return x

class _Classifier(nn.Module):
    def __init__(self, num_classes=None, activation_fn = nn.LeakyReLU()):
        super(_Classifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(512, 256), activation_fn,
            nn.Linear(256, 128), activation_fn,
            nn.Linear(128, num_classes)
            )
    def forward(self, x):
        out = self.net(x)
        return out

class WDDomain(nn.Module):
    def __init__(self, num_classes=None, bn=False):
        super(WDDomain, self).__init__()
        assert num_classes != None, "Specify num_classes"
        self.features = _Features()
        self.num_classes = num_classes
        self.cls_train = False
        self.bn = bn
        self._discriminator = _Discriminator(bn=self.bn)
        self.normalize = RangeNormalize(-1,1)
        self.classifier = _Classifier(self.num_classes)

        # weight_init(self.features.modules())
        weight_init(self._discriminator.modules())
        if isinstance(self.features.basenet.fc, nn.Sequential):
            weight_init(self.features.basenet.fc.modules())
        if self.features.extra:
            weight_init(self.features.extra.modules())
        weight_init(self.classifier.modules())