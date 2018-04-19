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
                nn.Conv1d(31,31,70), activation_fn,
                nn.Conv1d(31,31,130), activation_fn,
                nn.Conv1d(31,16,58), activation_fn,
                nn.Conv1d(16,1,1), flatten()
            )

    def forward(self, x):
        out = self.net(x)
        return out

class _Features(nn.Module):
    """
    Input size: [batch_size, 3, 224, 224]
    """
    def __init__(self, activation_fn=nn.ReLU()):
        super(_Features, self).__init__()
        self.net = models.alexnet(pretrained=True)
        self.net.classifier = nn.Sequential(*list(self.net.classifier.children())[:-1])
        self.extra = nn.Linear(4096,256)
                
    def forward(self, x):
        x = self.net(x)
        x = self.extra(x)
        return x

class _Classifier(nn.Module):
    def __init__(self, num_classes=None, activation_fn = nn.ReLU()):
        super(_Classifier, self).__init__()
        self.net = nn.Linear(256, num_classes)
        
    def forward(self, x):
        out = self.net(x)
        return out

class WDDomain(nn.Module):
    def __init__(self, num_classes=None, bn=False):
        super(WDDomain, self).__init__()
        assert num_classes != None, "Specify num_classes"
        self.features = _Features()
        self.num_classes = num_classes
        self.bn = bn
        self._discriminator = _Discriminator(bn=self.bn)
        self.normalize = RangeNormalize(-1,1)
        self.classifier = _Classifier(self.num_classes)

        # weight_init(self.features.modules())
        weight_init(self._discriminator.modules())
        weight_init(self.classifier.modules())
        weight_init(self.features.extra.modules())