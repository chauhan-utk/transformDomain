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
    def __init__(self, bn=False):
        super(_Discriminator, self).__init__()
        # activation_fn = nn.ReLU(inplace=True)
        activation_fn = nn.LeakyReLU(0.2)
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
                nn.Linear(512,128), activation_fn,
                nn.Linear(128,1)
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
        params = []
        params += list(resnet.conv1.parameters())
        params += list(resnet.bn1.parameters())
        params += list(resnet.layer1.parameters())
        params += list(resnet.layer2.parameters())
        params += list(resnet.layer3.parameters())
        # params += list(resnet.layer4.parameters())
        for x in params:
            x.requires_grad=False
        resnet.fc = nn.Sequential(flatten(), nn.Linear(2048, 512), activation_fn)
        self.basenet = resnet
        self.extra = None
        
    def forward(self, x):
        x = self.basenet(x)
        if self.extra:
            x = self.extra(x)
        return x

class _Classifier(nn.Module):
    def __init__(self, num_classes=None, activation_fn = LReLU()):
        super(_Classifier, self).__init__()
        # activation_fn = nn.ReLU(inplace=True)
        self.net = nn.Sequential(
            nn.Linear(512, num_classes)
            )
    def forward(self, x):
        out = self.net(x)
        return out

class WDDomain(nn.Module):
    def __init__(self, num_classes=None, eps=0.5, bn=False):
        super(WDDomain, self).__init__()
        assert num_classes != None, "Specify num_classes"
        self.features = _Features()
        # activation_fn = nn.ReLU(inplace=True)
        activation_fn = LReLU()
        self.num_classes = num_classes
        self.eps = eps
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

    def load_cls(self, cls_path=None, ft_path=None):
        try:
            if cls_path:
                self.classifier.load_state_dict(torch.load(cls_path))
            else:
                self.classifier.load_state_dict(torch.load("classifier_dump.pth"))
            if ft_path:
                self.features.load_state_dict(torch.load(ft_path))
            else:
                self.features.load_state_dict(torch.load("features_dump.pth"))
        except:
            print("Error in loading the saved weights")

    def forward(self, x):
        if self.training:
            # assert x_t != None, "give target input for training"
            # print(type(x_s), type(x_t))
            if not self.cls_train:
                X = self.features(x)
                X_ = self.normalize(X)
                out = self._discriminator(X_)
                return X_, out
            else:
                X = self.features(x)
                out = self.classifier(X)
                return out
            
        else:
            x = self.features(x)
            out = self.classifier(x)
            return out