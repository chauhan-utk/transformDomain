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
    def __init__(self):
        super(_Discriminator, self).__init__()
        activation_fn = nn.ReLU(inplace=True)
        self.net = nn.Sequential(
            nn.Linear(512,1, bias=False)
        )

    def forward(self, x):
        out = self.net(x)
        return out

class _Features(nn.Module):
    """
    Input size: [batch_size, 3, 224, 224]
    """
    def __init__(self):
        super(_Features, self).__init__()
        resnet = models.resnet18(pretrained=True)
        params = []
        params += list(resnet.conv1.parameters())
        params += list(resnet.bn1.parameters())
        params += list(resnet.layer1.parameters())
        params += list(resnet.layer2.parameters())
        params += list(resnet.layer3.parameters())
        activation_fn = nn.ReLU(inplace=True)
        for x in params:
                x.requires_grad=False
        resnet.fc = flatten()
        self.basenet = resnet
        self.extra = None
        
    def forward(self, x):
        x = self.basenet(x)
        if self.extra:
            x = self.extra(x)
        return x

class _Classifier(nn.Module):
    def __init__(self, num_classes=None):
        super(_Classifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(512, num_classes)
            )
    def forward(self, x):
        out = self.net(x)
        return out

class WDDomain(nn.Module):
    def __init__(self, num_classes=None, eps=0.5):
        super(WDDomain, self).__init__()
        assert num_classes != None, "Specify num_classes"
        self.features = _Features()
        activation_fn = nn.ReLU(inplace=True)
        self.num_classes = num_classes
        self.eps = eps
        self.cls_train = False
        self._discriminator = _Discriminator()
        self.normalize = RangeNormalize(-1,1)
        self.classifier = _Classifier(self.num_classes)
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

        # weight_init(self.features.modules())
        # weight_init(self._discriminator.modules())
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

    def forward(self, x_s, x_t):
        if self.training:
            # assert x_t != None, "give target input for training"
            # print(type(x_s), type(x_t))
            if not self.cls_train:
                X = self.features(x_s) # X
                G = self.features(x_t) # G
                X = self.normalize(X)
                G = self.normalize(G)
                D_ = self._discriminator(G)
                D = self._discriminator(X)
                return X, G, D_, D
            else:
                X = self.features(x_s)
                # G = self.features(x_t)
                # D_ = self._discriminator(G)
                # D = self._discriminator(X)
                out = self.classifier(X)
                return out, out, out
            
        else:
            x_s = self.features(x_s)
            out = self.classifier(x_s)
            return out