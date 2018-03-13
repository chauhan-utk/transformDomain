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
            nn.Conv2d(512, 2048, 5), activation_fn,
            nn.Conv2d(2048, 1024, 3), activation_fn,
            flatten(), nn.Linear(1024,512), activation_fn,
            nn.Linear(512,1)
        )

    def forward(self, x):
        out = self.net(x)
        return out

class ADVDomain(nn.Module):
    def __init__(self, num_classes=None, eps=0.5):
        super(ADVDomain, self).__init__()
        assert num_classes != None, "Specify num_classes"
        resnet18 = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(resnet18.children())[:-2]) # output [batch_size, 512, 7, 7]
        activation_fn = nn.ReLU(inplace=True)
        self.num_classes = num_classes
        self.eps = eps
        self.cls_train = False
        self._discriminator = _Discriminator()
        # self.normalize = RangeNormalize(-1,1)
        self.classifier = nn.Sequential(
            nn.AvgPool2d(7,7), activation_fn, flatten(),
            nn.Linear(512, num_classes)
            )
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

        weight_init(self._discriminator.modules())
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


    def _gradient_penalty(self, X, G, eps):
        """
        Arguments:
        X : shape [batch_size, 512, 7, 7]
        """
        assert X.size() == G.size(), "X size: %s, G size: %s" % (str(X.size()), str(G.size()))
        x_inter = interpolate(X, G, eps)
        d_inter = self._discriminator(x_inter)
        assert d_inter.dim() == 2, "d_inter dim: %s" % str(d_inter.dim())
        ones = torch.ones(d_inter.size()) # since d_inter is not scalar
        grads = autograd.grad(d_inter, x_inter, grad_outputs=ones)
        assert grads.size() == x_inter.size(), "grads size: %s, x_inter size: %s" % (str(grads.size()), str(x_inter.size()))
        assert grads.size(0) == d_inter.size(0), "%s, %s" % (str(grads.size()), str(d_inter.size()))
        grads = grads ** 2
        slopes = torch.sqrt(grads.view(grads.size(0), -1))
        gp_loss = ((slopes - 1.)**2).mean()
        return gp_loss

    def _interpolate(self, x1, x2, eps=0.5):
        '''
        Arguments :
        epsilon : shape [batch_size, 1]
        Return:
        tensor of size same as x1
        '''
        shape = list(x1.size())[1:]
        assert x1.size() == x2.size(), "size mismatch %s %s" % (str(x1.size()), str(x2.size()))
        p = 1
        for s in shape:
            p *= s
        eps = torch.Tensor([eps])
        eps = eps.repeat(x1.size(0)).unsqueeze_(-1)# make matrix
        ones = torch.ones((1,p))
        if x1.is_cuda:
            eps = eps.cuda()
            ones = ones.cuda()
        e = torch.mm(eps, ones)
        # print("e size: ", e.size(), " x1.size: ", x1.size())
        e = e.view_as(x1) # e = e.view([-1] + shape)
        return torch.add(e * x1, (1. - e) * x2)

    def forward(self, x_s, x_t):
        if self.training:
            # assert x_t != None, "give target input for training"
            # print(type(x_s), type(x_t))
            if not self.cls_train:
                X = self.features(x_s) # X
                G = self.features(x_t) # G
                # X_n = self.normalize(X)
                # G_n = self.normalize(G)
                D_ = self._discriminator(G)
                D = self._discriminator(X)
                x_inter = self._interpolate(X.data, G.data, self.eps)
                x_inter = Variable(x_inter, requires_grad=True)
                d_inter = self._discriminator(x_inter)
                return x_inter,d_inter,D_,D
            else:
                X = self.features(x_s)
                out = self.classifier(X)
                return out
            
        else:
            x_s = self.features(x_s)
            out = self.classifier(x_s)
            return out