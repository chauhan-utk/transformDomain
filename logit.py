import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable

class Logit(nn.Module):
    def __init__(self):
        super(Logit, self).__init__()
    def forward(self, x):
        lg = torch.log(x)
        lg_ = torch.log(1.-x)
        return lg - lg_

class LReLU(nn.Module):
    def __init__(self, leak=0.2):
        super(LReLU, self).__init__()
        self.f1 = 0.5 * (1 + leak)
        self.f2 = 0.5 * (1 - leak)

    def forward(self, x):
        return self.f1 * x + self.f2 * torch.abs(x)

def weight_init(gen):
    for x in gen:
        if isinstance(x, nn.Conv2d) or isinstance(x, nn.ConvTranspose2d):
            init.xavier_normal(x.weight.data, gain=np.sqrt(2))
            if x.bias is not None:
                init.constant(x.bias.data, 0.)
            elif isinstance(x, nn.Linear):
                init.xavier_normal(x.weight.data)
                if x.bias is not None:
                    init.constant(x.bias.data, 0.)

def interpolate(x1, x2, eps=0.5):
    '''
    Arguments :
    epsilon : shape [batch_size, 1]
    Return:
    tensor of size same as x1
    '''
    shape = list(x1.size())[1:]
    p = 1
    for s in shape:
        p *= s
    eps = torch.Tensor([eps]).unsqueeze_(1) # make matrix
    e = torch.mm(eps, torch.ones((1, p)))
    e = e.view_as(x1) # e = e.view([-1] + shape)
    return torch.add(e * x1, (1. - e) * x2)

class flatten(nn.Module):
    '''
    Module to flatten out the input tensor
    '''
    def __init__(self):
        super(flatten, self).__init__()
    def forward(self, x):
        return x.view(x.size(0), -1)

# copied from: https://github.com/ncullen93/torchsample/blob/21507feb258a25bf6924e4844e578624cda72140/torchsample/transforms/tensor_transforms.py#L316
class RangeNormalize(object):
    def __init__(self,
                 min_val,
                 max_val):
        """
        Normalize a tensor between a min and max value
        Arguments
        ---------
        min_val : float
            lower bound of normalized tensor
        max_val : float
            upper bound of normalized tensor
        """
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, *inputs):
        outputs = []
        for idx, _input in enumerate(inputs):
            _min_val = _input.min()
            _max_val = _input.max()
            a = (self.max_val - self.min_val) / (_max_val - _min_val)
            b = self.max_val- a * _max_val
            _input = _input.mul(a).add(b)
            outputs.append(_input)
        return outputs if idx > 1 else outputs[0]

class L2_Loss(nn.Module):
    def __init__(self):
        super(L2_Loss, self).__init__()
    def forward(self, x):
        return torch.sum(x ** 2) / 2

class softmax_cross_entropy_with_logits(nn.Module):
    def __init__(self, num_classes=None):
        super(softmax_cross_entropy_with_logits, self).__init__()
        assert num_classes!=None, "Provide number of classes"
        self.num_classes = num_classes
        self.softmax = nn.Softmax(dim=1)
    def forward(self, logits=None, labels=None):
        ones = torch.sparse.torch.eye(self.num_classes)
        if labels.is_cuda:
            ones = ones.cuda()
        ones = Variable(ones, requires_grad=False)
        labels_one_hot = torch.index_select(ones, 0, labels)
        logits_softmax = self.softmax(logits)
        return (-torch.sum(labels_one_hot * torch.log(logits_softmax), dim=1)).mean()
