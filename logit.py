import torch
import torch.nn as nn

class Logit(nn.Module):
    def __init__(self):
        super(Logit, self).__init__()
    def forward(self, x):
        lg = torch.log(x)
        lg_ = torch.log(1.-x)
        return lg - lg_

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
