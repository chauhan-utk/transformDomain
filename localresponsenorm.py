import torch
import torch.nn as nn
from torch.nn import functional as F

class LocalResponseNorm(nn.Module):
  def __init__(self, size, alpha=1e-4, beta=0.75, k=1):
    super(LocalResponseNorm, self).__init__()
    self.size = size
    self.alpha = alpha
    self.beta = beta
    self.k = k    

  def forward(self, input):
    dim = input.dim()
    # print("Local Response Norm")
    if dim < 3:
      raise ValueError('Expected 3D or higher dimensionality \
                                input (got {} dimensions)'.format(dim))
    div = input.mul(input).unsqueeze(1)
    if dim == 3:
      div = F.pad(div, (0, 0, self.size // 2, (self.size - 1) // 2))
      div = F.avg_pool2d(div, (self.size, 1), stride=1).squeeze(1)
    else:
      sizes = input.size()
      div = div.view(sizes[0], 1, sizes[1], sizes[2], -1)
      div = F.pad(div, (0, 0, 0, 0, self.size // 2, (self.size - 1) // 2))
      div = F.avg_pool3d(div, (self.size, 1, 1), stride=1).squeeze(1)
      div = div.view(sizes)
    div = div.mul(self.alpha).add(self.k).pow(self.beta)        
    return input / div    

  def __repr__(self):
    return self.__class__.__name__ + '(' \
      + str(self.size) \
      + ', alpha=' + str(self.alpha) \
      + ', beta=' + str(self.beta) \
      + ', k=' + str(self.k) + ')'