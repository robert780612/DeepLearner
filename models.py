import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import pretrainedmodels


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM,self).__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps
    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)       
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)


def get_se_resnet50_gem(pretrain):
    if pretrain == 'imagenet':
        model = pretrainedmodels.__dict__['se_resnet50'](num_classes=1000, pretrained='imagenet')
    else:
        model = pretrainedmodels.__dict__['se_resnet50'](num_classes=1000, pretrained=None)
    model.avg_pool = GeM()
    model.last_linear = nn.Linear(2048, 1)
    return model

def get_se_resnet101_gem(pretrain):
    if pretrain == 'imagenet':
        model = pretrainedmodels.__dict__['se_resnet101'](num_classes=1000, pretrained='imagenet')
    else:
        model = pretrainedmodels.__dict__['se_resnet101'](num_classes=1000, pretrained=None)
    model.avg_pool = GeM()
    model.last_linear = nn.Linear(2048, 1)
    return model

def get_densenet121_gem(pretrain):
    if pretrain == 'imagenet':
        model = pretrainedmodels.__dict__['densenet121'](num_classes=1000, pretrained='imagenet')
    else:
        model = pretrainedmodels.__dict__['densenet121'](num_classes=1000, pretrained=None)
    model.avg_pool = GeM()
    model.last_linear = nn.Linear(1024, 1)
    return model

