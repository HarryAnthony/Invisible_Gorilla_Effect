import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

#https://github.com/deeplearning-wisc/cider/blob/master/models/resnet.py


model_dict = {
    'ResNet18': [resnet18, 512],
}

class LinearBatchNorm(nn.Module):
    """Implements BatchNorm1d by BatchNorm2d, for SyncBN purpose"""
    def __init__(self, dim, affine=True):
        super(LinearBatchNorm, self).__init__()
        self.dim = dim
        self.bn = nn.BatchNorm2d(dim, affine=affine)

    def forward(self, x):
        x = x.view(-1, self.dim, 1, 1)
        x = self.bn(x)
        x = x.view(-1, self.dim)
        return x

class SupCEResNet(nn.Module):
    """encoder + classifier"""
    def __init__(self, name='resnet18', normalize = False,  num_classes=10):
        super(SupCEResNet, self).__init__()
        model_fun, dim_in = model_dict[name]
        self.encoder = model_fun()
        self.fc = nn.Linear(dim_in, num_classes)
        self.normalize = normalize

    def forward(self, x):
        features = self.encoder(x)
        if self.normalize: 
            features =  F.normalize(features, dim=1)
        return self.fc(features)

class SupCEHeadResNet(nn.Module):
    """encoder + head"""
    def __init__(self, args, multiplier = 1):
        super(SupCEHeadResNet, self).__init__()
        #input(args.model)
        model_fun, dim_in = model_dict[args.net_type]
        #if args.in_dataset == 'ImageNet-100':
        #    model = models.resnet34(pretrained=True)
        #    for name, p in model.named_parameters():
        #        if not name.startswith('layer4'):
        #            p.requires_grad = False
        #    modules=list(model.children())[:-1] # remove last linear layer
        #    self.encoder =nn.Sequential(*modules)
        #else:

        self.encoder = model_fun()
        self.encoder.fc = nn.Identity() 
        self.fc = nn.Linear(dim_in, args.num_classes)
        self.multiplier = multiplier

        if args.head == 'linear':
            #print(dim_in)
            self.head = nn.Linear(dim_in, args.feat_dim)
        elif args.head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, args.feat_dim)
            )
    
    
    def forward(self, x):
        feat = self.encoder(x).squeeze()
        unnorm_features = self.head(feat)
        features= F.normalize(unnorm_features, dim=1)
        return features
    
    def intermediate_forward(self, x, normalize = True):
        feat = self.encoder(x).squeeze()
        if normalize: 
            return F.normalize(feat, dim=1)
        else: 
            return feat