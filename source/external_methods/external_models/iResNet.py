'''
This project makes use of code from the following open-source repository:

https://github.com/EvZissel/Residual-Flow/blob/master/Residual%20_flow_train.py

We thank the authors for making their work publicly available.

'''
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import pickle
import json
import math


class RealNVP(nn.Module):
    def __init__(self, mask, num_features, length_hidden, A, A_inv, log_abs_det_A_inv):
        super(RealNVP, self).__init__()

        self.mask = nn.Parameter(mask, requires_grad=False)
        self.t = torch.nn.ModuleList([Nett(num_features, length_hidden) for _ in range(len(mask))])
        self.s = torch.nn.ModuleList([Nets(num_features, length_hidden) for _ in range(len(mask))])
        # self.bn_flow = torch.nn.ModuleList([BatchNormStats1d(num_features) for _ in range(len(mask))])
        self.perm = torch.nn.ModuleList([Permutation(num_features) for _ in range(int(len(mask)/2))])
        self.A_ = nn.Parameter(A_inv, requires_grad=False)
        self.A = nn.Parameter(A, requires_grad=False)
        self.log_abs_det_A_ = nn.Parameter(log_abs_det_A_inv, requires_grad=False)

                # ✅ Initialize weights with He initialization
        def init_weights_he(m):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

        self.s.apply(init_weights_he)
        self.t.apply(init_weights_he)

        for i in range(len(mask)):
            #self.t[i].fc3.weight.data.fill_(0)
            #self.t[i].fc3.bias.data.fill_(0)
            #self.s[i].fc3.weight.data.fill_(0)
            #self.s[i].fc3.bias.data.fill_(0)
            self.t[i].fc6.weight.data.fill_(0)
            self.t[i].fc6.bias.data.fill_(0)
            self.s[i].fc6.weight.data.fill_(0)
            self.s[i].fc6.bias.data.fill_(0)

     

    def g(self, z, training):
        x = z.cuda()
        #zeros = torch.cuda.FloatTensor(x.shape).fill_(0)
        for i in range(len(self.t)):
            if (i % 2 > 0):
                # x, var = self.bn_flow[i].forward(x, training, inverse=True)
                x = self.perm[i // 2].forward(x, inverse=False)

            x_ = x * self.mask[i]
            s = self.s[i](x_) * (1 - self.mask[i])
            t = self.t[i](x_) * (1 - self.mask[i])
            x =  x_ + (1 - self.mask[i]) * (x * torch.exp(s) + t)

        x = torch.mm(self.A,x.transpose(1,0)).transpose(1,0)
        return x

    def f(self, x, training):
        log_det_J, z = torch.cuda.FloatTensor(x.shape[0]).fill_(0), x.cuda()
        z = torch.mm(self.A_,z.transpose(1,0)).transpose(1,0)
        log_det_J += self.log_abs_det_A_

        for i in reversed(range(len(self.t))):

            z_ = self.mask[i] * z
            s = self.s[i](z_) * (1 - self.mask[i])
            t = self.t[i](z_) * (1 - self.mask[i])
            z = (1 - self.mask[i]) * (z - t) * torch.exp(-1*s) + z_
            log_det_J -= s.sum(dim=1)

            if (i % 2 == 0):
                j = int(i/2)
                z = self.perm[j].forward(z, inverse=False)

        return z, log_det_J

    def log_prob(self, x, training):
        z, log_det_jacobian = self.f(x.cuda(), training)
        logp_z = -0.5 * z.pow(2).sum(dim=1)              # −½ ∥z∥²
        logp_z += -0.5 * z.size(1) * math.log(2 * math.pi)  # −½ d log(2π)

        return logp_z + log_det_jacobian
    



    def sample(self, batch_size):
        num_features = self.A_.shape[1]  # (d, d) → use the input dimensionality
        z = torch.randn(batch_size, num_features, device=self.A_.device)
        x = self.g(z, False)
        return x

class Nets(nn.Module):
    def __init__(self,num_features, length_hidden):
        super().__init__()
        # self.net = nn.Linear(num_features, num_features)
        self.fc1 = nn.Linear(num_features, int(length_hidden*num_features))
        self.fc2 = nn.Linear(int(length_hidden*num_features), int(length_hidden*num_features))
        self.fc3 = nn.Linear(int(length_hidden*num_features), num_features)
        self.fc4 = nn.Linear(int(length_hidden * num_features), int(length_hidden*num_features))
        self.fc5 = nn.Linear(int(length_hidden * num_features), num_features)
        self.fc6 = nn.Linear(int(length_hidden * num_features), num_features)
        self.rescale = nn.utils.weight_norm(Rescale(num_features))

    def forward(self, x):
        # x_ = self.net(x)
        x_ = self.fc1(x)
        x_ = F.leaky_relu(x_, inplace=True)
        x_ = self.fc2(x_)
        x_ = F.leaky_relu(x_, inplace=True)
        x_ = self.fc3(x_)
        x_ = F.leaky_relu(x_, inplace=True)
        x_ = self.fc4(x_)
        x_ = F.leaky_relu(x_, inplace=True)
        x_ = self.fc5(x_)
        x_ = F.leaky_relu(x_, inplace=True)
        x_ = self.fc6(x_)
        x_ = self.rescale(torch.tanh(x_))
        return x_

class Nett(nn.Module):
    def __init__(self,num_features, length_hidden):
        super(Nett, self).__init__()
        self.fc1 = nn.Linear(num_features, int(length_hidden*num_features))
        self.fc2 = nn.Linear(int(length_hidden*num_features), int(length_hidden*num_features))
        self.fc3 = nn.Linear(int(length_hidden*num_features), num_features)
        self.fc4 = nn.Linear(int(length_hidden * num_features), int(length_hidden*num_features))
        self.fc5 = nn.Linear(int(length_hidden * num_features), num_features)
        self.fc6 = nn.Linear(int(length_hidden * num_features), num_features)


    def forward(self, x):
        x_ = self.fc1(x)
        x_ = F.leaky_relu(x_, inplace=True)
        x_ = self.fc2(x_)
        x_ = F.leaky_relu(x_, inplace=True)
        x_ = self.fc3(x_)
        x_ = F.leaky_relu(x_, inplace=True)
        x_ = self.fc4(x_)
        x_ = F.leaky_relu(x_, inplace=True)
        x_ = self.fc5(x_)
        x_ = F.leaky_relu(x_, inplace=True)
        x_ = self.fc6(x_)
        return x_


class Rescale(nn.Module):
    """Per-channel rescaling. Need a proper `nn.Module` so we can wrap it
    with `torch.nn.utils.weight_norm`.
    Args:
        num_channels (int): Number of channels in the input.
    """
    def __init__(self, num_features):
        super(Rescale, self).__init__()
        self.weight = nn.Parameter(torch.ones(1, num_features))

    def forward(self, x):
        x = self.weight * x
        return x

class Nett_linear(nn.Module):
    def __init__(self,num_features):
        super(Nett_linear, self).__init__()
        self.net = nn.Linear(num_features, num_features, bias=False)

    def forward(self, x):
        x_ = self.net(x)
        return x_

class Nets_linear(nn.Module):
    def __init__(self,num_features):
        super(Nets_linear, self).__init__()
        self.net = nn.Linear(num_features, num_features)

    def forward(self, x):
        x_ = self.net(x)
        return x_


class BatchNormStats1d(nn.Module):
    """Compute BatchNorm1d normalization statistics: `mean` and `var`.
    Useful for keeping track of sum of log-determinant of Jacobians in flow models.
    Args:
        num_features (int): Number of features in the input.
        eps (float): Added to the denominator for numerical stability.
        decay (float): The value used for the running_mean and running_var computation.
            Different from conventional momentum, see `nn.BatchNorm1d` for more.
    """
    def __init__(self, num_features, eps=1e-5, decay=0.1):
        super(BatchNormStats1d, self).__init__()
        self.eps = eps

        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.weights = nn.Parameter(torch.ones(1,num_features), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(1,num_features), requires_grad=True)
        self.decay = decay
        self.init = True

    def forward(self, x, training, inverse):
        # Get mean and variance per channel
        if self.init == True:
            init_mean, init_var = x.mean(0), x.var(0)
            self.weights.data = init_var.sqrt()
            self.bias.data = init_mean
            self.running_mean = init_mean
            self.running_var = init_var
            self.init = False

        if training:
            used_mean, used_var = x.mean(0), x.var(0)
            curr_mean, curr_var = used_mean, used_var

            # Update variables
            tmp_running_mean = self.running_mean - self.decay * (self.running_mean - curr_mean)
            tmp_running_var = self.running_var - self.decay * (self.running_var - curr_var)

            self.running_mean = tmp_running_mean.detach().clone()
            self.running_var = tmp_running_var.detach().clone()

        else:
            used_mean = self.running_mean.detach().clone()
            used_var = self.running_var.detach().clone()

        # used_var += self.eps

        # Reshape
        used_mean = used_mean.view(1, x.size(1)).expand_as(x)
        used_var = used_var.view(1, x.size(1)).expand_as(x)

        used_weights = self.weights
        used_bias = self.bias

        used_weights = used_weights.view(1, x.size(1)).expand_as(x)
        used_bias = used_bias.view(1, x.size(1)).expand_as(x)
        if inverse:
            x = (x - used_bias) / used_weights
            x = x * used_var.sqrt()  + used_mean
        else:
            x = (x - used_mean) / used_var.sqrt()
            x = used_weights * x + used_bias

        return x, used_var, used_weights


class Permutation(nn.Module):
    """Permutation matrix with log determinant of zero.
    Args:
        num_channels (int): Number of channels in the input.
    """
    def __init__(self, num_features):
        super(Permutation, self).__init__()
        p = torch.randperm(num_features)
        self.register_buffer('perm', p)
        self.register_buffer('inv_perm', torch.LongTensor([(p == l).nonzero() for l in range(len(p))]))

        eye = torch.eye(num_features)
        self.register_buffer('W', eye[p, :])

    def forward(self, x, inverse = False):
        if inverse:
            x = x[:,self.inv_perm]
        else:
            x = x[:,self.perm]

        return x
