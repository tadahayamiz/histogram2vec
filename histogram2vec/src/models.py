# -*- coding: utf-8 -*-
"""
Created on Fri 29 15:46:32 2022

inspired by the following:
https://data-analytics.fun/2022/01/27/pytorch-vq-vae/

@author: tadahaya
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class Residual(nn.Module):
    def __init__(self, in_channels, hidden_dim, residual_hidden_dim):
        super(Residual, self).__init__()
        self._conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=residual_hidden_dim,
            kernel_size=3, stride=1, padding=1
        )
        self._conv2 = nn.Conv2d(
            in_channels=residual_hidden_dim,
            out_channels=hidden_dim,
            stride=1, padding=1
        )


    def forward(self, x):
        h = torch.relu(x)
        h = torch.relu(self._conv1(h))
        h = self._conv2(h)
        return h + x
        

class ResidualStack(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_residual_layers, residual_hidden_dim):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList(
            [Residual(in_channels, hidden_dim, residual_hidden_dim) for _ in range(num_residual_layers)]
        )


    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)


class Encoder(nn.Module):
    def __init__(
            self, in_channels, hidden_dim, num_residual_layers,
            residual_hidden_dim, name=None
            ):
        self._in_channels = in_channels
        self._hidden_dim = hidden_dim
        self._num_residual_layers = num_residual_layers
        self._residual_hidden_dim = residual_hidden_dim
        self._enc1 = nn.Conv2d(
            in_channels, hidden_dim // 2, kernel_size=4,
            stride=2, padding=1
                                )
        self._enc2 = nn.Conv2d(
            hidden_dim // 2, hidden_dim, kernel_size=4,
            stride=2, padding=1
                                )
        self._enc3 = nn.Conv2d(
            hidden_dim, hidden_dim, kernel_size=3,
            stride=1, padding=1
                                )
        self._residual_stack(
            hidden_dim, hidden_dim, num_residual_layers, residual_hidden_dim
        )
    

    def forward(self, x):
        h = torch.relu(self._enc1(x))
        h = torch.relu(self._enc2(h))
        h = self._enc3(h)
        return self._residual_stack(h)
        

class Decoder(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_residual_layers, residual_hidden_dim):
        super(Decoder, self).__init__()
        self._dec1 = None
        self._residual_stack = None
        self._dec2 = None
        self._dec3 = None


    def forward(self, x):
        h = self._dec1(x)
        h = self._residual_stack(h)
        h = torch.relu(self._dec2(h))
        h = self._dec3(h)
        return torch.sigmoid(h)




class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        return F.avg_pool2d(x,kernel_size=x.size()[2:]).view(-1,x.size(1))
