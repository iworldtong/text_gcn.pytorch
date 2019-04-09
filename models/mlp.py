#!/usr/bin/env python
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim, dropout_rate=0., num_classes=10):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(input_dim, 200)
        self.fc2 = nn.Linear(200, num_classes)
        
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.fc2(out)

        return out