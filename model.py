from turtle import st
import numpy as np
from torch import nn


class Model(nn.Module):

    def __init__(self, dump: bool = False):
        super(Model, self).__init__()
        self.stack = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(1 - 0.80),
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(1 - 0.80),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(1 - 0.80),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(1 - 0.80),

            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(256, 1024, bias=True),
            nn.ReLU(),
            nn.Linear(1024, 4096, bias=True),
            nn.ReLU(),
            nn.Linear(4096, 3200, bias=True),
            nn.ReLU(),
            nn.Linear(3200, 43, bias=True)
        )

    def forward(self, x):
        return self.stack(x)
