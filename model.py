import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

np.random.seed(0)
torch.manual_seed(0)

class Imggen(nn.Module):

    def __init__(self, in_features=16, width=16, height=16):
        super(Imggen, self).__init__()
        self.F1 = nn.Linear(in_features, 128)
        self.F2 = nn.Linear(128, 512)
        self.F3 = nn.Linear(512, 512)
        self.F4 = nn.Linear(512, width * height * 3)

    def forward(self, x):
        x = F.relu(self.F1(x))
        x = F.relu(self.F2(x))
        x = F.relu(self.F3(x))
        return F.relu(self.F4(x))

