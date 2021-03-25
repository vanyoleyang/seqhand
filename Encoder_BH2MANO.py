import torch
import torch.nn as nn
import math

class Encoder_BH2MANO(nn.Module):
    def __init__(self):
        super(Encoder_BH2MANO, self).__init__()
        self.fc1 = nn.Linear(63, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, 512)
        self.fc5 = nn.Linear(512, 256)
        self.fc6 = nn.Linear(256, 128)
        self.fc7 = nn.Linear(128, 23)

        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        self.bn6 = nn.BatchNorm1d(128)

    def forward(self, x):
        x = torch.tanh(self.bn1(self.fc1(x)))
        x = torch.tanh(self.bn2(self.fc2(x)))
        x = torch.tanh(self.bn3(self.fc3(x)))
        x = torch.tanh(self.bn4(self.fc4(x)))
        x = torch.tanh(self.bn5(self.fc5(x)))
        x = torch.tanh(self.bn6(self.fc6(x)))
        param = torch.tanh(self.fc7(x))

        pose_param = param[:, :13].clone()
        shape_param = param[:, 13:].clone()
        pose_param[:, :3] *= math.pi
        pose_param[:, 3:] *= 3.5
        shape_param *= 2.
        return pose_param, shape_param

