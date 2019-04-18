import torch
import torch.nn as nn
from config import config


class DDPGActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(33 * config['history_len'], config['a_fc1'])
        self.bn1 = nn.BatchNorm1d(config['a_fc1'])
        self.fc2 = nn.Linear(config['a_fc1'], config['a_fc2'])
        self.bn2 = nn.BatchNorm1d(config['a_fc2'])
        self.fc3 = nn.Linear(config['a_fc2'], 4)
        
        self.actv = nn.LeakyReLU()
        self.output_actv = nn.Tanh()

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.bn1(self.actv(self.fc1(x)))
        x = self.bn2(self.actv(self.fc2(x)))
        x = self.output_actv(self.fc3(x))
        return x


class DDPGCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(33 * config['history_len'], config['c_fc1'])
        self.bn1 = nn.BatchNorm1d(config['c_fc1'])
        self.fc2 = nn.Linear(config['c_fc1'] + 4, config['c_fc2'])
        self.bn2 = nn.BatchNorm1d(config['c_fc2'])
        self.fc3 = nn.Linear(config['c_fc2'], 1)

        self.actv = nn.LeakyReLU()

    def forward(self, s, a):
        x = s.view(s.shape[0], -1)
        x = self.bn1(self.actv(self.fc1(x)))
        x = torch.cat((x, a), dim=1)
        x = self.bn2(self.actv(self.fc2(x)))
        x = self.fc3(x)
        return x
