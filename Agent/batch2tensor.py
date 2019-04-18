import torch
import numpy as np


def batch2tensor(batch):
    action_batch = np.vstack(batch[1])
    state = torch.Tensor(batch[0]).cuda()
    action = torch.Tensor(action_batch).cuda()
    reward = torch.Tensor(batch[2]).cuda()
    next_state = torch.Tensor(batch[3]).cuda()
    done = torch.Tensor(batch[4]).cuda()
    return state, action, reward, next_state, done
