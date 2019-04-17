import torch


def batch2tensor(batch):
    state = torch.Tensor(batch[0]).cuda()
    action = torch.Tensor(batch[1]).cuda().long()
    reward = torch.Tensor(batch[2]).cuda()
    next_state = torch.Tensor(batch[3]).cuda()
    done = torch.Tensor(batch[4]).cuda()
    return state, action, reward, next_state, done
