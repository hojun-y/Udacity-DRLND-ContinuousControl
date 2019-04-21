import torch
import torch.nn as nn
import torch.optim as optim
import Agent.memory_utils as memutils
import numpy as np
from Agent.batch2tensor import batch2tensor


class DDQNAgent:
    def __init__(self, actor, critic, config_dict):
        self._replay_memory = memutils.ReplayMemory(config_dict['replay_size'])
        self._config_dict = config_dict
        self._actor = actor().to('cuda')
        self._target_actor = actor().to('cuda')
        self._critic = critic().to('cuda')
        self._target_critic = critic().to('cuda')

        self._loss = nn.MSELoss()
        self._actor_optimizer = optim.Adam(
            self._actor.parameters(), config_dict['actor_lr'],
            betas=config_dict['betas'], weight_decay=config_dict['actor_l2'])
        self._critic_optimizer = optim.Adam(
            self._critic.parameters(), config_dict['critic_lr'],
            betas=config_dict['betas'], weight_decay=config_dict['critic_l2'])

        self.sync_network()
        self._target_actor.eval()
        self._target_critic.eval()

    def _get_loss(self, state, action, reward, next_state, done):
        target_action = self._target_actor(next_state)
        target_q = self._target_critic(next_state, target_action)
        q_value = self._critic(state, action)
        target_y = reward + (self._config_dict['discount_factor'] * target_q)
        loss = self._loss(q_value, target_y)
        return loss

    def _optimize(self, loss):
        self._actor_optimizer.zero_grad()
        self._critic_optimizer.zero_grad()
        loss.backward()

        """
        if self._config_dict['gradient_clip']:
            for actor_var, critic_var in zip(self._actor.parameters(), self._critic.parameters()):
                torch.clamp_(actor_var.grad.data, -self._config_dict['clip_value'], self._config_dict['clip_value'])
                torch.clamp_(critic_var.grad.data, -self._config_dict['clip_value'], self._config_dict['clip_value'])
        """

        self._actor_optimizer.step()
        self._critic_optimizer.step()

    def append_memory(self, state, action, reward, next_state, done):
        self._replay_memory.append(state, action, reward, next_state, done)

    def sync_network(self):
        self._target_actor.load_state_dict(self._actor.state_dict())
        self._target_critic.load_state_dict(self._critic.state_dict())

    def soft_update(self, tau):
        def update(params, target_params):
            for var, target_var in zip(params, target_params):
                target_var.data = tau * var + (1 - tau) * target_var

        update(self._actor.parameters(), self._target_actor.parameters())
        update(self._critic.parameters(), self._target_critic.parameters())

    def get_action(self, state):
        with torch.no_grad():
            self._actor.eval()
            action = self._actor.forward(torch.Tensor([state]).cuda())
            action = action.cpu().numpy()
            add_noise = action + np.random.normal(0, self._config_dict['noise_size'], 4)

            return add_noise

    def train(self):
        batch = self._replay_memory.get_batch(self._config_dict['batch_size'])
        state, action, reward, next_state, done = batch2tensor(batch)
        loss = self._get_loss(state, action, reward, next_state, done)
        self._optimize(loss)
        return loss

    def save_weights(self, path):
        torch.save(self._actor.state_dict(), path + 'actor.weights')
        torch.save(self._critic.state_dict(), path + 'critic.weights')

    def restore_weights(self, path):
        actor_state_dict = torch.load(path + 'actor.weights')
        critic_state_dict = torch.load(path + 'critic.weights')
        self._actor.load_state_dict(actor_state_dict)
        self._critic.load_state_dict(critic_state_dict)
        self.sync_network()
