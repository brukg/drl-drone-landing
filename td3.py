import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''
 Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
 Paper: https://arxiv.org/abs/1802.09477
'''

class Actor(nn.Module):

    def __init__(self, state_dim, action_dim, max_action, nhid):

        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, nhid)
        self.l2 = nn.Linear(nhid, nhid)
        self.l3 = nn.Linear(nhid, action_dim)
        self.max_action = max_action

    def forward(self, state):

        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):

    def __init__(self, state_dim, action_dim, nhid):

        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, nhid)
        self.l2 = nn.Linear(nhid, nhid)
        self.l3 = nn.Linear(nhid, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, nhid)
        self.l5 = nn.Linear(nhid, nhid)
        self.l6 = nn.Linear(nhid, 1)

    def forward(self, state, action):

        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):

        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class TD3:

    def __init__(
           self,
           state_dim,
           action_dim,
           max_action,
           nhid,
           discount=0.99,
           tau=0.005,
           policy_noise=0.2,
           noise_clip=0.5,
           policy_freq=2):

        self.actor = Actor(state_dim, action_dim, max_action, nhid).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=3e-4)

        self.critic = Critic(state_dim, action_dim, nhid).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=3e-4)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=100):
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward, not_done = (
            replay_buffer.sample(batch_size))

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_action = (
                self.actor_target(next_state) + noise
            ).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = (F.mse_loss(current_Q1, target_Q) +
                       F.mse_loss(current_Q2, target_Q))

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor losse
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(),
                                           self.critic_target.parameters()):
                target_param.data.copy_(self.tau *
                                        param.data + (1 - self.tau) *
                                        target_param.data)

            for param, target_param in zip(self.actor.parameters(),
                                           self.actor_target.parameters()):
                target_param.data.copy_(self.tau *
                                        param.data + (1 - self.tau) *
                                        target_param.data)

    def get(self):

        return (self.critic.state_dict(),
                self.critic_optimizer.state_dict(),
                self.actor.state_dict(),
                self.actor_optimizer.state_dict())

    def set(self, parts):

        self.critic.load_state_dict(parts[0])
        self.critic_optimizer.load_state_dict(parts[1])
        self.actor.load_state_dict(parts[2])
        self.actor_optimizer.load_state_dict(parts[3])

        self.critic_target = copy.deepcopy(self.critic)
        self.actor_target = copy.deepcopy(self.actor)


def eval_policy(policy, env, eval_episodes=10, render=False, is_bullet=False, dump=False):
    '''
    Runs policy for X episodes and returns average reward
    '''

    total_reward = 0.
    total_steps = 0

    # Start rendering thread for PyBullet if needed
    if is_bullet:
        env.render()

    for _ in range(eval_episodes):

        state, done = env.reset(), False

        while not done:

            action = policy.select_action(np.array(state))

            if dump:
                print(action)

            state, reward, done, _ = env.step(action)

            if render:
                if not is_bullet:
                    env.render()
                time.sleep(.02)

            total_reward += reward
            total_steps += 1

    return total_reward / eval_episodes, total_steps//eval_episodes


class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device('cuda'
                                   if torch.cuda.is_available()
                                   else 'cpu')

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
                torch.FloatTensor(self.state[ind]).to(self.device),
                torch.FloatTensor(self.action[ind]).to(self.device),
                torch.FloatTensor(self.next_state[ind]).to(self.device),
                torch.FloatTensor(self.reward[ind]).to(self.device),
                torch.FloatTensor(self.not_done[ind]).to(self.device)
                )
