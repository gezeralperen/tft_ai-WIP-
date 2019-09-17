import numpy as np
import torch.nn as nn
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torch.nn.functional as F
import torch.optim as optim
import json
import os
from time import time
import random


class ReplayGetter(object):
    def __init__(self, memory_size):
        self.memory_size = memory_size
        self.memory = []

    def get_last_data(self):
        files = os.listdir('replay_experience/')
        all_data = []
        if len(files) > self.memory_size:
            files = files[self.memory_size:]
        if len(files) > 20:
            files = random.choices(files, k=20)
        for file in files:
            # print('Getting ' + file)
            f = open('replay_experience/' + file, 'r')
            try:
                all_data.extend(json.load(f))
                f.close()
            except json.decoder.JSONDecodeError:
                try:
                    f.close()
                    os.remove('replay_experience/' + file)
                    # print(file + ' is not included to train data and deleted.')
                except PermissionError:
                    # print(file + ' is not included to train data but couldn\'t be deleted.')
                    f.close()
        self.memory = all_data

    def get_last_stats(self):
        files = os.listdir('replay_experience/')
        all_data = []
        if len(files) > self.memory_size:
            files = files[self.memory_size:]
        if len(files) > 20:
            files = random.choices(files, k=20)
        for file in files:
            # print('Getting ' + file)
            f = open('replay_experience/' + file, 'r')
            try:
                all_data.extend(json.load(f))
                f.close()
            except json.decoder.JSONDecodeError:
                try:
                    f.close()
                    os.remove('replay_experience/' + file)
                    # print(file + ' is not included to train data and deleted.')
                except PermissionError:
                    # print(file + ' is not included to train data but couldn\'t be deleted.')
                    f.close()
        self.memory = all_data

    def get_sample(self,batch_size):
        ind = np.random.randint(0, len(self.memory), size=batch_size)
        batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = [],[],[],[],[]
        for i in ind:
            state = np.array(self.memory[i]['State'])
            state = np.reshape(state, (1, 32, 10))
            batch_states.append(state)
            next_state = np.array(self.memory[i]['Next State'])
            next_state = np.reshape(next_state, (1, 32, 10))
            batch_next_states.append(next_state)
            batch_actions.append(self.memory[i]['Action'])
            batch_rewards.append(self.memory[i]['Q'])
            batch_dones.append(self.memory[i]['Done'])
        return np.array(batch_states), np.array(batch_next_states), np.array(batch_actions), np.array(batch_rewards).reshape(-1,1), np.array(batch_dones).reshape(-1,1)

    def get_sample_cnn(self,batch_size):
        ind = np.random.randint(0, len(self.memory), size=batch_size)
        batch_states = []
        for i in ind:
            state = np.array(self.memory[i]['State'])[0]
            batch_states.append(state)
        return np.array(batch_states)



class StatsEncoder(torch.nn.Module):
    def __init__(self):
        super(StatsEncoder, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 128, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = torch.nn.Linear(64 * 32 * 10, 512)
        self.fc2 = torch.nn.Linear(512, 64)
        self.fc1t = torch.nn.Linear(64, 64 * 32 * 10)
        self.conv2t = torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv1t = torch.nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # Encode
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 64 * 32 * 10)
        x = F.relu(self.fc1(x))

        # Result
        x = self.fc2(x)

        # Decode
        x = F.relu(self.fc1t(x))
        x = x.view(-1,64,32,10)
        x = F.relu(self.conv2t(x))
        x = F.relu(self.conv1t(x))
        return x

    def encode(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, 64 * 32 * 10)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.layer1 = nn.Linear(state_dim, 400)
        self.layer2 = nn.Linear(400, 300)
        self.layer3 = nn.Linear(300, action_dim)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.tanh(self.layer3(x))
        return x

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.layer1 = nn.Linear(state_dim + action_dim, 400)
        self.layer2 = nn.Linear(400, 300)
        self.layer3 = nn.Linear(300, 1)
        # Critic 2
        self.layer4 = nn.Linear(state_dim + action_dim, 400)
        self.layer5 = nn.Linear(400, 300)
        self.layer6 = nn.Linear(300, 1)

    def forward(self, x, u):
        xu = torch.cat([x,u], 1)
        xu1 = F.relu(self.layer1(xu))
        xu1 = F.relu(self.layer2(xu1))
        xu1 = self.layer3(xu1)
        xu2 = F.relu(self.layer4(xu))
        xu2 = F.relu(self.layer5(xu2))
        xu2 = self.layer6(xu2)
        return xu1, xu2

    def Q1(self, x, u):
        xu = torch.cat([x,u], 1)
        xu1 = F.relu(self.layer1(xu))
        xu1 = F.relu(self.layer2(xu1))
        xu1 = self.layer3(xu1)
        return xu1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
replay = ReplayGetter(100000)

class TD3(object):
    def __init__(self, state_dim=64, action_dim=4):
        self.encoder = StatsEncoder().to(device)
        self.encoder.load_state_dict(torch.load(f'models/encoder.pth'))
        self.actor = Actor(state_dim,action_dim).to(device)
        self.actor_target = Actor(state_dim,action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters())

        self.critic = Critic(state_dim,action_dim).to(device)
        self.critic_target = Critic(state_dim,action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters())

    def select_action(self, state):
        state = torch.Tensor(state).to(device)
        state = self.encoder.encode(state).to(device).reshape((-1,64))
        action = self.actor(state).data.cpu().numpy().flatten()
        return action

    def train(self, iterations, batch_size = 100, discount=0.99, tau = 0.05, policy_noise=0.2,
              noise_clip=0.5, policy_freq = 2):
        replay.get_last_data()
        for it in range(iterations):
            total_loss = 0.
            tactor_loss = 0.
            sa = 0.
            s = 0.
            batch_states, batch_next_states, batch_actions, batch_rewards, _ = replay.get_sample(batch_size)
            state = torch.Tensor(batch_states).to(device)
            state = self.encoder.encode(state)
            next_state = torch.Tensor(batch_next_states).to(device)
            next_state = self.encoder.encode(next_state)
            action = torch.Tensor(batch_actions).to(device)
            next_action = self.actor(next_state)
            noise = torch.Tensor(batch_actions).data.normal_(0, policy_noise).to(device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = next_action + noise
            reward = torch.Tensor(batch_rewards).to(device)
            target_q1, target_q2 = self.critic_target(state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward * (1-tau) + tau * target_q
            current_q1, current_q2 = self.critic(state, action)

            critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
            total_loss += critic_loss
            s+=1.
            self.critic_optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            self.critic_optimizer.step()

            if it%policy_freq == 0:
                actor_loss = -self.critic.Q1(state,self.actor(state)).mean()
                tactor_loss += actor_loss
                sa += 1.
                print(f'Iteration : {it}/{iterations}\tCritic Loss:{total_loss/s:.6f}\tActor Loss:{tactor_loss/sa:.6f}')
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau*param.data + (1-tau) * target_param.data)

                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), f'{directory}/{filename}_actor.pth')
        torch.save(self.critic.state_dict(), f'{directory}/{filename}_critic.pth')

    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load(f'{directory}/{filename}_actor.pth'))
        self.critic.load_state_dict(torch.load(f'{directory}/{filename}_critic.pth'))


def TrainEncoder():
    encoder = StatsEncoder().cuda()
    optimizer = optim.Adam(encoder.parameters())
    for y in range(100):
        replay.get_last_stats()
        total_loss = 0
        for x in range(10000):
            batch_states = replay.get_sample_cnn(50)
            state = torch.Tensor(batch_states).to(device)
            output = encoder(state)
            loss = F.mse_loss(output, state)
            total_loss += loss
            print(f'Iteration : {x+y*10000}/{1000000}\t\tEncoder Loss:{total_loss/(x+1):.6f}')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        torch.save(encoder.state_dict(), f'models/encoder.pth')

