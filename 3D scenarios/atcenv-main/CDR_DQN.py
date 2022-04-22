import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
    def __init__(self, N_STATES, N_HIDDEN, N_ACTIONS):
        super(Net, self).__init__()
        self.N_STATES = N_STATES
        self.N_HIDDEN = N_HIDDEN
        self.N_ACTIONS = N_ACTIONS
        self.fc1 = nn.Linear(self.N_STATES, self.N_HIDDEN)
        self.fc1.weight.data.normal_(0, 0.1)  # initialization
        self.fc2 = nn.Linear(self.N_HIDDEN, self.N_HIDDEN)
        self.fc2.weight.data.normal_(0, 0.1)  # initialization
        self.out = nn.Linear(self.N_HIDDEN, self.N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)  # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = f.relu(x)
        actions_value = self.out(x)
        return actions_value


class DQN(object):
    def __init__(self, BATCH_SIZE, LEARNING_RATE, EPSILON, GAMMA, TARGET_REPLACE_ITER, MEMORY_CAPACITY, N_HIDDEN,
                 N_ACTIONS, N_STATES, ENV_A_SHAPE, EPISODE, SHOW_ITER, TRAINED_NN=None):
        self.BATCH_SIZE = BATCH_SIZE
        self.LEARNING_RATE = LEARNING_RATE
        self.EPSILON = EPSILON
        self.GAMMA = GAMMA
        self.TARGET_REPLACE_ITER = TARGET_REPLACE_ITER
        self.MEMORY_CAPACITY = MEMORY_CAPACITY
        self.N_HIDDEN = N_HIDDEN
        self.N_ACTIONS = N_ACTIONS
        self.N_STATES = N_STATES
        self.ENV_A_SHAPE = ENV_A_SHAPE
        self.EPISODE = EPISODE
        self.SHOW_ITER = SHOW_ITER
        if TRAINED_NN is None:
            self.eval_net, self.target_net = Net(self.N_STATES, self.N_HIDDEN, self.N_ACTIONS), \
                                             Net(self.N_STATES, self.N_HIDDEN, self.N_ACTIONS)
        else:
            self.eval_net, self.target_net = torch.load(f'./nn/{TRAINED_NN}'), torch.load(f'./nn/{TRAINED_NN}')
        self.eval_net.to(device)
        self.target_net.to(device)
        self.learn_step_counter = 0  # for target updating
        self.memory_counter = 0  # for storing memory
        self.memory = np.zeros((self.MEMORY_CAPACITY, self.N_STATES * 2 + 2))  # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.LEARNING_RATE)
        self.loss_func = nn.MSELoss()
        self.loss_func.to(device)

    def choose_action(self, x, episode):
        x = torch.unsqueeze(torch.FloatTensor(x.flatten()), 0).to(device)
        if episode % self.SHOW_ITER == 0:
            greedy = 1
        else:
            greedy = self.EPSILON + (1 - self.EPSILON) * (episode + 1) / self.EPISODE
        if np.random.uniform() < greedy:
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].cpu().data.numpy()
            action = action[0] if self.ENV_A_SHAPE == 0 else action.reshape(self.ENV_A_SHAPE)  # return the argmax index
        else:  # random
            action = np.random.randint(0, self.N_ACTIONS)
            action = action if self.ENV_A_SHAPE == 0 else action.reshape(self.ENV_A_SHAPE)
        return action

    def store_transition(self, s, a, r, s_):
        s = s.flatten()
        s_ = s_.flatten()
        transition = np.hstack((s, a, r, s_))
        # replace the old memory with new memory
        index = self.memory_counter % self.MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target parameter update
        if self.learn_step_counter % self.TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(self.MEMORY_CAPACITY, self.BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :self.N_STATES]).to(device)
        b_a = torch.LongTensor(b_memory[:, self.N_STATES:self.N_STATES+1]).to(device)
        b_r = torch.FloatTensor(b_memory[:, self.N_STATES+1:self.N_STATES+2]).to(device)
        b_s_ = torch.FloatTensor(b_memory[:, -self.N_STATES:]).to(device)

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape(batch, 1)
        q_next = self.target_net(b_s_).detach()
        q_target = b_r + self.GAMMA * q_next.max(1)[0].view(self.BATCH_SIZE, 1)  # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
