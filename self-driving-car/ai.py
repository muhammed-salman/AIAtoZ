#the brain of self driving car (AI)

#importing libraries
import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

#architecture of the neural network

class Network(nn.Module):
    
    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        
        #all the neuron of i/p layer is connected to all the neuron of hidden layer (full connec)
        self.fc1 = nn.Linear(input_size, 30)
        self.fc2 = nn.Linear(30,nb_action) #hidden to o/p layer
        
        
    #activates neuron, returns Q-value for each possible action    
    def forward(self, state):
        x = F.relu(self.fc1(state)) #rectifier function (activates hidden neurons)
        q_values = self.fc2(x)
        return q_values
    
#implementing Experience replay

class ReplayMemory(object):
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        
    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]
    
    def sample(self, batch_size):
        #zip function: - reshape the list ex ((state1,action1,reward1),(state2,action2,reward2)....) will become  ((state1,state2,..) (action1,action2,..), (reward1,reward2,..), ...)
        samples = zip(*random.sample(self.memory, batch_size))
        return map(lambda x: Variable(torch.cat(x,0)), samples)
    

#Implementing Deep Q-Learning 

class Dqn():
    
    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma
        self.reward_window = []
        self.model = Network(input_size, nb_action)
        self.memory = ReplayMemory(100000)
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001)
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.last_action = 0
        self.last_reward = 0
        
    def select_action(self, state):
        #self.model is o/p of NN, state is a tensor Variable() converts it into torch var with gradient volatile=True includes gradient assoc with i/p states to the graph of all the conditions of nn module
        # volatile=True won't be running backpropagation in order to conserve memory.
        probs = F.softmax(self.model(Variable(state, volatile = True))*100) #Temperature (T)=7
       # action = probs.multinomial()
       # return action.data[0,0]
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        return action
    
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        target = self.gamma * next_outputs + batch_reward
        td_loss = F.smooth_l1_loss(outputs, target) 
        self.optimizer.zero_grad() 
        td_loss.backward(retain_graph = True) #backpropogation
        self.optimizer.step() #updates weights 
        
    def update(self, reward, new_signal):
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
        action = self.select_action(new_state)
        if len(self.memory.memory) > 100:
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        return action
        
    def score(self):
        return sum(self.reward_window)/(len(self.reward_window)+1) #+1 to avoid div by zero err
    
    def save(self):
        torch.save({'state_dict': self.model.state_dict(), 
                    'optimizer': self.optimizer.state_dict(),
                    }, 'last_brain.pth')
    
    def load(self):
        #current working dir os.path
        if os.path.isfile('last_brain.pth'):
            print('=> loading checkpoint...')
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("Done!..")
        else:
            print("no checkpoint found... (file not found)")
        
        