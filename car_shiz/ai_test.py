import numpy as np
import random 
import os
import torch

#to use torch.nn
import torch.nn
import torch.nn.functional as F
import torch.optim as optim

#import variable class to make some conversions to tensors for tensor+gradient
import torch.autograd as autograd
from torch.autograd import Variable


#create architecture of neural network
class Network(torch.nn.Module):#inherit

    def __init__(self,input_size,nb_action) : #input==5 ---- output==3(left right straight)
        super(Network,self).__init__()

        self.input_size=input_size #input layer

        self.nb_action=nb_action #output layer
        
        #making connections
        self.fc1=torch.nn.Linear(input_size, 30)     #how many units 2nd argument // in tf no of units
        self.fc2=torch.nn.Linear(30,nb_action)
    
    
    def forward(self,state):   #state is inputs to nn  ##forward propogation
        #returns q value for each action
        x= F.relu(self.fc1(state))
        q_values=self.fc2(x)
        
        return q_values


#implement experience replay
# markov decision process
# memory of last 100 states


class ReplayMemory(object):
    def __init__(self,capacity) : #
        self.capacity=capacity
        self.memory = []

    def push(self,event):
        #keeps only 100 states in memory
        self.memory.append(event)

        if len(self.memory)>self.capacity:
            del self.memory[0]
    
    def sample(self,batch_size):
        samples=zip(*random.sample(self.memory,batch_size))
        return map(lambda x: Variable(torch.cat(x,0)),random.samples)


#implementing deep q learning








