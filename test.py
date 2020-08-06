import torch
from torch import nn
import numpy as np
import os
from utils.wrappers import flat_np_lst
from collections import namedtuple
import copy
Transition = namedtuple('Transition', ('h1', 'h2'))
import math

from time import sleep


import torch.nn as nn
import torch.nn.functional as F


from sklearn.model_selection import ParameterGrid
if __name__ == "__main__":
   for i in range(2,8):
      print(i)
   # for i in range(5): #wait 5h before execution
   #    print("waiting... {}".format(i))
   #    sleep(60*60)

   #  class PRIMAL_Base(nn.Module):
   #      def __init__(self, input_dim, hidden_dim=None, nonlin = None):
   #          self.hidden_dim = hidden_dim
   #          self.nonlin = nonlin
   #          super().__init__()
   #          (channels, d1, d2) =  input_dim
   #          k = 3
   #          p=1

   #          self.c1 = nn.Conv2d(channels, 128, k, padding=0)
   #          d = d1 #assume square image
   #          d = (d+2*p - (k-1) - 1)/1 + 1
   #          self.c2 = nn.Conv2d(128, 128, k, padding=p)
   #          self.c3 = nn.Conv2d(128, 128, k, padding=p)
   #       #   self.mp1 = nn.MaxPool2d(2,2)
   #          d = (d-(2-1) - 1)/2 + 1
   #          self.c4 = nn.Conv2d(channels, 256, k, padding=p)
   #          self.c5 = nn.Conv2d(channels, 256, k, padding=p)
   #          self.c6 = nn.Conv2d(channels, 128, k, padding=p)
   #       #   self.mp2 = nn.MaxPool2d(3,2)
   #          k=2
   #          self.c7 = nn.Conv2d(channels, 500, k)

   #          # self.model = nn.Sequential(
   #          #     self.c1,
   #          #     nn.ReLU(),
   #          #     self.c2,
   #          #     nn.ReLU(),
   #          #     self.c3,
   #          #     nn.ReLU(),
   #          #     self.mp1,
   #          #     self.c4,
   #          #     nn.ReLU(),
   #          #     self.c5,
   #          #     nn.ReLU(),
   #          #     self.c6,
   #          #     nn.ReLU(),
   #          #     self.mp2,
   #          # )

   #          #self.flat_cnn_out_size = 500
   #          #self.fc1 = nn.Linear(self.flat_cnn_out_size, hidden_dim)
    
   #      def forward(self, x):
   #          batch_size = x.size(0)
   #          #x = self.model(x)

   #          x1 = self.c1(x)
   #          x = F.relu(x[0])
   #          x = self.c2(x),
   #          x = F.relu(x)
   #          x = self.c3(x),
   #          x = F.relu(x)
   #          x = self.mp1(x),
   #       #   x = F.relu(x)
   #          x = self.c4(x),
   #          x = F.relu(x)
   #          x = self.c5(x),
   #          x = F.relu(x)
   #          x = self.c6(x),
   #          x = F.relu(x)
   #         x = self.mp2(x),




   #          x = x.reshape((batch_size,-1))
   #          return x

   #  m = PRIMAL_Base((5,10,10))
   #  t = torch.rand((1,5,10,10))
   #  b = m.forward(t)
    
    
    # class t():
    #     def __init__(self, a):
    #         self.a =a
    #     @property
    #     def g(self):
    #         return self.a
    # k = t(5)
    # print(k.g)
    # a = {0: np.array([[1,2],[3,4]]),
    # 1: np.array([[5,6],[7,8]])}

    # b= flat_np_lst(a, flat=False)
    # c = torch.from_numpy(b)
    # print(b)
    # print(b.shape)
    # print(c.size())

    # h1 = [1,2,3]
    # h2 = [4,5,5]
    # h3 = [33,44,55]
    # h4 = [88,99,10]
    # k = Transition(h1, h2)
    # b = [k, Transition(h3, h4)]

    # hldr = Transition(*zip(*b))
    # print(hldr)



   # a = torch.tensor([[1],])
    # print("abc")
    # for i in range(5, -1, -1):
    #     print(i)
   

