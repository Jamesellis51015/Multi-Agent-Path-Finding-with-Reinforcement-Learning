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

def myf1():
      parser = argparse.ArgumentParser("abc")
      parser.add_argument("--a", default = 4, type=int)
      parser.add_argument("--b", default = 4, type=int)
      parser.add_argument("--c", default = 4, type=int)

      args2, unkn = parser.parse_known_args()
      print(args2.a)
      print(args2.b)

from sklearn.model_selection import ParameterGrid
if __name__ == "__main__":

   import argparse
   parser = argparse.ArgumentParser("Generate Data")

    #
   parser.add_argument("--m", default = 5, type=int)
   parser.add_argument("--n", default = 4, type=int)

   args1, unkn = parser.parse_known_args()
   print(args1)
   args2, unk = parser.parse_known_args(["--m", "22", "--n", "99"])
   print(args2)
   #print(vars(args1))
   # print(args1)
   # print(unkn)

   # if args1.m == 1:
   #    myf1()
   # else:
   #    print("none")

   




   # class A():
   #    def __init__(self, b):
   #       self.b = b
   #    @property
   #    def c(self):
   #       return "abc"
      
   # hldr = A(1)
   # print(hldr.c)
   # h2 = A(3)
   
   # e1 = hldr
   # e2 = hldr

   # e2.b = 99

   # print(e1.b)

   # d = {"a":1, "b":2}

   # if 1 in d:
   #    print("yes")
   # hldr2 = []
   # hldr2.append(((1,2), (3,4)))
   # hldr2.append(((1,2), (2,4)))
   # hldr2.append(((3,4), (2,2), (1,1)))
   # hldr2.append(((3,4), (1,1), (1,1)))
   # hldr2.append(((3,4), (1,2), (1,1),(3,4), (1,2), (1,1)))
   # hldr2.append(((3,4), (1,2), (1,1),(5,4), (5,2), (5,1)))
   # hldr2.append(((3,4), (1,2), (1,1),(5,1), (5,2), (5,1)))

   # def is_colliding(v):
   #    hldr = set()
   #    for i, vi in enumerate(v):
   #       for i2, vi2 in enumerate(v):
   #          if i != i2:
   #                if vi == vi2:
   #                   hldr.add(i)
   #                   hldr.add(i2)
   #    return hldr 

   # for h in hldr2:
   #    print("V is {}      Col: {}".format(h, is_colliding(h)))
    #  print(is_colliding(h))
   # print(is_colliding(a))
   # print(is_colliding(b))

   # class Vertex():

   #    def __init__(self, v):
   #       MAXCOST = 1e6
   #       assert type(v) == tuple, "Input is not tuple"
   #       self.v = v
   #       self.cost_g = MAXCOST
   #       self._collision_set = set()
   #       self._back_set = {}
   #       self.is_colliding = self._is_colliding(v)
   #       self.joint_collision_set(self.is_colliding)
        
   #    def add_back_set(self, new_v):
   #       assert isinstance(new_v, self), "Input is not a Vertex class"

   #       if new_v.v in self._back_set:
   #             print("WARNING: overwriting vertex which already exist in _back_set")
   #       self._back_set[new_v.v] = new_v
      
   #    def add_collision(self, agent_handle):
   #       self._collision_set.add(agent_handle)
   #    def joint_collision_set(self, other_set):
   #       self._collision_set.union(other_set)
   #    def is_col_subset(self, other_set):
   #       return other_set.issubset(self._collision_set)
   #    def get_collision_set(self):
   #       return self._collision_set
   #    def set_cost(self, new_cost):
   #       self.cost_g = new_cost
   #    @property
   #    def get_cost(self):
   #       return self.cost_g

   #    def _is_colliding(self, v):
   #       hldr = set()
   #       for i, vi in enumerate(v):
   #             for i2, vi2 in enumerate(v):
   #                if i != i2:
   #                   if vi == vi2:
   #                         hldr.add(i)
   #                         hldr.add(i2)
   #       return hldr 
   #    #For priority que:
   #    def __eq__(self, other_v):
   #       return self.get_cost == other_v.get_cost
   #    def __gt__(self, other_v):
   #       return self.get_cost > other_v.get_cost
   #    def __ge__(self, other_v):
   #       return self.get_cost >= other_v.get_cost
   #    def __lt__(self, other_v):
   #       return self.get_cost < other_v.get_cost
   #    def __le__(self, other_v):
   #       return self.get_cost <= other_v.get_cost

   # from queue import PriorityQueue

   # class newP_Q(PriorityQueue):
   #    def __init__(self):
   #       super().__init__()
   #       self.lookup_table = set()
   #    def put(self, item):
   #       super().put(item)
   #       self.lookup_table.add(item[-1].v)
   #    def get(self):
   #       result = super().get()
   #       self.lookup_table.remove(result[-1].v)
   #       return result
   #    def __contains__(self, key):
   #       return key in self.lookup_table

   # b = newP_Q()
   # v1 = Vertex(((1,2), (3,4)))
   # v2 = Vertex(((1,2), (3,7)))
   # v3 = Vertex(((1,2), (3,6)))
   # b.put((1,v1))
   # b.put((2,v2))
   # b.put((2,v3))
   # print("lookup: {}".format(b.lookup_table))
   # h1,h2 = b.get()
   # print(h1,h2.v)
   # print("lookup: {}".format(b.lookup_table))
   # h1,h2 = b.get()
   # print(h1,h2.v)

   # print(((1,2), (3,4)) in b)
   # print(((1,2), (3,6)) in b)

   # a = {1,2}
   # b= {3}
   # a.union(b)
   # print(a)
   # c ={6,7}
   # c = c.union(b)
   # print(c)
   # j = {1,3,5,7}
   # for i in j: print(i)

   # a = [1,2,3,4]
   # print(a[:-1])
   # from queue import PriorityQueue
   # a = PriorityQueue()
   # a.put((1, "a"))
   # a.put((2, "b"))
   # a = np.arange(5)
   # b = np.random.choice(a, 5, replace = False)

   # print(b)

   
   # while True:
   #    val = input("Enter your value: ") 
   #    print(val) 
   # b = (2, "b")

   # if b in :
   #    print(a)
   # # a = {1,2,3}
   # b = {2,3}
   
   # print(b.issubset(a))
   #a.add(b)
   #print(a)

   # print(a==b)

   # c = set()
   # c.add(a)
   # d =  set()
   # d.add(b)
   # print(c.)


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
   

