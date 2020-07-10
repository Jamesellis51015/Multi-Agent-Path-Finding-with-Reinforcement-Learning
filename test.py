import torch
from torch import nn
import numpy as np
import os
from utils.wrappers import flat_np_lst
from collections import namedtuple
import copy
Transition = namedtuple('Transition', ('h1', 'h2'))
import math

from sklearn.model_selection import ParameterGrid
if __name__ == "__main__":

    def get_events_paths(base_folder, remove_duplicates = True):
        '''Return dict of {exp_name: event_file_path} '''
        experiments = {}
        tfeventstr = "events.out.tfevents"
        for root, dirs, file in os.walk(base_folder, topdown=False):
            for f in file:
                if tfeventstr in f:
                    name = root.split('/')[-1]
                    if remove_duplicates and 'N' in name:
                        hldr = name.split('_')[-1]
                        num = int(hldr[1:])
                        if num > 0:
                            continue
                    experiments[name] = os.path.join(root, f)
        return experiments

    filename = '/home/james/Desktop/Gridworld/test'
    exp = get_events_paths(filename, True)
    for k,v in exp.items():
        print("key: {} \n val:{} \n\n".format(k, v))
    # pos = {
    #     0: [(0,0), (1,1), (2,2)],
    #     1: [(3,3)],
    #     2: [(4,4)],
    #     3: [(5,5), (6,6)]
    # }

    # cominations = ParameterGrid(pos)

    # for c in cominations:
    #     print(( c[0], c[1], c[2], c[3] ) )

    # a = [1,"a"]
    # b = [2, "b"]
    # p = list(zip(a,b))
    # (c, d) = p
    # print(d)

    # for i in zip(a,b):
    #     print(i)
    # for i in a:
    #     print(i)

    # a.add(((1,2),(3,5)))
    # print(a)
    # a = set([1,2,])
    # b = set([1,5,3])
    # print(a==b)




    # a = torch.rand(10,3)
    # ind = np.arange(10)
    # np.random.shuffle(ind)
    # print(ind[0:3])
    # ind = torch.from_numpy(ind)
    # print(a)
    # print(a[ind[0:3]])
    # a = torch.tensor(5)
    # b = torch.clamp(a, -1,4)
    # print(b
    # # )
    # class myclass():
    #     def __init__(self, flag1):
    #         self.flag1 = flag1

    #         if flag1:
    #             def is_true(self):
    #                 print("true")
    #         else:
    #             def is_false(self):
    #                 print("flase")
    # a = myclass(True)
    # print(hasattr(a, "is_true"))
    # print(hasattr(a, "is_false"))rs

    # test = {1:1, 2:2}

    # class Tester():
    #     def __init__(self, a):
    #         self.aref = a
    # c = Tester(test)
    # print(c.aref[1])
    # test[1] = "dgsdg"
    # print(c.aref[1])
    

   
    # c = {a : 1,
    #     b: 2}
    # print(c[(-1,-2)])


    # f = '/home/desktop123/Documents/Academics/Code/Multiagent_Gridworld/TEST_walk'

    # a = [i for i in os.walk(f)]

    # print(a)


    
    
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
   

