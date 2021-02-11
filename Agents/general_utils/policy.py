import numpy as np
import torch as torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gym import spaces


def make_base_policy(policy_type, double_obs_space = False):
    if double_obs_space:
        if policy_type == "mlp":
            base_policy = Mlp_Base2
        elif policy_type == "primal7":
            base_policy = PRIMAL_Base7
        elif policy_type == "primal9":
            base_policy = PRIMAL_Base9
        else:
            raise Exception("Base policy type not implemented")
        return base_policy
    else:
        if policy_type == "mlp":
            base_policy = Mlp_Base
        elif policy_type == "cnn_old":
            base_policy = CNN_Base_Old
        elif policy_type == "cnn_new":
            base_policy = CNN_Base_New
        elif policy_type == "primal1":
            base_policy = PRIMAL_Base
        elif policy_type == "primal2":
            base_policy = PRIMAL_Base2
        elif policy_type == "primal2_2":
            base_policy = PRIMAL_Base2_2
        elif policy_type == "primal3":
            base_policy = PRIMAL_Base3
        elif policy_type == "primal4":
            base_policy = PRIMAL_Base4
        elif policy_type == "primal5":
            base_policy = PRIMAL_Base5
        elif policy_type == "primal6":
            base_policy = PRIMAL_Base6
        elif policy_type == "primal7":
            base_policy = PRIMAL_Base7
        elif policy_type == "primal9":
            base_policy = PRIMAL_Base9
        else:
            raise Exception("Base policy type not implemented")
        return base_policy


class CNN_Base_Old(nn.Module):
    def __init__(self, input_dim, hidden_dim= 120, nonlin = F.leaky_relu):
        super().__init__()
        self.nonlin = nonlin
        self.hidden_dim = hidden_dim
        #For strides of 1:
        (channels, d1, d2) =  input_dim
        k1 = 2
        self.c1 = nn.Conv2d(channels, 3*channels, k1)
        d1c1 = d1- (k1 - 1) 
        d2c1 = d2 -(k1 - 1)

        k2 = k1
        self.c2 = nn.Conv2d(3*channels, 2*channels, k2)
        d1c2 = d1c1- (k2 - 1) 
        d2c2 = d2c1 -(k2 - 1)
        self.flat_cnn_out_size = 2*channels*(d1c2)*(d2c2)
        self.fc1 = nn.Linear(self.flat_cnn_out_size, hidden_dim)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.c1(x)
        x = self.nonlin(x)
        x = self.c2(x)
        x = self.nonlin(x)
        x = x.reshape((batch_size,-1))
        x = self.fc1(x)
        x = self.nonlin(x)
        return x

class CNN_Base_New(nn.Module):
    def __init__(self, input_dim, hidden_dim= 120, nonlin = F.leaky_relu):
        super().__init__()
        self.nonlin = nonlin
        (channels, d1, d2) =  input_dim
        k1 = 3
        self.c1 = nn.Conv2d(channels, 6*channels, k1)
        d1c1 = d1- (k1 - 1) 
        d2c1 = d2 -(k1 - 1)
        k2 = k1
        self.c2 = nn.Conv2d(6*channels, 4*channels, k2)
        d1c2 = d1c1- (k2 - 1) 
        d2c2 = d2c1 -(k2 - 1)
        self.flat_cnn_out_size = 4*channels*(d1c2)*(d2c2)
        self.fc1 = nn.Linear(self.flat_cnn_out_size, hidden_dim)
    
    def forward(self, x):
        batch_size = x.size(0)
        x = self.c1(x)
        x = self.nonlin(x)
        x = self.c2(x)
        x = self.nonlin(x)
        x = x.reshape((batch_size,-1))
        x = self.fc1(x)
        x = self.nonlin(x)
        return x

class Mlp_Base(nn.Module):
    def __init__(self, input_dim, hidden_dim= 120, nonlin = F.leaky_relu):
        super().__init__()
        self.nonlin = nonlin
        obs_shape = input_dim
        flat_dim = 1
        for i in obs_shape:
            flat_dim *= i
        self.fc1 = nn.Linear(flat_dim, hidden_dim)
        self.hidden_dim = hidden_dim
    def forward(self, x):
        batch_size = x.size()[0]
        x = self.fc1(x.reshape(batch_size, -1))
        x = self.nonlin(x)
        return x

class Mlp_Base2(nn.Module):
    def __init__(self, input_dim1_shape,input_dim2, hidden_dim= 120, nonlin = F.leaky_relu):
        super().__init__()
        self.nonlin = nonlin
        obs_shape = input_dim1_shape
        flat_dim = 1
        for i in obs_shape:
            flat_dim *= i
        self.fc1 = nn.Linear(flat_dim, hidden_dim)
        self.fc1_2 = nn.Linear(input_dim2, hidden_dim)
        self.hidden_dim = hidden_dim*2
    def forward(self, x):
        (x1,x2) = x
        batch_size = x1.size()[0]
        x1 = self.fc1(x1.reshape(batch_size, -1))
        x1 = self.nonlin(x1)
        x2 = self.fc1_2(x2.reshape(batch_size, -1))
        x2 = self.nonlin(x2)
        x3 = torch.cat([x1,x2], dim=-1)
        return x3


class PRIMAL_Base(nn.Module):
    def __init__(self, input_dim, hidden_dim=None, nonlin = None):
        self.hidden_dim = hidden_dim
        self.nonlin = nonlin
        super().__init__()
        (channels, d1, d2) =  input_dim
        k = 3
        p=1
        self.c1 = nn.Conv2d(channels, 128, k, padding=p)
        d = d1 #assume square image
        d = (d+2*p - (k-1) - 1)/1 + 1
        self.c2 = nn.Conv2d(128, 128, k, padding=p)
        self.c3 = nn.Conv2d(128, 128, k, padding=p)
        self.mp1 = nn.MaxPool2d(2,2)
        d = (d-(2-1) - 1)/2 + 1
        self.c4 = nn.Conv2d(128, 256, k, padding=p)
        self.c5 = nn.Conv2d(256, 256, k, padding=p)
        self.c6 = nn.Conv2d(256, 128, k, padding=p)
        self.mp2 = nn.MaxPool2d(3,2)
        k=2
        self.c7 = nn.Conv2d(128, 500, k)


        self.flat_cnn_out_size = 500
        self.fc1 = nn.Linear(self.flat_cnn_out_size, hidden_dim)
    
    def forward(self, x):
        batch_size = x.size(0)

        x = self.c1(x)
        x = F.relu(x)
        x = self.c2(x)
        x = F.relu(x)
        x = self.c3(x)
        x = F.relu(x)
        x = self.mp1(x)
        x = self.c4(x)
        x = F.relu(x)
        x = self.c5(x)
        x = F.relu(x)
        x = self.c6(x)
        x = F.relu(x)
        x = self.mp2(x)
        x = self.c7(x)
        x = F.relu(x)

        x = x.reshape((batch_size,-1))
        x = self.fc1(x)
        x = F.relu(x)
        return x

class PRIMAL_Base2(nn.Module):
    def __init__(self, input_dim, hidden_dim=None, nonlin = None):
        self.hidden_dim = hidden_dim
        self.nonlin = nonlin
        super().__init__()
        (channels, d1, d2) =  input_dim
        k = 3
        p=1
        self.c1 = nn.Conv2d(channels, 128, k, padding=p)
        d = d1 #assume square image
        d = (d+2*p - (k-1) - 1)/1 + 1
        self.c2 = nn.Conv2d(128, 128, k, padding=p)
        self.c3 = nn.Conv2d(128, 128, k, padding=p)
        self.mp1 = nn.MaxPool2d(3,1)
        d = (d-(2-1) - 1)/2 + 1
        self.c4 = nn.Conv2d(128, 256, k, padding=p)
        self.c5 = nn.Conv2d(256, 256, k, padding=p)
        self.c6 = nn.Conv2d(256, 128, k, padding=p)
        self.mp2 = nn.MaxPool2d(3,2)
        k=2
        self.c7 = nn.Conv2d(128, 500, k)

        self.flat_cnn_out_size = 500
        self.fc1 = nn.Linear(self.flat_cnn_out_size, hidden_dim)
    
    def forward(self, x):
        batch_size = x.size(0)
        #x = self.model(x)

        x = self.c1(x)
        x = F.relu(x)
        x = self.c2(x)
        x = F.relu(x)
        x = self.c3(x)
        x = F.relu(x)
        x = self.mp1(x)
        x = self.c4(x)
        x = F.relu(x)
        x = self.c5(x)
        x = F.relu(x)
        x = self.c6(x)
        x = F.relu(x)
        x = self.mp2(x)
        x = self.c7(x)
        x = F.relu(x)

        x = x.reshape((batch_size,-1))
        x = self.fc1(x)
        x = F.relu(x)
        return x

class PRIMAL_Base2_2(nn.Module):
    def __init__(self, input_dim, hidden_dim=None, nonlin = None):
        self.hidden_dim = hidden_dim
        self.nonlin = nonlin
        super().__init__()
        (channels, d1, d2) =  input_dim
        k = 3
        p=1
        self.c1 = nn.Conv2d(channels, 128, k, padding=p)
        d = d1 #assume square image
        d = (d+2*p - (k-1) - 1)/1 + 1
        self.c2 = nn.Conv2d(128, 128, k, padding=p)
        self.c3 = nn.Conv2d(128, 128, k, padding=p)
        self.mp1 = nn.MaxPool2d(3,1)
        d = (d-(2-1) - 1)/2 + 1
        self.c4 = nn.Conv2d(128, 256, k, padding=p)
        self.c5 = nn.Conv2d(256, 256, k, padding=p)
        self.c6 = nn.Conv2d(256, 128, k, padding=p)
        self.mp2 = nn.MaxPool2d(3,2)
        k=1
        self.c7 = nn.Conv2d(128, 500, k)

        self.flat_cnn_out_size = 500
        self.fc1 = nn.Linear(self.flat_cnn_out_size, hidden_dim)
    
    def forward(self, x):
        batch_size = x.size(0)
        #x = self.model(x)

        x = self.c1(x)
        x = F.relu(x)
        x = self.c2(x)
        x = F.relu(x)
        x = self.c3(x)
        x = F.relu(x)
        x = self.mp1(x)
        x = self.c4(x)
        x = F.relu(x)
        x = self.c5(x)
        x = F.relu(x)
        x = self.c6(x)
        x = F.relu(x)
        x = self.mp2(x)
        x = self.c7(x)
        x = F.relu(x)

        x = x.reshape((batch_size,-1))
        x = self.fc1(x)
        x = F.relu(x)
        return x

class PRIMAL_Base3(nn.Module):
    def __init__(self, input_dim, hidden_dim=None, nonlin = None):
        self.hidden_dim = hidden_dim
        self.nonlin = nonlin
        super().__init__()
        (channels, d1, d2) =  input_dim
        k = 3
        p=1
        self.c1 = nn.Conv2d(channels, 128, k, padding=p)
        d = d1 #assume square image
        d = (d+2*p - (k-1) - 1)/1 + 1
        self.c2 = nn.Conv2d(128, 128, k, padding=p)
        self.c3 = nn.Conv2d(128, 128, k, padding=p)
        self.mp1 = nn.MaxPool2d(3,1)
        d = (d-(2-1) - 1)/2 + 1
        self.c4 = nn.Conv2d(128, 256, k, padding=p)
        self.c5 = nn.Conv2d(256, 256, 2, padding=0)
        self.c6 = nn.Conv2d(256, 128, 2, padding=0)
        #self.mp2 = nn.MaxPool2d(3,2)
        #k=2
        self.c7 = nn.Conv2d(128, 500, 3)

        self.flat_cnn_out_size = 500
        self.fc1 = nn.Linear(self.flat_cnn_out_size, hidden_dim)
    
    def forward(self, x):
        batch_size = x.size(0)
        #x = self.model(x)

        x = self.c1(x)
        x = F.relu(x)
        x = self.c2(x)
        x = F.relu(x)
        x = self.c3(x)
        x = F.relu(x)
        x = self.mp1(x)
        x = self.c4(x)
        x = F.relu(x)
        x = self.c5(x)
        x = F.relu(x)
        x = self.c6(x)
        x = F.relu(x)
       # x = self.mp2(x)
        x = self.c7(x)
        x = F.relu(x)

        x = x.reshape((batch_size,-1))
        x = self.fc1(x)
        x = F.relu(x)
        return x


class PRIMAL_Base4(nn.Module):
    def __init__(self, input_dim, hidden_dim=None, nonlin = None):
        self.hidden_dim = hidden_dim
        self.nonlin = nonlin
        super().__init__()
        (channels, d1, d2) =  input_dim
        k = 3
        p=1
        self.c1 = nn.Conv2d(channels, 128, k, padding=p)
        d = d1 #assume square image
        d = (d+2*p - (k-1) - 1)/1 + 1
        self.c2 = nn.Conv2d(128, 128, k, padding=p)
        self.c3 = nn.Conv2d(128, 128, k, padding=p)
        self.mp1 = nn.MaxPool2d(3,1)
        d = (d-(2-1) - 1)/2 + 1
        self.c4 = nn.Conv2d(128, 256, k, padding=p)
        self.c5 = nn.Conv2d(256, 256, k, padding=p)
        self.c6 = nn.Conv2d(256, 128, k, padding=p)
        self.mp2 = nn.MaxPool2d(3,1)
        k=3
        self.c7 = nn.Conv2d(128, 500, k)

        self.flat_cnn_out_size = 500
        self.fc1 = nn.Linear(self.flat_cnn_out_size, hidden_dim)
    
    def forward(self, x):
        batch_size = x.size(0)
        #x = self.model(x)

        x = self.c1(x)
        x = F.relu(x)
        x = self.c2(x)
        x = F.relu(x)
        x = self.c3(x)
        x = F.relu(x)
        x = self.mp1(x)
        x = self.c4(x)
        x = F.relu(x)
        x = self.c5(x)
        x = F.relu(x)
        x = self.c6(x)
        x = F.relu(x)
        x = self.mp2(x)
        x = self.c7(x)
        x = F.relu(x)

        x = x.reshape((batch_size,-1))
        x = self.fc1(x)
        x = F.relu(x)
        return x

class PRIMAL_Base5(nn.Module):
    def __init__(self, input_dim, hidden_dim=None, nonlin = None):
        self.hidden_dim = hidden_dim
        self.nonlin = nonlin
        super().__init__()
        (channels, d1, d2) =  input_dim
        k = 3
        p=1
        self.c1 = nn.Conv2d(channels, 128, k, padding=p)
        d = d1 #assume square image
        d = (d+2*p - (k-1) - 1)/1 + 1
        self.c2 = nn.Conv2d(128, 128, k, padding=p)
        self.c3 = nn.Conv2d(128, 128, k, padding=p)
        self.mp1 = nn.MaxPool2d(3,1)
        d = (d-(2-1) - 1)/2 + 1
        self.c4 = nn.Conv2d(128, 256, k, padding=p)
        self.c5 = nn.Conv2d(256, 256, k, padding=p)
        self.c6 = nn.Conv2d(256, 128, k, padding=p)
        self.mp2 = nn.MaxPool2d(3,2)
        k=2
        self.c7 = nn.Conv2d(128, self.hidden_dim, k)

        self.flat_cnn_out_size = self.hidden_dim
        self.fc1 = nn.Linear(self.flat_cnn_out_size, hidden_dim)
    
    def forward(self, x):
        batch_size = x.size(0)
        #x = self.model(x)

        x = self.c1(x)
        x = F.relu(x)
        x = self.c2(x)
        x = F.relu(x)
        x = self.c3(x)
        x = F.relu(x)
        x = self.mp1(x)
        x = self.c4(x)
        x = F.relu(x)
        x = self.c5(x)
        x = F.relu(x)
        x = self.c6(x)
        x = F.relu(x)
        x = self.mp2(x)
        x = self.c7(x)
        x = F.relu(x)

        x = x.reshape((batch_size,-1))
        x = self.fc1(x)
        x = F.relu(x)
        return x

class PRIMAL_Base6(nn.Module): #For env size  = 5
    def __init__(self, input_dim, hidden_dim=None, nonlin = None, cat_end = None):
        self.hidden_dim = hidden_dim
        self.nonlin = nonlin
        super().__init__()
        (channels, d1, d2) =  input_dim
        k = 2
        p=0
        self.c1 = nn.Conv2d(channels, 128, k, padding=p)
        d = d1 #assume square image
        d = (d+2*p - (k-1) - 1)/1 + 1
        self.c2 = nn.Conv2d(128, 128, k, padding=p)
        self.c3 = nn.Conv2d(128, 128, k, padding=p)
        k=2
        self.c7 = nn.Conv2d(128, self.hidden_dim, k)

        self.flat_cnn_out_size = self.hidden_dim
        if cat_end is None:
            self.fc1 = nn.Linear(self.flat_cnn_out_size, hidden_dim)
        else:
            self.fc1 = nn.Linear(self.flat_cnn_out_size + cat_end, hidden_dim)
    
    def forward(self, x, cat_end = None):

        if type(x) == tuple:
            (x, cat_end) = x
        batch_size = x.size(0)
        x = self.c1(x)
        x = F.relu(x)
        x = self.c2(x)
        x = F.relu(x)
        x = self.c3(x)
        x = F.relu(x)
        x = self.c7(x)
        x = F.relu(x)
        x = x.reshape((batch_size,-1))
        if cat_end is None:
            x = self.fc1(x)
        else:
            x = self.fc1(torch.cat([x,cat_end], dim=1))
        x = F.relu(x)
        return x

class PRIMAL_Base7(nn.Module): #For env size  = 7
    def __init__(self, input_dim, hidden_dim=None, nonlin = None, cat_end = None):
        self.hidden_dim = hidden_dim
        self.nonlin = nonlin
        super().__init__()
        (channels, d1, d2) =  input_dim
        k = 2
        p=0
        self.c1 = nn.Conv2d(channels, 128, k, padding=p)
        d = d1 #assume square image
        d = (d+2*p - (k-1) - 1)/1 + 1
        self.c2 = nn.Conv2d(128, 128, k, padding=p)
        self.c3 = nn.Conv2d(128, 128, k, padding=p)
        self.c8 = nn.Conv2d(128, 128, k, padding=p)
        k=3
        self.c7 = nn.Conv2d(128, self.hidden_dim, k)

        self.flat_cnn_out_size = self.hidden_dim

        self.cat_end_mid = None
        if cat_end is None:
            self.fc1 = nn.Linear(self.flat_cnn_out_size, hidden_dim)
        else:
            self.fc1 = nn.Linear(self.flat_cnn_out_size + cat_end, hidden_dim)
    
    def forward(self, x, cat_end = None):
        if type(x) == tuple:
            (x, cat_end) = x
        batch_size = x.size(0)
        x = self.c1(x)
        x = F.relu(x)
        x = self.c2(x)
        x = F.relu(x)
        x = self.c3(x)
        x = F.relu(x)
        x = self.c8(x)
        x = F.relu(x)
        x = self.c7(x)
        x = F.relu(x)
        x = x.reshape((batch_size,-1))
        if cat_end is None:
            x = self.fc1(x)
        else:
            x = self.fc1(torch.cat([x,cat_end], dim=1))
        x = F.relu(x)
        return x



class PRIMAL_Base9(nn.Module): #For env size  = 9
    def __init__(self, input_dim, hidden_dim=None, nonlin = None, cat_end = None):
        self.hidden_dim = hidden_dim
        self.nonlin = nonlin
        super().__init__()
        (channels, d1, d2) =  input_dim
        k = 3
        p=1
        self.c1 = nn.Conv2d(channels, 128, k, padding=p)
        d = d1 #assume square image
        d = (d+2*p - (k-1) - 1)/1 + 1
        self.c2 = nn.Conv2d(128, 128, k, padding=p)
        self.c3 = nn.Conv2d(128, 128, k, padding=p)
        self.mp1 = nn.MaxPool2d(2,2, ceil_mode=True)
        self.c4 = nn.Conv2d(128, 256, k, padding=p)
        self.c5 = nn.Conv2d(256, 256, k, padding=p)
        self.c6 = nn.Conv2d(256, 256, k, padding=p)
        self.mp2 = nn.MaxPool2d(2,2)

        if cat_end is not None:
            self.c7 = nn.Conv2d(256, self.hidden_dim - 12, 2)
        else:
            self.c7 = nn.Conv2d(256, self.hidden_dim, 2)

        self.flat_cnn_out_size = self.hidden_dim

        self.cat_end_mid = None
        if cat_end is None:
            pass
            #self.fc1 = nn.Linear(self.flat_cnn_out_size, hidden_dim)
        else:
            #self.cat_end_mid = nn.Linear(cat_end)
            #self.fc1 = nn.Linear(self.flat_cnn_out_size + cat_end, hidden_dim)
            self.cat_end_mid = nn.Linear(cat_end, 12)
        #self.fc1 = nn.Linear(self.flat_cnn_out_size, hidden_dim)
    
    def forward(self, x, cat_end = None):
        if type(x) == tuple:
            (x, cat_end) = x
        batch_size = x.size(0)
        x = self.c1(x)
        x = F.relu(x)
        x = self.c2(x)
        x = F.relu(x)
        x = self.c3(x)
        x = self.mp1(x)
        x = F.relu(x)
        x = self.c4(x)
        x = F.relu(x)
        x = self.c5(x)
        x = F.relu(x)
        x = self.c6(x)
        x = F.relu(x)
        x = self.mp2(x)
        x = self.c7(x)
        x = F.relu(x.reshape(batch_size, -1))
        if cat_end is None:
            pass
        else:
            cat_end = F.relu(self.cat_end_mid(cat_end))
            x = torch.cat([x,cat_end], dim=1)
        return x