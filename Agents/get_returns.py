"""Functions for calculating returns """
import numpy as np
import torch as torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gym import spaces
import config

from collections import namedtuple

Transition = namedtuple('Transition', ('state', 'action_taken', 'action_prob', 'value', 'episode_mask', 'episode_mini_mask', 'next_state',
                                       'reward', 'misc'))

def mc_return(episode_mask, episode_mini_mask, actions_taken, value, reward, discount):
    """Calculates the montecarlo return for a particular agent"""
    size = len(reward)
    G = np.zeros(shape = (size,), dtype=float)
    for i, hldr in enumerate(reversed([a for a in zip(reward, value, actions_taken, episode_mini_mask, episode_mask)])):
        #NB: episode_mini_mask ignored ...episode does not finish until all agents finish
        if i == 0:
            if hldr[-1] == 0:
                G[size - i - 1] = hldr[0] #Reward in terminal state
            else:
                G[size - i - 1] = hldr[1] #The value [expected return] of being in that state (don't know the value of the next state)
        else:
            if hldr[-1]==1:
                G[size - i -1] = hldr[0] + discount*G[size - i]
            else:
                G[size - i -1] = hldr[0]       
    return torch.tensor([G]).double()

def general_advantage_estimate(episode_mask, episode_mini_mask, actions_taken, value, reward, discount, lambda_):
    size = len(reward)
    #Calculate deltas
    delta = np.zeros(shape = (size,), dtype=float)
    for i, hldr in enumerate(reversed([a for a in zip(reward, value, actions_taken, episode_mini_mask, episode_mask)])):
        #NB: episode_mini_mask ignored ...episode does not finish until all agents finish
        if i == 0:
            if hldr[-1] == 0: #If the episode step is terminal
                delta[size - i - 1] = hldr[0] #Reward in terminal state
            else:
                delta[size - i - 1] = hldr[1] #The value [expected return] of being in that state (don't know the value of the next state)
        else:
            if hldr[-1]==1: #if the episode step is not terminal
                delta[size - i -1] = hldr[0] + discount*value[size - i] - hldr[1]
            else:
                delta[size - i -1] = hldr[0]
    #Calculate advantages:
    adv = np.zeros(shape = (size,), dtype=float)  
    for i, d in enumerate(reversed([k for k in zip(delta, episode_mask)])):
        if i ==0 or d[-1] == 0:
            adv[size - i -1] = d[0]
        else:
            adv[size - i -1] = d[0] + discount * lambda_ * adv[size - i]     
    return torch.tensor([adv]).double()

def one_step_td_return(episode_mask, value, reward, discount):
    size = len(reward)
    #Calculate deltas
    delta = np.zeros(shape = (size,), dtype=float)
    for i, hldr in enumerate(reversed([a for a in zip(reward, value, episode_mask)])):
        #NB: episode_mini_mask ignored ...episode does not finish until all agents finish
        if i == 0:
            if hldr[-1] == 0: #If the episode step is terminal
                delta[size - i - 1] = hldr[0] #Reward in terminal state
            else:
                delta[size - i - 1] = hldr[1] #The value [expected return] of being in that state (don't know the value of the next state)
        else:
            if hldr[-1]==1: #if the episode step is not terminal
                delta[size - i -1] = hldr[0] + discount*value[size - i] - hldr[1]
            else:
                delta[size - i -1] = hldr[0]
        
    return torch.tensor([delta]).double()
    
