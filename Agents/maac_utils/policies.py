'''
Modified from https://github.com/shariqiqbal2810/MAAC 
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from Agents.maac_utils.misc import onehot_from_logits, categorical_sample
from Agents.general_utils.policy import make_base_policy

def make_policy(policy_type):
    BasePolicy = make_base_policy(policy_type)

    class ActorBase(BasePolicy):
        def __init__(self, *args, nonlin = F.leaky_relu, **kwargs):
            (num_in_pol, num_out_pol) = args
            self.nonlin = nonlin
            del kwargs["onehot_dim"]
            super(ActorBase, self).__init__(num_in_pol,nonlin=nonlin, **kwargs)
            h_dim = kwargs["hidden_dim"]
            self.start_layer = nn.Linear(h_dim, h_dim)
            self.out_layer = nn.Linear(h_dim, num_out_pol)
        def forward(self, obs):
            out1 = super(ActorBase, self).forward(obs.float())
            out2 = self.out_layer(out1)
            return out2

    class ActorPolicy(ActorBase):
        def __init__(self, *args, **kwargs):
            super(ActorPolicy, self).__init__(*args, **kwargs)

        def forward(self, obs, sample=True, return_all_probs=False,
                    return_log_pi=False, regularize=False,
                    return_entropy=False):
            out = super(ActorPolicy, self).forward(obs)
            probs = F.softmax(out, dim=1)
            on_gpu = next(self.parameters()).is_cuda
            if sample:
                int_act, act = categorical_sample(probs, use_cuda=on_gpu)
            else:
                act = onehot_from_logits(probs)
            rets = [act]
            if return_log_pi or return_entropy:
                log_probs = F.log_softmax(out, dim=1)
            if return_all_probs:
                rets.append(probs)
            if return_log_pi:
                # return log probability of selected action
                rets.append(log_probs.gather(1, int_act))
            if regularize:
                rets.append([(out**2).mean()])
            if return_entropy:
                rets.append(-(log_probs * probs).sum(1).mean())
            if len(rets) == 1:
                return rets[0]
            return rets

    return ActorPolicy