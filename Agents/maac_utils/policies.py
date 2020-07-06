import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from Agents.maac_utils.misc import onehot_from_logits, categorical_sample
from Agents.general_utils.policy import make_base_policy


# class BasePolicy(nn.Module):
#     """
#     Base policy network
#     """
#     def __init__(self, input_dim, out_dim, hidden_dim=64, nonlin=F.leaky_relu,
#                  norm_in=True, onehot_dim=0):
#         """
#         Inputs:
#             input_dim (int): Number of dimensions in input
#             out_dim (int): Number of dimensions in output
#             hidden_dim (int): Number of hidden dimensions
#             nonlin (PyTorch function): Nonlinearity to apply to hidden layers
#         """
#         super(BasePolicy, self).__init__()

#         if norm_in:  # normalize inputs
#             self.in_fn = nn.BatchNorm1d(input_dim, affine=False)
#         else:
#             self.in_fn = lambda x: x
#         self.fc1 = nn.Linear(input_dim + onehot_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, hidden_dim)
#         self.fc3 = nn.Linear(hidden_dim, out_dim)
#         self.nonlin = nonlin

#     def forward(self, X):
#         """
#         Inputs:
#             X (PyTorch Matrix): Batch of observations (optionally a tuple that
#                                 additionally includes a onehot label)
#         Outputs:
#             out (PyTorch Matrix): Actions
#         """
#         onehot = None
#         if type(X) is tuple:
#             X, onehot = X
#         inp = self.in_fn(X.float())  # don't batchnorm onehot
#         if onehot is not None:
#             inp = torch.cat((onehot, inp), dim=1)
#         h1 = self.nonlin(self.fc1(inp))
#         h2 = self.nonlin(self.fc2(h1))
#         out = self.fc3(h2)
#         return out

def make_policy(policy_type):
    BasePolicy = make_base_policy(policy_type)

    class ActorBase(BasePolicy):
        def __init__(self, *args, nonlin = F.leaky_relu, **kwargs):
            (num_in_pol, num_out_pol) = args
            self.nonlin = nonlin
            del kwargs["onehot_dim"]
            super(ActorBase, self).__init__(num_in_pol, **kwargs)
            h_dim = kwargs["hidden_dim"]
            self.start_layer = nn.Linear(h_dim, h_dim)
            self.out_layer = nn.Linear(h_dim, num_out_pol)
        def forward(self, obs):
            out1 = self.nonlin(super(ActorBase, self).forward(obs.float()))
            out2 = self.out_layer(out1)
            return out2

    class ActorPolicy(ActorBase):
        """
        Policy Network for discrete action spaces
        """
        def __init__(self, *args, **kwargs):
            super(ActorPolicy, self).__init__(*args, **kwargs)

        def forward(self, obs, sample=True, return_all_probs=False,
                    return_log_pi=False, regularize=False,
                    return_entropy=False):
            out = super(ActorPolicy, self).forward(obs)
            probs = F.softmax(out, dim=1)
            on_gpu = next(self.parameters()).is_cuda
            if sample:
                # print("sample")
                int_act, act = categorical_sample(probs, use_cuda=on_gpu)
                # act_shape = list(act.numpy().shape)
                # act_shape = act_shape[:-1]
                # act = torch.from_numpy(np.argmax(act.numpy(), axis = -1).reshape(act_shape))
                # print("act is {}".format(act))
            else:
                #print("not sample")
                act = onehot_from_logits(probs)
                # act_shape = list(act.numpy().shape)
                # act_shape = act_shape[:-1]
                # act = torch.from_numpy(np.argmax(act.numpy(), axis = -1).reshape(act_shape))
                #act = torch.from_numpy(np.argmax(act.numpy())) #*
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