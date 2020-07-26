from Agents.PPO.utils import make_actor_net, make_value_net
import numpy as np
import torch as torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gym import spaces
MSE_Loss = torch.nn.MSELoss()
CrossEntropyLoss = nn.CrossEntropyLoss()
BCELoss = nn.BCELoss(reduction = 'none')

class PPO():
    def __init__(self, n_actions, obs_space, policy_type, n_agents, share_actor, share_value, 
                    K_epochs, minibatch_len, lr_a = 0.001,lr_v = 0.001,
                    hidden_dim = 120, eps_clip = 0.2, entropy_coeff = 0.0, recurrent = False,
                    heur_block = False):
        # obs_space = obs_space_shape
        self.recurrent = recurrent
        self.hidden_dim = hidden_dim
        self.heur_block = heur_block
        if type(obs_space) != tuple:
            obs_space = obs_space.shape
            double_obs_flag = False
        else:
            double_obs_flag = True
        self.share_actor = share_actor
        self.share_value = share_value
        if share_actor:
            Ac = make_actor_net(policy_type, double_obs_flag, recurrent, heur_block)
            ac = Ac(obs_space, hidden_dim, n_actions, lr = lr_a)
            self.actors = [ac for _ in range(n_agents)]
        else:
            Ac = make_actor_net(policy_type,double_obs_flag, recurrent, heur_block)
            self.actors = [Ac(obs_space, hidden_dim, n_actions, lr = lr_a)\
                for _ in range(n_agents)]
        if share_value:
            Cr = make_value_net(policy_type,double_obs_flag, recurrent)
            cr = Cr(obs_space, hidden_dim, lr = lr_v)
            self.critics = [cr for _ in range(n_agents)]
        else:
            Cr = make_value_net(policy_type,double_obs_flag, recurrent)
            self.critics = [Cr(obs_space, hidden_dim, lr = lr_v) \
                for _ in range(n_agents)]

        self.entropy_coeff = entropy_coeff
        self.K_epochs = K_epochs
        self.eps_clip = eps_clip
        self.minibatch_len = minibatch_len
        self.current_device = 'cpu'

        self.ac_pol_old = Ac(obs_space, hidden_dim, n_actions, lr = lr_a)
        self.cr_pol_old = Cr(obs_space, hidden_dim, lr = lr_a)

        #for saving
        self.params = [n_actions, obs_space, policy_type, n_agents, share_actor, share_value, 
                    K_epochs, minibatch_len, lr_a, lr_v,
                    hidden_dim, eps_clip, entropy_coeff, recurrent, heur_block]
    
    def init_hx_cx(self, dev):
        if self.recurrent:
            hc = (self.tens_to_dev(dev, torch.zeros((1,self.hidden_dim), requires_grad=True)), \
                self.tens_to_dev(dev, torch.zeros((1,self.hidden_dim), requires_grad=True)))
        else:
            hc = None
        return hc
    def extend_agent_indexes(self, nagents):
        assert (self.share_actor == True and self.share_value==True), \
            "Parameters are not shared!"
        hldr = self.actors[0]
        self.actors = [hldr for i in range(nagents)]
        hldr = self.critics[0]
        self.critics = [hldr for i in range(nagents)]

    def forward(self, obs, hid_cell_actr = None, hid_cell_cr = None, greedy = False, dev = 'cpu', valid_act_heur = None):
        '''Forwards a dictionary of observations where each key value corresponds to a agent '''
        if self.current_device != dev:
            print("Changing device for PPO policy")
            self.prep_device(dev)
        a_probs = {}
        a_select = {}
        value = {}
        hx_cx_actr = {}
        hx_cx_cr = {}
        blocking = {}
        for i, ob in obs.items():
            if self.recurrent:
                h_c_acr =( self.tens_to_dev(dev, hid_cell_actr[i][0]), self.tens_to_dev(dev, hid_cell_actr[i][1]))
                h_c_cr = ( self.tens_to_dev(dev, hid_cell_cr[i][0]), self.tens_to_dev(dev, hid_cell_cr[i][1]))
            else:
                h_c_acr = None
                h_c_cr = None
            if type(ob) == tuple:
                ob1 = torch.from_numpy(ob[0]).float().unsqueeze(0)
                ob2 = torch.from_numpy(ob[1]).float().unsqueeze(0)
                ob = (self.tens_to_dev(dev, ob1), self.tens_to_dev(dev, ob2))
            else:
                ob = self.tens_to_dev(dev, torch.from_numpy(ob).float().unsqueeze(0))
            if not valid_act_heur is None:
                if not valid_act_heur[i] is None:
                    valid_act = valid_act_heur[i]
                else:
                    valid_act = None
            else:
                valid_act = None
            a_probs[i], a_select[i], hx_cx_actr[i], blocking[i] = self.actors[i].take_action(ob, greedy = greedy, \
                                                                hid_cell_state=h_c_acr, valid_act = valid_act)
            value[i], hx_cx_cr[i] = self.critics[i].forward(ob, hid_cell_state = h_c_cr)
        return (a_probs, a_select, value, hx_cx_actr, hx_cx_cr, blocking)


    def update(self, obs, a_prob, a_select, adv, v, agent_id, hx_cx_actr=None, hx_cx_cr=None, blk_labels = None, blk_pred = None, dev=None):
        '''Updates a specific agents policy with that agent's tensor batches '''
        self.ac_pol_old.load_state_dict(self.actors[agent_id].state_dict())
        self.cr_pol_old.load_state_dict(self.critics[agent_id].state_dict())
        ac_pol = self.actors[agent_id]
        cr_pol = self.critics[agent_id]

        if not dev is None:
            self.prep_device(dev)
            if type(obs) == tuple:
                ob1 = self.tens_to_dev(dev, obs[0])
                ob2 = self.tens_to_dev(dev, obs[1])
                obs = (ob1, ob2)
            else:
                obs = self.tens_to_dev(dev, obs)
            a_prob = self.tens_to_dev(dev, a_prob)
            a_select = self.tens_to_dev(dev, a_select)
            adv = self.tens_to_dev(dev, adv)
            v = self.tens_to_dev(dev, v)
            if self.recurrent:
                (hx, cx) = hx_cx_actr
                hx_cx_actr = (self.tens_to_dev(dev, hx), self.tens_to_dev(dev, cx))
                (hx, cx) = hx_cx_cr
                hx_cx_cr = (self.tens_to_dev(dev, hx), self.tens_to_dev(dev, cx))
            if not blk_labels is None and not blk_pred is None:
                blk_labels = self.tens_to_dev(dev, blk_labels)
                blk_pred = self.tens_to_dev(dev, blk_pred)
            

        nbatch = a_prob.size(0)
        assert nbatch % self.minibatch_len == 0
        a_prob_select_old = torch.gather(a_prob, dim= -1, index = a_select)
        if not blk_pred is None:
            blk_old = blk_pred
        total_loss = []
        ###########
    #     ind = np.arange(nbatch)
    #     np.random.shuffle(ind)
    #     if type(obs) == tuple:
    #         ob1 = obs[0][ind]
    #         ob2 = obs[1][ind]
    #         mb_ob = (ob1, ob2)
    #     else:
    #         mb_ob = obs[ind]
    #     mb_as = a_select[ind]
    #     mb_aps = a_prob_select_old[ind]
    #     mb_adv = adv[ind]
    #     if self.recurrent:
    #         mb_hx = hx_cx_actr[0][ind]
    #         mb_cx = hx_cx_actr[1][ind]
    #         mb_hxcx = (mb_hx, mb_cx)
    #     else:
    #         mb_hxcx = None
    #     entropy = torch.distributions.Categorical(mb_aps).entropy() * self.entropy_coeff
    #     loss = -torch.log(mb_aps) * mb_adv
    #     loss = loss.mean()
    #     loss += entropy.mean()

    #     if not blk_labels is None:
    #         blk_loss = MSE_Loss(blk_pred, blk_labels)*10
    #         loss += blk_loss

    #     ac_pol.optimizer.zero_grad()
    #     loss.backward()
    #     ac_pol.optimizer.step()
    #     total_loss.append(loss.item())

    #     if self.recurrent:
    #         v_hx = hx_cx_cr[0] #[ind]
    #         v_cx = hx_cx_cr[1] #[ind]
    #         v_hxcx = (v_hx, v_cx)
    #     else:
    #         v_hxcx = None
    #     #####
    #   #  obs = obs[ind]
    #     #####
    #     #v, _ = cr_pol.forward(obs, v_hxcx)
    #     val_loss = MSE_Loss(v, adv + v.detach())
    #     cr_pol.optimizer.zero_grad()
    #     val_loss.backward()
    #     cr_pol.optimizer.step()
        ############################################################
        for K in range(self.K_epochs):
            ind = np.arange(nbatch)
            np.random.shuffle(ind)
            for i in range(0, nbatch, self.minibatch_len):
                mb_ind = ind[i:i+self.minibatch_len]
                if type(obs) == tuple:
                    ob1 = obs[0][mb_ind]
                    ob2 = obs[1][mb_ind]
                    mb_ob = (ob1, ob2)
                else:
                    mb_ob = obs[mb_ind]
                mb_as = a_select[mb_ind]
                mb_aps = a_prob_select_old[mb_ind]
                mb_adv = adv[mb_ind]
                if self.recurrent:
                    mb_hx = hx_cx_actr[0][mb_ind]
                    mb_cx = hx_cx_actr[1][mb_ind]
                    mb_hxcx = (mb_hx.detach(), mb_cx.detach())
                else:
                    mb_hxcx = None
                (a_prob_new, _,_,blk_pred_new) = ac_pol.forward(mb_ob, mb_hxcx)
                if not blk_pred is None:
                    n_elements = blk_pred_new.numel() / 2
                    mb_blk_old = blk_old[mb_ind]
                    mb_blk_lbl = blk_labels[mb_ind].flatten().long()
                    blk_loss = CrossEntropyLoss(blk_pred_new.detach(), mb_blk_lbl)
                   #blk_loss = MSE_Loss(blk_pred_new.detach(), mb_blk_lbl)
                    rat_blk = blk_pred_new / mb_blk_old.detach()
                    blk_sur1 = rat_blk * blk_loss
                    blk_sur2 = torch.clamp(rat_blk, 1-self.eps_clip, 1+self.eps_clip) * blk_loss
                    blk_loss2 = torch.min(blk_sur1, blk_sur2).mean()

                entropy = torch.distributions.Categorical(a_prob_new).entropy() * self.entropy_coeff
                a_prob_select_new = torch.gather(a_prob_new, dim= -1, index = mb_as)
                ratio = a_prob_select_new / mb_aps.detach()
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * mb_adv
                loss = -torch.min(surr1, surr2).mean() - entropy.mean()
                if not blk_pred is None:
                    loss += blk_loss2
                ac_pol.optimizer.zero_grad()
                loss.backward(retain_graph = True)
                grad_norm = torch.nn.utils.clip_grad_norm(ac_pol.parameters(), 0.5)
                ac_pol.optimizer.step()
                total_loss.append(loss.item())
        
        if self.recurrent:
            v_hx = hx_cx_cr[0] #[ind]
            v_cx = hx_cx_cr[1] #[ind]
            v_hxcx = (v_hx, v_cx)
        else:
            v_hxcx = None
        #####
      #  obs = obs[ind]
        #####
        v, _ = cr_pol.forward(obs, v_hxcx)
        val_loss = MSE_Loss(v, adv + v.detach())
        cr_pol.optimizer.zero_grad()
        val_loss.backward()
        cr_pol.optimizer.step()

        # #train blocking pred:
        # if not blk_labels is None:
        #     blk_loss = MSE_Loss(blk_pred, blk_labels)*10
        #     ac_pol.optimizer.zero_grad()
        #     blk_loss.backward(retain_graph = True)
        #     ac_pol.optimizer.step()

        ####################################################################

        self.prep_device('cpu')
        return sum(total_loss), val_loss.item()
    
    def prep_device(self, dev):
        if self.current_device == dev:
            return
        self.current_device = dev
        if dev == 'gpu':
            dev = torch.device('cuda')
        else:
            dev = torch.device('cpu')

        if self.share_actor:
            self.actors[0].to(dev)
            self.ac_pol_old.to(dev)
        else:
            for a, aold in zip(self.actors, self.ac_pol_old):
                a.to(dev)
                aold.to(dev)

        if self.share_value:
            self.critics[0].to(dev)
            self.cr_pol_old.to(dev)
        else:
            for c, cold in zip(self.critics, self.cr_pol_old):
                c.to(dev)
                cold.to(dev)

    def tens_to_dev(self, dev, tens):
        if dev == 'gpu':
            fn = lambda x: x.cuda()
        else:
            fn = lambda x: x.cpu()
        return fn(tens)

    def load(self, model_info):
        params = model_info["params"]
        params[3] = self.params[3] #Number of agents may change
        critic_state_dict = model_info["critic_state_dict"]
        actor_state_dict = model_info["actor_state_dict"]
        self.__init__(*params)
        self.actors[0].load_state_dict(actor_state_dict) #assuming parameter sharing
        self.critics[0].load_state_dict(critic_state_dict)

    def get_model_dict(self):
        model_info = {"params": self.params,
        "critic_state_dict": self.critics[0].state_dict(),
        "actor_state_dict": self.actors[0].state_dict()}
        return model_info
    
    def save(self, path):
        model_inf = self.get_model_dict()
        torch.save(model_inf, path)


        
        





                







              
