import numpy as np
import torch as torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from Agents.PPO.utils import advantages

class PPO_Buffer():
    def __init__(self, nagents, nworkers, nrollouts, recurrent = False):
        self.model = None
        self.nagents = nagents
        self.nworkers = nworkers
        self.nrollouts = nrollouts
        self.nbatch = nworkers * nrollouts
        self.recurrent = recurrent

        reduced_nworkers = nworkers // nagents
        if nworkers % nagents > 0:
            reduced_nworkers += 1

        nworkers = reduced_nworkers
        self.nworkers = nworkers
       # if recurrent:
        self.hc_actr_buff = {i: [[] for _ in range(nworkers)] for i in range(nagents)}
        self.hc_cr_buff = {i: [[] for _ in range(nworkers)] for i in range(nagents)}
        self.hc_actr_n_buff = {i: [[] for _ in range(nworkers)] for i in range(nagents)}
        self.hc_cr_n_buff = {i: [[] for _ in range(nworkers)] for i in range(nagents)}

        self.obs_buff = {i: [[] for _ in range(nworkers)] for i in range(nagents)}
        self.block_buff = {i: [[] for _ in range(nworkers)] for i in range(nagents)}
        self.d_buff = {i: [[] for _ in range(nworkers)] for i in range(nagents)}
        self.r_buff = {i: [[] for _ in range(nworkers)] for i in range(nagents)}
        self.adv_buff = {i: [[] for _ in range(nworkers)] for i in range(nagents)}
        self.v_buff = {i: [[] for _ in range(nworkers)] for i in range(nagents)}
        self.nobs_buff = {i: [[] for _ in range(nworkers)] for i in range(nagents)}
        self.ap_buff = {i: [[] for _ in range(nworkers)] for i in range(nagents)}
        self.as_buff = {i: [[] for _ in range(nworkers)] for i in range(nagents)}
        self.val_act_buff = {i: [[] for _ in range(nworkers)] for i in range(nagents)}
        self.inf_buff = [[] for _ in range(nworkers)]
    
    # def re_init(self, nagents, nworkers, nrollouts):
    #     self.nagents = nagents
    #     self.nworkers = nworkers
    #     self.nrollouts = nrollouts
    #     self.nbatch = nworkers * nrollouts

    #     reduced_nworkers = nworkers // nagents
    #     if nworkers % nagents > 0:
    #         reduced_nworkers += 1

    #     nworkers = reduced_nworkers
    #     self.nworkers = nworkers

    #     self.obs_buff = {i: [[] for _ in range(nworkers)] for i in range(nagents)}
    #     self.d_buff = {i: [[] for _ in range(nworkers)] for i in range(nagents)}
    #     self.r_buff = {i: [[] for _ in range(nworkers)] for i in range(nagents)}
    #     self.adv_buff = {i: [[] for _ in range(nworkers)] for i in range(nagents)}
    #     self.v_buff = {i: [[] for _ in range(nworkers)] for i in range(nagents)}
    #     self.nobs_buff = {i: [[] for _ in range(nworkers)] for i in range(nagents)}
    #     self.ap_buff = {i: [[] for _ in range(nworkers)] for i in range(nagents)}
    #     self.as_buff = {i: [[] for _ in range(nworkers)] for i in range(nagents)}
    #     self.inf_buff = [[] for _ in range(nworkers)]
        
    def init_model(self, model):
        self.model = model

    def add(self, obs, reward, value, nobs,a_prob, a_select, info, dones, hc_act, hc_cr, hc_act_n, hc_cr_n, blocking, val_act):
        for i, (ob, r, v, nob, a_p, a_s, inf, don, hc_a, hc_c, hc_a_n, hc_c_n, block, val_act_hldr) in enumerate(zip(obs, reward, value, nobs, a_prob, a_select, info, dones, hc_act, hc_cr, hc_act_n, hc_cr_n, blocking, val_act)):
            for i2,d2 in don.items():
                self.d_buff[i2][i].append(d2)
            for i2,ob2 in ob.items():
                self.obs_buff[i2][i].append(ob2)
            for i2,r2 in r.items():
                self.r_buff[i2][i].append(r2)
            for i2,v2 in v.items():
                self.v_buff[i2][i].append(v2)
            for i2,nob2 in nob.items():
                self.nobs_buff[i2][i].append(nob2)
            for i2,a_p2 in a_p.items():
                self.ap_buff[i2][i].append(a_p2)
            for i2,a_s2 in a_s.items():
                self.as_buff[i2][i].append(a_s2)
            for i2,hc_a2 in hc_a.items():
                self.hc_actr_buff[i2][i].append(hc_a2)
            for i2,hc_c2 in hc_c.items():
                self.hc_cr_buff[i2][i].append(hc_c2)
            for i2,hc_a_n2 in hc_a_n.items():
                self.hc_actr_n_buff[i2][i].append(hc_a_n2)
            for i2,hc_c_n2 in hc_c_n.items():
                self.hc_cr_n_buff[i2][i].append(hc_c_n2)
            for i2,hc_c_n2 in hc_c_n.items():
                self.hc_cr_n_buff[i2][i].append(hc_c_n2)
            for i2,bl in block.items():
                self.block_buff[i2][i].append(bl)
            for i2,vld in val_act_hldr.items():
                self.val_act_buff[i2][i].append(vld)
            
            self.inf_buff[i].append(inf)
    
    def add_v_next(self, v_next):
        '''Adds the value of the next state when the buffer is full,
        but terminal stare of an rollout not reached. '''
        for i, v_n in enumerate(v_next):
            for i2,v2 in v_n.items():
                self.v_buff[i2][i].append(v2)

    @property
    def is_full(self):
        full_flags = []
        #Check info buffer only
        for inf_rol in self.inf_buff:
            full_flags.append(len(inf_rol) == self.nrollouts)
        return all(full_flags)
            
    
    def sample(self,gamma, lambda_, empty = True, blocking = False, use_valid_act = False):
        self.calculate_adv(gamma, lambda_)
        # #remove last item in value rollouts
        # for agnt_ind in range(len(self.v_buff)):
        #     for env_ind in range(len(self.v_buff[agnt_ind])):
        #         del self.v_buff[agnt_ind][env_ind][-1]
        obs_hldr = []
        obs_vec_hldr = []
        v_hldr = []
        a_prob_hldr = []
        a_select_hldr = []
        adv_hldr = []
        hc_actr_hldr = []
        hc_cr_hldr = []
        blk_hldr = []
        vld_act_hldr = []
        for agnt_ob, agnt_ap, agnt_as, agnt_adv, agnt_v, hc_actr, hc_cr, hc_actr_n, hc_cr_n, block, val_act \
        in zip(self.obs_buff.values(), self.ap_buff.values(), self.as_buff.values(),\
        self.adv_buff.values(), self.v_buff.values(), self.hc_actr_buff.values(), \
        self.hc_cr_buff.values(), self.hc_actr_n_buff.values(), self.hc_cr_n_buff.values(), \
        self.block_buff.values(), self.val_act_buff.values()):
            
            for roll_ob, roll_ap, roll_as, roll_adv, roll_v, roll_hc_a, roll_hc_c, roll_hc_n, roll_hc_n, roll_blk, roll_val_act  \
            in zip(agnt_ob, agnt_ap, agnt_as, agnt_adv, agnt_v, hc_actr, hc_cr, hc_actr_n, hc_cr_n, block, val_act):
                if type(roll_ob[0]) == tuple:
                    roll_ob_hldr = [torch.from_numpy(r_ob[0]).float() for r_ob in roll_ob]
                    roll_ob_vec = [torch.from_numpy(r_ob[1]).float() for r_ob in roll_ob]
                    obs_vec_hldr.append(torch.stack(roll_ob_vec))
                else:
                    roll_ob_hldr  = [torch.from_numpy(r_ob).float() for r_ob in roll_ob]
                obs_hldr.append(torch.stack(roll_ob_hldr))
                v_hldr.append(torch.cat(roll_v))
                a_prob_hldr.append(torch.cat(roll_ap))
                a_select_hldr.append(torch.cat(roll_as))
                adv_hldr.append(torch.cat(roll_adv))
                if blocking:
                    blk_hldr.append(torch.cat(roll_blk))
                if self.recurrent:
                    (ha, hc) = zip(*roll_hc_a)
                    hc_actr_hldr.append((torch.cat(ha),torch.cat(hc)))
                    (ha, hc) = zip(*roll_hc_c)
                    hc_cr_hldr.append((torch.cat(ha),torch.cat(hc)))
                    # (ha, hc) = zip(*roll_hc_a_n)
                    # hc_actr_n_hldr.append((torch.cat(ha),torch.cat(hc)))
                    # (ha, hc) = zip(*roll_hc_c_n)
                    # hc_cr_n_hldr.append((torch.cat(ha),torch.cat(hc)))
                if use_valid_act:
                    #number of actions is 5:
                    for roll_val_act_hldr2 in roll_val_act:
                        multi_hot = torch.zeros(size=(1, 5), dtype=torch.float32) 
                        for i in roll_val_act_hldr2:
                            multi_hot[0][i] = 1
                        vld_act_hldr.append(multi_hot)

        
        if self.recurrent:
            (ha, hc) = zip(*hc_actr_hldr)
            hc_actr = (torch.cat(ha), torch.cat(hc))
            (ha, hc) = zip(*hc_cr_hldr)
            hc_cr = (torch.cat(ha), torch.cat(hc))
            # (ha, hc) = zip(*hc_actr_n_hldr)
            # hc_actr_n = (torch.cat(ha), torch.cat(hc))
            # (ha, hc) = zip(*hc_cr_n_hldr)
            # hc_cr_n = (torch.cat(ha), torch.cat(hc))
        else:
            hc_actr, hc_cr = None, None

        
        obs = torch.cat(obs_hldr)
        if len(obs_vec_hldr) !=0:
            obs_vec = torch.cat(obs_vec_hldr)
        else:
            obs_vec = []
        a_prob = torch.cat(a_prob_hldr)
        a_select = torch.cat(a_select_hldr)
        adv = torch.cat(adv_hldr)

        #Normalize Batch of rewards
     #   adv = (adv - adv.mean())/(adv.std() + 1e-5)
       ## hldr2 = adv.mean()
        v = torch.cat(v_hldr)

        if blocking:
            blocking_pred = torch.cat(blk_hldr, dim = 0)
            blocking_labels = []
            for ag_id in self.inf_buff[0][0]["blocking"].keys():
                for worker in self.inf_buff:
                    for inf in worker:
                        blocking_labels.append(inf["blocking"][ag_id])
            blocking_labels = torch.tensor(blocking_labels, dtype=torch.float32).unsqueeze(-1)
           # blocking_labels[blocking_labels == 0] = 0.05 
        else:
            blocking_pred, blocking_labels = None, None

        if use_valid_act:
            val_act = torch.cat(vld_act_hldr, dim=0)
        else:
            val_act = None
        inf = []
        for i in self.inf_buff:
            for i2 in i:
                inf.append(i2)
        if empty:
            self.empty()

        #Discard excess samples if number of workers not divisible by n_agents
        if obs.size(0)>self.nbatch:
            obs = obs[:self.nbatch]
            a_prob = a_prob[:self.nbatch]
            a_select = a_select[:self.nbatch]
            adv = adv[:self.nbatch]
            v = v[:self.nbatch]
            if len(obs_vec) != 0:
                obs_vec = obs_vec[:self.nbatch]
            if self.recurrent:
                hc_actr = (hc_actr[0][:self.nbatch], hc_actr[1][:self.nbatch])
                hc_cr = (hc_cr[0][:self.nbatch], hc_cr[1][:self.nbatch])
               # hc_actr_n = hc_actr_n[:self.nbatch]
               # hc_cr_n = hc_cr_n[:self.nbatch]
            if blocking:
                blocking_pred = blocking_pred[:self.nbatch]
                blocking_labels = blocking_labels[:self.nbatch]
            if use_valid_act:
                val_act = val_act[:self.nbatch]

        if len(obs_vec) != 0:
            obs = (obs, obs_vec)
        return obs, a_prob, a_select, adv, v, inf, hc_actr, hc_cr, blocking_labels, blocking_pred, val_act
    
    def calculate_adv(self, gamma, lambda_):
        for i in range(self.nagents):
            for env_id, (r, v,d,nob, inf, hc_actr, hc_cr) in enumerate(zip(self.r_buff[i], \
                self.v_buff[i],self.d_buff[i],self.nobs_buff[i], self.inf_buff, self.hc_actr_n_buff[i], self.hc_cr_n_buff[i])):
                r = torch.tensor(r, dtype = torch.float32).reshape(-1,1)
                v = torch.cat(v)
                if self.recurrent:
                    hc_a = hc_actr #(torch.cat([h[0] for h in hc_actr]), torch.cat([c[1] for c in hc_actr]))
                    hc_c = hc_cr #(torch.cat([h[0] for h in hc_cr]), torch.cat([c[1] for c in hc_cr]))
                else:
                    hc_a = None
                    hc_c = None
                #nob = [torch.from_numpy(no).float() for no in nob]
                #nob = torch.stack(nob)
                ter_mask = [info['terminate'] for info in inf]
                adv = advantages(self.model, i, r, v, ter_mask, d, nob, hc_a, hc_c, gamma, lambda_)
                self.adv_buff[i][env_id].append(adv)

    def empty(self):
        self.obs_buff = {i: [[] for _ in range(self.nworkers)] for i in range(self.nagents)}
        self.block_buff = {i: [[] for _ in range(self.nworkers)] for i in range(self.nagents)}
        self.d_buff = {i: [[] for _ in range(self.nworkers)] for i in range(self.nagents)}
        self.r_buff = {i: [[] for _ in range(self.nworkers)] for i in range(self.nagents)}
        self.v_buff = {i: [[] for _ in range(self.nworkers)] for i in range(self.nagents)}
        self.nobs_buff = {i: [[] for _ in range(self.nworkers)] for i in range(self.nagents)}
        self.ap_buff = {i: [[] for _ in range(self.nworkers)] for i in range(self.nagents)}
        self.as_buff = {i: [[] for _ in range(self.nworkers)] for i in range(self.nagents)}
        self.adv_buff = {i: [[] for _ in range(self.nworkers)] for i in range(self.nagents)}
        self.val_act_buff = {i: [[] for _ in range(self.nworkers)] for i in range(self.nagents)}
        self.inf_buff = [[] for _ in range(self.nworkers)]

        #if self.recurrent:
        self.hc_actr_buff = {i: [[] for _ in range(self.nworkers)] for i in range(self.nagents)}
        self.hc_cr_buff = {i: [[] for _ in range(self.nworkers)] for i in range(self.nagents)}
        self.hc_actr_n_buff = {i: [[] for _ in range(self.nworkers)] for i in range(self.nagents)}
        self.hc_cr_n_buff = {i: [[] for _ in range(self.nworkers)] for i in range(self.nagents)}


