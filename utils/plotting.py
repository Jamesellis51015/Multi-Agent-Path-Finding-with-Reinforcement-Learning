from tensorflow.python.summary.summary_iterator import summary_iterator
import tensorflow as tf

import matplotlib
import numpy as np
import matplotlib.pyplot as plt


def get_fields(file_location):
    fields = {}
    try:
        for summary in summary_iterator(file_location):
            for v in summary.summary.value:
                if v.tag not in fields:
                    fields[v.tag] = []
                fields[v.tag].append(v.simple_value)
    except:
        return fields
    return fields

        
# def histogram(file_location, key, bins):
#     #print(tf.data.TFRecordDataset(file_location))
#    # fields = get_fields(file_location)

#     test_data = np.arange(10)
#     test_data[:5] *=2


#     fig, ax = plt.subplots()
#     num_bins = 5

#     n, bins, patches = ax.hist(test_data, num_bins)
#     print(n)
#     print(bins)
#     ax.plot(bins)

#     plt.show()

# def plot_hist_data(name_dir_dict, search_terms, data_groups):
#     '''data_group is dict of key: [] '''
#     #exp_names = {name.split("/")[-2]:name for name in experiment_dirs}

#     filter = name_dir_dict.keys()
#     for term in search_terms:
#         filter = [fil for fil in filter if term in fil]
    
#     exp_groups = {}
#     for key in data_groups.keys():
#         exp_groups[key] = [i for i in filter if key in i]
    
#     for key, vals in exp_groups.items():
#         sub_groups = data_groups[key]
#         if len(add_lables) == 0:
#             exp_groups[key] = {str(i):v for i,v in enumerate(vals)}
#         else:
#             sub_group_dict = {}
#             for group in sub_groups:
#                 sub_group_dict["".join(group)] = [v for v in vals if all(g in v for g in group)]
#             exp_groups[key] = sub_group_dict
    
#     label_dir = {}
#     for key, val in exp_groups.items():


                
def plot_hist(file_paths, field, bins = None):
    all_data = [get_fields(fil) for fil in file_paths]
    
    relevant_data = [d[field] for d in all_data]





        


if __name__ == "__main__":
    #file = "/home/desktop123/Documents/Academics/Code/Testing/events.out.tfevents.1586894008.8d0afe01b80f"
    #file = '/home/desktop123/Desktop/Gridworld2/Exp0CN/cnv1_obj_d0.0IC3_enc_type_mlp_disc0.99_bat500_val_coeff_0.01_ent_coeff_0.0_agnts4_0/benchmark/benchmark_4999/events.out.tfevents.1587337018.65e0ac645d53'
    #print(get_fields(file))
    #histogram(file, [], [])
    #a = np.array([1,1,1,1,1,2,2,2,2,5,5,5,5,5,5,5,5,5,5,5,5,5])
    #b = np.array([1,1,3,3,3,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,5,5])
   # file1 = '/home/desktop123/Desktop/Gridworld_Exp0CN_results/Exp0CN/cnv0_obj_d0.0_envsz5IC3_enc_type_mlp_disc0.99_bat500_val_coeff_0.01_ent_coeff_0.0_agnts4_0/benchmark/benchmark_5999/events.out.tfevents.1587384575.b2c7586a6495'
    #file2 = '/home/desktop123/Desktop/Gridworld_Exp0CN_results/Exp0CN/cnv0_obj_d0.0_envsz5MAAC_enc_type_mlp_reward_scale100_agnts4_0/benchmark/benchmark_49999/events.out.tfevents.1587408007.07a954f55605'
    #file2 ='/home/desktop123/Desktop/Gridworld_Exp0CN_results/Exp0CN/cnv0_obj_d0.0_envsz5MAAC_enc_type_mlp_reward_scale100_agnts4_0/benchmark/benchmark_49000/events.out.tfevents.1587407934.07a954f55605'
    #file2 = '/home/desktop123/Desktop/Gridworld_Exp0CN_results/Exp0CN/cnv0_obj_d0.0_envsz5MAAC_enc_type_mlp_reward_scale100_agnts4_0/benchmark/benchmark_48000/events.out.tfevents.1587407859.07a954f55605'
    
    file1 = '/home/desktop123/Desktop/Gridworld_Exp0CN_results/Exp0CN/cnv0_obj_d0.0_envsz5IC3_enc_type_mlp_disc0.99_bat500_val_coeff_0.01_ent_coeff_0.0_agnts6_0/benchmark/benchmark_5999/events.out.tfevents.1587396009.b2c7586a6495'

    file2 = '/home/desktop123/Desktop/Gridworld_Exp0CN_results/Exp0CN/cnv0_obj_d0.0_envsz5MAAC_enc_type_mlp_reward_scale100_agnts6_0/benchmark/benchmark_58000/events.out.tfevents.1587570916.d8a9656cc728'
    #file1 = '/home/desktop123/Desktop/Gridworld_Exp0CN_results/Exp0CN/cnv1_obj_d0.0IC3_enc_type_mlp_disc0.99_bat500_val_coeff_0.01_ent_coeff_0.0_agnts4_0/benchmark/benchmark_4999/events.out.tfevents.1587337018.65e0ac645d53'
    #file2 = '/home/desktop123/Desktop/Gridworld_Exp0CN_results/Exp0CN/cnv1_obj_d0.0_envsz5MAAC_enc_type_mlp_reward_scale100_agnts4_0/benchmark/benchmark_43000/events.out.tfevents.1587467787.8aa10b15b267'


    
    description = "6 Agents - No obstaces - Partially Observable"

    col = ['red', 'lime']
    labels = ['IC3', 'MAAC']

    all_files = [file1, file2]
    all_data = [get_fields(fil) for fil in all_files]
    #self.log_keys = ['total_steps', 'terminate', 'total_rewards', 'total_agent_collisions', 'total_obstacle_collisions', 'total_avg_agent_r', 'total_ep_global_r', 'agent_dones']
    field = 'agent_dones'
    fields = ['total_agent_collisions', 'total_avg_agent_r', 'agent_dones']
    min_max_step = [(0,40,2), (-1.5, 1.5, 0.1), (0,1.2, 0.2)]
    titles = ["Agents Collisions", "Average Agent Rewards", "Agent Dones"]
    for field, (mn,mx,step), title in zip(fields, min_max_step, titles):
        relevant_data = [d[field] for d in all_data]
        dat_len = [len(d) for d in relevant_data]
        min_len = min(dat_len)
        relevant_data = [d[0:min_len] for d in relevant_data]
        #bins = [0,2,4,6,8,10, 12,14, 16,18,20,30,40,50]
       # step =0.2
    #  bins = [i for i in range(0, 1,step)]
        bins = np.arange(mn, mx, step)
        n, bins, pathches = plt.hist(relevant_data, bins=bins, color = col, label=labels)
        plt.legend()
        ax = plt.axes()
        ax.xaxis.set_ticks(np.arange(min(bins), max(bins) + step, step))
        ax.set_title("{} \n {}".format(title, description))
        plt.show()