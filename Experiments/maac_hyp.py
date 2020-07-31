import main
from sklearn.model_selection import ParameterGrid


def maac_1A1():
    episodes = str(15000)
    # experiment_group_name = "TestMaac"
    # work_dir = experiment_group_name
    # plot_dir = experiment_group_name + "_Central_TestBases"

    experiment_group_name = "Maac_1A2"
    work_dir = '/home/james/Desktop/Gridworld/EXPERIMENTS/' + experiment_group_name
    plot_dir = '/home/james/Desktop/Gridworld/CENTRAL_TENSORBOARD/' + experiment_group_name #+ "_Central"
    #work_dir = experiment_group_name
    #plot_dir = experiment_group_name + "_Central"
 
    parmeter_grid1 = {
        "seed": [1],
        #"batch_sizes": [500,1000,2000],
       # "entropy_coeff":[0.0],
        "discount": [0.9,0.95,1.0],  #[0.1,0.5,0.8], #,0.9,0.95,1.0],
       # "discount":[0.9],
        "rollout_threads": [1],
        "reward_scale": [10],
        "policy": ["MAAC"],
        "base_policy_type": ["mlp"],
        "n_agents": [4],
        "obj_density": [0.2],
        "env_size": [5],
        ###
        "maac_steps_per_update":[100],
        "maac_num_updates":[4],
        "maac_attend_heads":[4],
        "maac_batch_size": [1024]
    }

    grid1 = ParameterGrid(parmeter_grid1)
    
    for param in grid1:
        args = []
        args = ["--working_directory", work_dir,"--alternative_plot_dir", plot_dir]

        name = "MAAC" + "_disc_" + str(param["discount"]) \
            + "_rewardscale_" + str(param["reward_scale"]) \
            + "_minibatch_" + str(param["maac_batch_size"]) \
            + "_nupdates_" + str(param["maac_steps_per_update"]) \
            + "_attheads_" + str(param["maac_attend_heads"]) \
            + "_envsize_" + str(param["env_size"]) \
            + "_nagents_"+ str(param["n_agents"]) \
            + "_objdensity_"+ str(param["obj_density"]) \
            + "_seed_"+ str(param["seed"])


        args.extend(["--env_name","independent_navigation-v0"])
        args.extend(["--n_agents", str(param["n_agents"])])
        args.extend(["--map_shape", str(param["env_size"])])
        args.extend(["--obj_density", str(param["obj_density"])])

        args.extend(["--maac_buffer_length", str(int(1e5))])
        args.extend(["--policy", str(param["policy"])])
        args.extend(["--name", name])
        args.extend(["--maac_use_gpu"])
        args.extend(["--maac_share_actor"])
        args.extend(["--maac_reward_scale", str(param["reward_scale"])])
        args.extend(["--maac_n_rollout_threads", str(param["rollout_threads"])])
        args.extend(["--maac_steps_per_update", str(param["maac_steps_per_update"])])
        args.extend(["--maac_num_updates", str(param["maac_num_updates"])])
        args.extend(["--maac_attend_heads", str(param["maac_attend_heads"])])
        args.extend(["--maac_n_episodes", episodes])
        args.extend(["--maac_batch_size", str(param["maac_batch_size"])])
        args.extend(["--maac_base_policy_type", param["base_policy_type"]]) 
        args.extend(["--benchmark_frequency", str(int(100000))])
        args.extend(["--benchmark_num_episodes", str(int(50))])
        args.extend(["--benchmark_render_length", str(int(25))])
        args.extend(["--render_rate", str(int(50))])

        args.extend(["--maac_gamma", str(param["discount"])])

       # parser.add_argument("--maac_gamma", default=0.9, type=float)

        args.extend(["--seed", str(param["seed"])])
    
        main.main(args)
    
def maac_1B1():
    episodes = str(15000)
    # experiment_group_name = "TestMaac"
    # work_dir = experiment_group_name
    # plot_dir = experiment_group_name + "_Central_TestBases"

    experiment_group_name = "Maac_1B2"
    work_dir = '/home/james/Desktop/Gridworld/EXPERIMENTS/' + experiment_group_name
    plot_dir = '/home/james/Desktop/Gridworld/CENTRAL_TENSORBOARD/' + experiment_group_name #+ "_Central"
  #  work_dir = experiment_group_name
  #  plot_dir = experiment_group_name + "_Central"

 
    parmeter_grid1 = {
        "seed": [1],
        #"batch_sizes": [500,1000,2000],
       # "entropy_coeff":[0.0],
        "discount":[0.9],
        "rollout_threads": [1],
        "reward_scale": [10,50,100],
        "policy": ["MAAC"],
        "base_policy_type": ["mlp"],
        "n_agents": [4],
        "obj_density": [0.2],
        "env_size": [5],
        ###
        "maac_steps_per_update":[100],
        "maac_num_updates":[4],
        "maac_attend_heads":[4],
        "maac_batch_size": [1024]
    }

    grid1 = ParameterGrid(parmeter_grid1)
    
    for param in grid1:
        args = []
        args = ["--working_directory", work_dir,"--alternative_plot_dir", plot_dir]

        name = "MAAC" + "_disc_" + str(param["discount"]) \
            + "_rewardscale_" + str(param["reward_scale"]) \
            + "_minibatch_" + str(param["maac_batch_size"]) \
            + "_nupdates_" + str(param["maac_steps_per_update"]) \
            + "_attheads_" + str(param["maac_attend_heads"]) \
            + "_envsize_" + str(param["env_size"]) \
            + "_nagents_"+ str(param["n_agents"]) \
            + "_objdensity_"+ str(param["obj_density"]) \
            + "_seed_"+ str(param["seed"])


        args.extend(["--env_name","independent_navigation-v0"])
        args.extend(["--n_agents", str(param["n_agents"])])
        args.extend(["--map_shape", str(param["env_size"])])
        args.extend(["--obj_density", str(param["obj_density"])])

        args.extend(["--maac_buffer_length", str(int(1e5))])
        args.extend(["--policy", str(param["policy"])])
        args.extend(["--name", name])
        args.extend(["--maac_use_gpu"])
        args.extend(["--maac_share_actor"])
        args.extend(["--maac_reward_scale", str(param["reward_scale"])])
        args.extend(["--maac_n_rollout_threads", str(param["rollout_threads"])])
        args.extend(["--maac_steps_per_update", str(param["maac_steps_per_update"])])
        args.extend(["--maac_num_updates", str(param["maac_num_updates"])])
        args.extend(["--maac_attend_heads", str(param["maac_attend_heads"])])
        args.extend(["--maac_n_episodes", episodes])
        args.extend(["--maac_batch_size", str(param["maac_batch_size"])])
        args.extend(["--maac_base_policy_type", param["base_policy_type"]]) 
        args.extend(["--benchmark_frequency", str(int(1000))])
        args.extend(["--benchmark_num_episodes", str(int(50))])
        args.extend(["--benchmark_render_length", str(int(25))])
        args.extend(["--render_rate", str(int(50))])

        args.extend(["--maac_gamma", str(param["discount"])])

        args.extend(["--seed", str(param["seed"])])
    
        main.main(args)

def maac_1C1():
    episodes = str(15000)
    # experiment_group_name = "TestMaac"
    # work_dir = experiment_group_name
    # plot_dir = experiment_group_name + "_Central_TestBases"

    experiment_group_name = "Maac_1C2"
    work_dir = '/home/james/Desktop/Gridworld/EXPERIMENTS/' + experiment_group_name
    plot_dir = '/home/james/Desktop/Gridworld/CENTRAL_TENSORBOARD/' + experiment_group_name #+ "_Central"
  #  work_dir = experiment_group_name
  #  plot_dir = experiment_group_name + "_Central"

 
    parmeter_grid1 = {
        "seed": [1],
        #"batch_sizes": [500,1000,2000],
       # "entropy_coeff":[0.0],
        "discount":[0.9],
        "rollout_threads": [1],
        "reward_scale": [10],
        "policy": ["MAAC"],
        "base_policy_type": ["mlp"],
        "n_agents": [4],
        "obj_density": [0.2],
        "env_size": [5],
        ###
        "maac_steps_per_update":[100],
        "maac_num_updates":[4],
        "maac_attend_heads":[4],
       # "maac_batch_size": [256]
        "maac_batch_size": [256, 512,1024,2048]
    }

    grid1 = ParameterGrid(parmeter_grid1)
    
    for param in grid1:
        args = []
        args = ["--working_directory", work_dir,"--alternative_plot_dir", plot_dir]

        name = "MAAC" + "_disc_" + str(param["discount"]) \
            + "_rewardscale_" + str(param["reward_scale"]) \
            + "_minibatch_" + str(param["maac_batch_size"]) \
            + "_nupdates_" + str(param["maac_steps_per_update"]) \
            + "_attheads_" + str(param["maac_attend_heads"]) \
            + "_envsize_" + str(param["env_size"]) \
            + "_nagents_"+ str(param["n_agents"]) \
            + "_objdensity_"+ str(param["obj_density"]) \
            + "_seed_"+ str(param["seed"])


        args.extend(["--env_name","independent_navigation-v0"])
        args.extend(["--n_agents", str(param["n_agents"])])
        args.extend(["--map_shape", str(param["env_size"])])
        args.extend(["--obj_density", str(param["obj_density"])])

        args.extend(["--maac_buffer_length", str(int(1e5))])
        args.extend(["--policy", str(param["policy"])])
        args.extend(["--name", name])
        args.extend(["--maac_use_gpu"])
        args.extend(["--maac_share_actor"])
        args.extend(["--maac_reward_scale", str(param["reward_scale"])])
        args.extend(["--maac_n_rollout_threads", str(param["rollout_threads"])])
        args.extend(["--maac_steps_per_update", str(param["maac_steps_per_update"])])
        args.extend(["--maac_num_updates", str(param["maac_num_updates"])])
        args.extend(["--maac_attend_heads", str(param["maac_attend_heads"])])
        args.extend(["--maac_n_episodes", episodes])
        args.extend(["--maac_batch_size", str(param["maac_batch_size"])])
        args.extend(["--maac_base_policy_type", param["base_policy_type"]]) 
        args.extend(["--benchmark_frequency", str(int(100000))])
        args.extend(["--benchmark_num_episodes", str(int(50))])
        args.extend(["--benchmark_render_length", str(int(25))])
        args.extend(["--render_rate", str(int(50))])

        args.extend(["--maac_gamma", str(param["discount"])])

        args.extend(["--seed", str(param["seed"])])
    
        main.main(args)


def maac_1D1():
    episodes = str(30000)
    # experiment_group_name = "TestMaac"
    # work_dir = experiment_group_name
    # plot_dir = experiment_group_name + "_Central_TestBases"

    experiment_group_name = "Maac_1D2"
    work_dir = '/home/james/Desktop/Gridworld/EXPERIMENTS/' + experiment_group_name
    plot_dir = '/home/james/Desktop/Gridworld/CENTRAL_TENSORBOARD/' + experiment_group_name #+ "_Central"
    # work_dir = experiment_group_name
    # plot_dir = experiment_group_name + "_Central"

 
    parmeter_grid1 = {
        "seed": [1],
        #"batch_sizes": [500,1000,2000],
       # "entropy_coeff":[0.0],
        "discount":[0.9],
        "rollout_threads": [1],
        "reward_scale": [10],
        "policy": ["MAAC"],
        "base_policy_type": ["mlp"],
        "n_agents": [8],
        "obj_density": [0.2],
        "env_size": [7],
        ###
        "maac_steps_per_update":[100],
        "maac_num_updates":[4],
        "maac_attend_heads":[4,8],
        "maac_batch_size": [1024]
    }

    grid1 = ParameterGrid(parmeter_grid1)
    
    for param in grid1:
        args = []
        args = ["--working_directory", work_dir,"--alternative_plot_dir", plot_dir]

        name = "MAAC" + "_disc_" + str(param["discount"]) \
            + "_rewardscale_" + str(param["reward_scale"]) \
            + "_minibatch_" + str(param["maac_batch_size"]) \
            + "_nupdates_" + str(param["maac_steps_per_update"]) \
            + "_attheads_" + str(param["maac_attend_heads"]) \
            + "_envsize_" + str(param["env_size"]) \
            + "_nagents_"+ str(param["n_agents"]) \
            + "_objdensity_"+ str(param["obj_density"]) \
            + "_seed_"+ str(param["seed"])


        args.extend(["--env_name","independent_navigation-v0"])
        args.extend(["--n_agents", str(param["n_agents"])])
        args.extend(["--map_shape", str(param["env_size"])])
        args.extend(["--obj_density", str(param["obj_density"])])

        args.extend(["--maac_buffer_length", str(int(1e5))])
        args.extend(["--policy", str(param["policy"])])
        args.extend(["--name", name])
        args.extend(["--maac_use_gpu"])
        args.extend(["--maac_share_actor"])
        args.extend(["--maac_reward_scale", str(param["reward_scale"])])
        args.extend(["--maac_n_rollout_threads", str(param["rollout_threads"])])
        args.extend(["--maac_steps_per_update", str(param["maac_steps_per_update"])])
        args.extend(["--maac_num_updates", str(param["maac_num_updates"])])
        args.extend(["--maac_attend_heads", str(param["maac_attend_heads"])])
        args.extend(["--maac_n_episodes", episodes])
        args.extend(["--maac_batch_size", str(param["maac_batch_size"])])
        args.extend(["--maac_base_policy_type", param["base_policy_type"]]) 
        args.extend(["--benchmark_frequency", str(int(1000))])
        args.extend(["--benchmark_num_episodes", str(int(50))])
        args.extend(["--benchmark_render_length", str(int(25))])
        args.extend(["--render_rate", str(int(50))])

        args.extend(["--maac_gamma", str(param["discount"])])

        args.extend(["--seed", str(param["seed"])])
    
        main.main(args)


def maac_1E1():
    #Buffer size:


    episodes = str(15000)
    # experiment_group_name = "TestMaac"
    # work_dir = experiment_group_name
    # plot_dir = experiment_group_name + "_Central_TestBases"

    experiment_group_name = "Maac_1E"
    #work_dir = '/home/james/Desktop/Gridworld/EXPERIMENTS/' + experiment_group_name
    #plot_dir = '/home/james/Desktop/Gridworld/CENTRAL_TENSORBOARD/' + experiment_group_name #+ "_Central"
    work_dir = experiment_group_name
    plot_dir = experiment_group_name + "_Central"

 
    parmeter_grid1 = {
        "seed": [1],
        #"batch_sizes": [500,1000,2000],
       # "entropy_coeff":[0.0],
        "discount":[0.5],
        "rollout_threads": [1],
        "reward_scale": [10],
        "policy": ["MAAC"],
        "base_policy_type": ["mlp"],
        "n_agents": [6],
        "obj_density": [0.2],
        "env_size": [7],
        ###
        "maac_steps_per_update":[100],
        "maac_num_updates":[4],
        "maac_attend_heads":[4],
        "maac_batch_size": [1024],
        "bufflen": [1e5, 5e5, 1e6]
    }

    grid1 = ParameterGrid(parmeter_grid1)
    
    for param in grid1:
        args = []
        args = ["--working_directory", work_dir,"--alternative_plot_dir", plot_dir]

        name = "MAAC" + "_disc_" + str(param["discount"]) \
            + "_rewardscale_" + str(param["reward_scale"]) \
            + "_minibatch_" + str(param["maac_batch_size"]) \
            + "_nupdates_" + str(param["maac_steps_per_update"]) \
            + "_attheads_" + str(param["maac_attend_heads"]) \
            + "_envsize_" + str(param["env_size"]) \
            + "_nagents_"+ str(param["n_agents"]) \
            + "_objdensity_"+ str(param["obj_density"]) \
            + "_bufflen_"+ str(param["bufflen"]) \
            + "_seed_"+ str(param["seed"])


        args.extend(["--env_name","independent_navigation-v0"])
        args.extend(["--n_agents", str(param["n_agents"])])
        args.extend(["--map_shape", str(param["env_size"])])
        args.extend(["--obj_density", str(param["obj_density"])])

        args.extend(["--maac_buffer_length", str(int(param["bufflen"]))])
        args.extend(["--policy", str(param["policy"])])
        args.extend(["--name", name])
        args.extend(["--maac_use_gpu"])
        args.extend(["--maac_share_actor"])
        args.extend(["--maac_reward_scale", str(param["reward_scale"])])
        args.extend(["--maac_n_rollout_threads", str(param["rollout_threads"])])
        args.extend(["--maac_steps_per_update", str(param["maac_steps_per_update"])])
        args.extend(["--maac_num_updates", str(param["maac_num_updates"])])
        args.extend(["--maac_attend_heads", str(param["maac_attend_heads"])])
        args.extend(["--maac_n_episodes", episodes])
        args.extend(["--maac_batch_size", str(param["maac_batch_size"])])
        args.extend(["--maac_base_policy_type", param["base_policy_type"]]) 
        args.extend(["--benchmark_frequency", str(int(1000))])
        args.extend(["--benchmark_num_episodes", str(int(50))])
        args.extend(["--benchmark_render_length", str(int(25))])
        args.extend(["--render_rate", str(int(50))])

        args.extend(["--seed", str(param["seed"])])
    
        main.main(args)


####################################################
##  NR 2: REWARD STRUCTURE
##########################################


def maac_2A1():
    episodes = str(30000)
    # experiment_group_name = "TestMaac"
    # work_dir = experiment_group_name
    # plot_dir = experiment_group_name + "_Central_TestBases"

    experiment_group_name = "2A"
    work_dir = '/home/james/Desktop/Gridworld/EXPERIMENTS/' + experiment_group_name
    plot_dir = '/home/james/Desktop/Gridworld/CENTRAL_TENSORBOARD/' + experiment_group_name #+ "_Central"
    #work_dir = experiment_group_name
    #plot_dir = experiment_group_name + "_Central"
 
    parmeter_grid1 = {
        "seed": [1],
        #"batch_sizes": [500,1000,2000],
       # "entropy_coeff":[0.0],
        #"discount":[0.1,0.5,0.9,0.95],
        "discount":[1.0],#[0.9],
        "rollout_threads": [1],
        "reward_scale": [10],
        "policy": ["MAAC"],
        "base_policy_type": ["mlp"],
        "n_agents": [3],
        "obj_density": [0.0],
        "env_size": [5],
        ###
        "maac_steps_per_update":[100],
        "maac_num_updates":[1],
        "maac_attend_heads":[4],
        "maac_batch_size": [1024],
        "env": ["independent_navigation-v4_1"]
    }

    grid1 = ParameterGrid(parmeter_grid1)
    
    for param in grid1:
        args = []
        args = ["--working_directory", work_dir,"--alternative_plot_dir", plot_dir]

        name = "MAAC_" + param["env"] \
            + "_disc_" + str(param["discount"]) \
            + "_rewardscale_" + str(param["reward_scale"]) \
            + "_minibatch_" + str(param["maac_batch_size"]) \
            + "_nupdates_" + str(param["maac_steps_per_update"]) \
            + "_attheads_" + str(param["maac_attend_heads"]) \
            + "_envsize_" + str(param["env_size"]) \
            + "_nagents_"+ str(param["n_agents"]) \
            + "_objdensity_"+ str(param["obj_density"]) \
            + "_seed_"+ str(param["seed"])


        args.extend(["--env_name", param["env"]]) #,"independent_navigation-v0"])
        args.extend(["--n_agents", str(param["n_agents"])])
        args.extend(["--map_shape", str(param["env_size"])])
        args.extend(["--obj_density", str(param["obj_density"])])

        args.extend(["--maac_buffer_length", str(int(1e5))])
        args.extend(["--policy", str(param["policy"])])
        args.extend(["--name", name])
        args.extend(["--maac_use_gpu"])
        args.extend(["--maac_share_actor"])
        args.extend(["--maac_reward_scale", str(param["reward_scale"])])
        args.extend(["--maac_n_rollout_threads", str(param["rollout_threads"])])
        args.extend(["--maac_steps_per_update", str(param["maac_steps_per_update"])])
        args.extend(["--maac_num_updates", str(param["maac_num_updates"])])
        args.extend(["--maac_attend_heads", str(param["maac_attend_heads"])])
        args.extend(["--maac_n_episodes", episodes])
        args.extend(["--maac_batch_size", str(param["maac_batch_size"])])
        args.extend(["--maac_base_policy_type", param["base_policy_type"]]) 
        args.extend(["--benchmark_frequency", str(int(100000))])
        args.extend(["--benchmark_num_episodes", str(int(50))])
        args.extend(["--benchmark_render_length", str(int(25))])
        args.extend(["--render_rate", str(int(50))])

        args.extend(["--maac_gamma", str(param["discount"])])

       # parser.add_argument("--maac_gamma", default=0.9, type=float)

        args.extend(["--seed", str(param["seed"])])
    
        main.main(args)

def maac_2A2():
    episodes = str(30000)
    # experiment_group_name = "TestMaac"
    # work_dir = experiment_group_name
    # plot_dir = experiment_group_name + "_Central_TestBases"

    experiment_group_name = "2A"
    work_dir = '/home/james/Desktop/Gridworld/EXPERIMENTS/' + experiment_group_name
    plot_dir = '/home/james/Desktop/Gridworld/CENTRAL_TENSORBOARD/' + experiment_group_name #+ "_Central"
    #work_dir = experiment_group_name
    #plot_dir = experiment_group_name + "_Central"
 
    parmeter_grid1 = {
        "seed": [1],
        #"batch_sizes": [500,1000,2000],
       # "entropy_coeff":[0.0],
        #"discount":[0.1,0.5,0.9,0.95],
        "discount":[1.0],#[0.9],
        "rollout_threads": [1],
        "reward_scale": [10],
        "policy": ["MAAC"],
        "base_policy_type": ["mlp"],
        "n_agents": [3],
        "obj_density": [0.0],
        "env_size": [5],
        ###
        "maac_steps_per_update":[100],
        "maac_num_updates":[1],
        "maac_attend_heads":[4],
        "maac_batch_size": [1024],
        "env": ["independent_navigation-v4_2"]
    }

    grid1 = ParameterGrid(parmeter_grid1)
    
    for param in grid1:
        args = []
        args = ["--working_directory", work_dir,"--alternative_plot_dir", plot_dir]

        name = "MAAC_" + param["env"] \
            + "_disc_" + str(param["discount"]) \
            + "_rewardscale_" + str(param["reward_scale"]) \
            + "_minibatch_" + str(param["maac_batch_size"]) \
            + "_nupdates_" + str(param["maac_steps_per_update"]) \
            + "_attheads_" + str(param["maac_attend_heads"]) \
            + "_envsize_" + str(param["env_size"]) \
            + "_nagents_"+ str(param["n_agents"]) \
            + "_objdensity_"+ str(param["obj_density"]) \
            + "_seed_"+ str(param["seed"])


        args.extend(["--env_name", param["env"]]) #,"independent_navigation-v0"])
        args.extend(["--n_agents", str(param["n_agents"])])
        args.extend(["--map_shape", str(param["env_size"])])
        args.extend(["--obj_density", str(param["obj_density"])])

        args.extend(["--maac_buffer_length", str(int(1e5))])
        args.extend(["--policy", str(param["policy"])])
        args.extend(["--name", name])
        args.extend(["--maac_use_gpu"])
        args.extend(["--maac_share_actor"])
        args.extend(["--maac_reward_scale", str(param["reward_scale"])])
        args.extend(["--maac_n_rollout_threads", str(param["rollout_threads"])])
        args.extend(["--maac_steps_per_update", str(param["maac_steps_per_update"])])
        args.extend(["--maac_num_updates", str(param["maac_num_updates"])])
        args.extend(["--maac_attend_heads", str(param["maac_attend_heads"])])
        args.extend(["--maac_n_episodes", episodes])
        args.extend(["--maac_batch_size", str(param["maac_batch_size"])])
        args.extend(["--maac_base_policy_type", param["base_policy_type"]]) 
        args.extend(["--benchmark_frequency", str(int(100000))])
        args.extend(["--benchmark_num_episodes", str(int(50))])
        args.extend(["--benchmark_render_length", str(int(25))])
        args.extend(["--render_rate", str(int(50))])

        args.extend(["--maac_gamma", str(param["discount"])])

       # parser.add_argument("--maac_gamma", default=0.9, type=float)

        args.extend(["--seed", str(param["seed"])])
    
        main.main(args)

def maac_2A3():
    episodes = str(30000)
    # experiment_group_name = "TestMaac"
    # work_dir = experiment_group_name
    # plot_dir = experiment_group_name + "_Central_TestBases"

    experiment_group_name = "2A"
    work_dir = '/home/james/Desktop/Gridworld/EXPERIMENTS/' + experiment_group_name
    plot_dir = '/home/james/Desktop/Gridworld/CENTRAL_TENSORBOARD/' + experiment_group_name #+ "_Central"
    #work_dir = experiment_group_name
    #plot_dir = experiment_group_name + "_Central"
 
    parmeter_grid1 = {
        "seed": [1],
        #"batch_sizes": [500,1000,2000],
       # "entropy_coeff":[0.0],
        #"discount":[0.1,0.5,0.9,0.95],
        "discount":[1.0],#[0.9],
        "rollout_threads": [1],
        "reward_scale": [10],
        "policy": ["MAAC"],
        "base_policy_type": ["mlp"],
        "n_agents": [3],
        "obj_density": [0.0],
        "env_size": [5],
        ###
        "maac_steps_per_update":[100],
        "maac_num_updates":[1],
        "maac_attend_heads":[4],
        "maac_batch_size": [1024],
        "env": ["independent_navigation-v4_3"]
    }

    grid1 = ParameterGrid(parmeter_grid1)
    
    for param in grid1:
        args = []
        args = ["--working_directory", work_dir,"--alternative_plot_dir", plot_dir]

        name = "MAAC_" + param["env"] \
            + "_disc_" + str(param["discount"]) \
            + "_rewardscale_" + str(param["reward_scale"]) \
            + "_minibatch_" + str(param["maac_batch_size"]) \
            + "_nupdates_" + str(param["maac_steps_per_update"]) \
            + "_attheads_" + str(param["maac_attend_heads"]) \
            + "_envsize_" + str(param["env_size"]) \
            + "_nagents_"+ str(param["n_agents"]) \
            + "_objdensity_"+ str(param["obj_density"]) \
            + "_seed_"+ str(param["seed"])


        args.extend(["--env_name", param["env"]]) #,"independent_navigation-v0"])
        args.extend(["--n_agents", str(param["n_agents"])])
        args.extend(["--map_shape", str(param["env_size"])])
        args.extend(["--obj_density", str(param["obj_density"])])

        args.extend(["--maac_buffer_length", str(int(1e5))])
        args.extend(["--policy", str(param["policy"])])
        args.extend(["--name", name])
        args.extend(["--maac_use_gpu"])
        args.extend(["--maac_share_actor"])
        args.extend(["--maac_reward_scale", str(param["reward_scale"])])
        args.extend(["--maac_n_rollout_threads", str(param["rollout_threads"])])
        args.extend(["--maac_steps_per_update", str(param["maac_steps_per_update"])])
        args.extend(["--maac_num_updates", str(param["maac_num_updates"])])
        args.extend(["--maac_attend_heads", str(param["maac_attend_heads"])])
        args.extend(["--maac_n_episodes", episodes])
        args.extend(["--maac_batch_size", str(param["maac_batch_size"])])
        args.extend(["--maac_base_policy_type", param["base_policy_type"]]) 
        args.extend(["--benchmark_frequency", str(int(100000))])
        args.extend(["--benchmark_num_episodes", str(int(50))])
        args.extend(["--benchmark_render_length", str(int(25))])
        args.extend(["--render_rate", str(int(50))])

        args.extend(["--maac_gamma", str(param["discount"])])

       # parser.add_argument("--maac_gamma", default=0.9, type=float)

        args.extend(["--seed", str(param["seed"])])
    
        main.main(args)


############################
#With obstacles:
############################

def maac_2A4():
    episodes = str(100000)
    # experiment_group_name = "TestMaac"
    # work_dir = experiment_group_name
    # plot_dir = experiment_group_name + "_Central_TestBases"

    experiment_group_name = "2A"
    work_dir = '/home/james/Desktop/Gridworld/EXPERIMENTS/' + experiment_group_name
    plot_dir = '/home/james/Desktop/Gridworld/CENTRAL_TENSORBOARD/' + experiment_group_name #+ "_Central"
    #work_dir = experiment_group_name
    #plot_dir = experiment_group_name + "_Central"
 
    parmeter_grid1 = {
        "seed": [1],
        #"batch_sizes": [500,1000,2000],
       # "entropy_coeff":[0.0],
        #"discount":[0.1,0.5,0.9,0.95],
        "discount":[0.9],
        "rollout_threads": [1],
        "reward_scale": [10],
        "policy": ["MAAC"],
        "base_policy_type": ["mlp"],
        "n_agents": [3],
        "obj_density": [0.4],
        "env_size": [5],
        ###
        "maac_steps_per_update":[100],
        "maac_num_updates":[1],
        "maac_attend_heads":[4],
        "maac_batch_size": [1024],
        "env": ["independent_navigation-v4_1"]
    }

    grid1 = ParameterGrid(parmeter_grid1)
    
    for param in grid1:
        args = []
        args = ["--working_directory", work_dir,"--alternative_plot_dir", plot_dir]

        name = "MAAC_" + param["env"] \
            + "_disc_" + str(param["discount"]) \
            + "_rewardscale_" + str(param["reward_scale"]) \
            + "_minibatch_" + str(param["maac_batch_size"]) \
            + "_nupdates_" + str(param["maac_steps_per_update"]) \
            + "_attheads_" + str(param["maac_attend_heads"]) \
            + "_envsize_" + str(param["env_size"]) \
            + "_nagents_"+ str(param["n_agents"]) \
            + "_objdensity_"+ str(param["obj_density"]) \
            + "_seed_"+ str(param["seed"])


        args.extend(["--env_name", param["env"]]) #,"independent_navigation-v0"])
        args.extend(["--n_agents", str(param["n_agents"])])
        args.extend(["--map_shape", str(param["env_size"])])
        args.extend(["--obj_density", str(param["obj_density"])])

        args.extend(["--maac_buffer_length", str(int(1e5))])
        args.extend(["--policy", str(param["policy"])])
        args.extend(["--name", name])
        args.extend(["--maac_use_gpu"])
        args.extend(["--maac_share_actor"])
        args.extend(["--maac_reward_scale", str(param["reward_scale"])])
        args.extend(["--maac_n_rollout_threads", str(param["rollout_threads"])])
        args.extend(["--maac_steps_per_update", str(param["maac_steps_per_update"])])
        args.extend(["--maac_num_updates", str(param["maac_num_updates"])])
        args.extend(["--maac_attend_heads", str(param["maac_attend_heads"])])
        args.extend(["--maac_n_episodes", episodes])
        args.extend(["--maac_batch_size", str(param["maac_batch_size"])])
        args.extend(["--maac_base_policy_type", param["base_policy_type"]]) 
        args.extend(["--benchmark_frequency", str(int(100000))])
        args.extend(["--benchmark_num_episodes", str(int(50))])
        args.extend(["--benchmark_render_length", str(int(25))])
        args.extend(["--render_rate", str(int(50))])

        args.extend(["--maac_gamma", str(param["discount"])])

       # parser.add_argument("--maac_gamma", default=0.9, type=float)

        args.extend(["--seed", str(param["seed"])])
    
        main.main(args)

def maac_2A5():
    episodes = str(100000)
    # experiment_group_name = "TestMaac"
    # work_dir = experiment_group_name
    # plot_dir = experiment_group_name + "_Central_TestBases"

    experiment_group_name = "2A"
    work_dir = '/home/james/Desktop/Gridworld/EXPERIMENTS/' + experiment_group_name
    plot_dir = '/home/james/Desktop/Gridworld/CENTRAL_TENSORBOARD/' + experiment_group_name #+ "_Central"
    #work_dir = experiment_group_name
    #plot_dir = experiment_group_name + "_Central"
 
    parmeter_grid1 = {
        "seed": [1],
        #"batch_sizes": [500,1000,2000],
       # "entropy_coeff":[0.0],
        #"discount":[0.1,0.5,0.9,0.95],
        "discount":[0.9],
        "rollout_threads": [1],
        "reward_scale": [10],
        "policy": ["MAAC"],
        "base_policy_type": ["mlp"],
        "n_agents": [3],
        "obj_density": [0.4],
        "env_size": [5],
        ###
        "maac_steps_per_update":[100],
        "maac_num_updates":[4], #was 1 in experiment
        "maac_attend_heads":[4],
        "maac_batch_size": [1024],
        "env": ["independent_navigation-v4_2"]
    }

    grid1 = ParameterGrid(parmeter_grid1)
    
    for param in grid1:
        args = []
        args = ["--working_directory", work_dir,"--alternative_plot_dir", plot_dir]

        name = "MAAC_" + param["env"] \
            + "_disc_" + str(param["discount"]) \
            + "_rewardscale_" + str(param["reward_scale"]) \
            + "_minibatch_" + str(param["maac_batch_size"]) \
            + "_nupdates_" + str(param["maac_steps_per_update"]) \
            + "_attheads_" + str(param["maac_attend_heads"]) \
            + "_envsize_" + str(param["env_size"]) \
            + "_nagents_"+ str(param["n_agents"]) \
            + "_objdensity_"+ str(param["obj_density"]) \
            + "_seed_"+ str(param["seed"])


        args.extend(["--env_name", param["env"]]) #,"independent_navigation-v0"])
        args.extend(["--n_agents", str(param["n_agents"])])
        args.extend(["--map_shape", str(param["env_size"])])
        args.extend(["--obj_density", str(param["obj_density"])])

        args.extend(["--maac_buffer_length", str(int(1e5))])
        args.extend(["--policy", str(param["policy"])])
        args.extend(["--name", name])
        args.extend(["--maac_use_gpu"])
        args.extend(["--maac_share_actor"])
        args.extend(["--maac_reward_scale", str(param["reward_scale"])])
        args.extend(["--maac_n_rollout_threads", str(param["rollout_threads"])])
        args.extend(["--maac_steps_per_update", str(param["maac_steps_per_update"])])
        args.extend(["--maac_num_updates", str(param["maac_num_updates"])])
        args.extend(["--maac_attend_heads", str(param["maac_attend_heads"])])
        args.extend(["--maac_n_episodes", episodes])
        args.extend(["--maac_batch_size", str(param["maac_batch_size"])])
        args.extend(["--maac_base_policy_type", param["base_policy_type"]]) 
        args.extend(["--benchmark_frequency", str(int(100000))])
        args.extend(["--benchmark_num_episodes", str(int(50))])
        args.extend(["--benchmark_render_length", str(int(25))])
        args.extend(["--render_rate", str(int(50))])

        args.extend(["--maac_gamma", str(param["discount"])])

       # parser.add_argument("--maac_gamma", default=0.9, type=float)

        args.extend(["--seed", str(param["seed"])])
    
        main.main(args)

def maac_2A6():
    episodes = str(100000)
    # experiment_group_name = "TestMaac"
    # work_dir = experiment_group_name
    # plot_dir = experiment_group_name + "_Central_TestBases"

    experiment_group_name = "2A"
    work_dir = '/home/james/Desktop/Gridworld/EXPERIMENTS/' + experiment_group_name
    plot_dir = '/home/james/Desktop/Gridworld/CENTRAL_TENSORBOARD/' + experiment_group_name #+ "_Central"
    #work_dir = experiment_group_name
    #plot_dir = experiment_group_name + "_Central"
 
    parmeter_grid1 = {
        "seed": [1],
        #"batch_sizes": [500,1000,2000],
       # "entropy_coeff":[0.0],
        #"discount":[0.1,0.5,0.9,0.95],
        "discount":[0.9],
        "rollout_threads": [1],
        "reward_scale": [10],
        "policy": ["MAAC"],
        "base_policy_type": ["mlp"],
        "n_agents": [3],
        "obj_density": [0.4],
        "env_size": [5],
        ###
        "maac_steps_per_update":[100],
        "maac_num_updates":[4],
        "maac_attend_heads":[4],
        "maac_batch_size": [1024],
        "env": ["independent_navigation-v4_3"]
    }

    grid1 = ParameterGrid(parmeter_grid1)
    
    for param in grid1:
        args = []
        args = ["--working_directory", work_dir,"--alternative_plot_dir", plot_dir]

        name = "MAAC_" + param["env"] \
            + "_disc_" + str(param["discount"]) \
            + "_rewardscale_" + str(param["reward_scale"]) \
            + "_minibatch_" + str(param["maac_batch_size"]) \
            + "_nupdates_" + str(param["maac_steps_per_update"]) \
            + "_attheads_" + str(param["maac_attend_heads"]) \
            + "_envsize_" + str(param["env_size"]) \
            + "_nagents_"+ str(param["n_agents"]) \
            + "_objdensity_"+ str(param["obj_density"]) \
            + "_seed_"+ str(param["seed"])


        args.extend(["--env_name", param["env"]]) #,"independent_navigation-v0"])
        args.extend(["--n_agents", str(param["n_agents"])])
        args.extend(["--map_shape", str(param["env_size"])])
        args.extend(["--obj_density", str(param["obj_density"])])

        args.extend(["--maac_buffer_length", str(int(1e5))])
        args.extend(["--policy", str(param["policy"])])
        args.extend(["--name", name])
        args.extend(["--maac_use_gpu"])
        args.extend(["--maac_share_actor"])
        args.extend(["--maac_reward_scale", str(param["reward_scale"])])
        args.extend(["--maac_n_rollout_threads", str(param["rollout_threads"])])
        args.extend(["--maac_steps_per_update", str(param["maac_steps_per_update"])])
        args.extend(["--maac_num_updates", str(param["maac_num_updates"])])
        args.extend(["--maac_attend_heads", str(param["maac_attend_heads"])])
        args.extend(["--maac_n_episodes", episodes])
        args.extend(["--maac_batch_size", str(param["maac_batch_size"])])
        args.extend(["--maac_base_policy_type", param["base_policy_type"]]) 
        args.extend(["--benchmark_frequency", str(int(100000))])
        args.extend(["--benchmark_num_episodes", str(int(50))])
        args.extend(["--benchmark_render_length", str(int(25))])
        args.extend(["--render_rate", str(int(50))])

        args.extend(["--maac_gamma", str(param["discount"])])

       # parser.add_argument("--maac_gamma", default=0.9, type=float)

        args.extend(["--seed", str(param["seed"])])
    
        main.main(args)

def maac_test2():
    episodes = str(30000)
    # experiment_group_name = "TestMaac"
    # work_dir = experiment_group_name
    # plot_dir = experiment_group_name + "_Central_TestBases"

    experiment_group_name = "TestMAAC_CNN"
    work_dir = '/home/james/Desktop/Gridworld/EXPERIMENTS/' + experiment_group_name
    plot_dir = '/home/james/Desktop/Gridworld/CENTRAL_TENSORBOARD/' + experiment_group_name #+ "_Central"
    #work_dir = experiment_group_name
    #plot_dir = experiment_group_name + "_Central"
 
    parmeter_grid1 = {
        "seed": [1],
        #"batch_sizes": [500,1000,2000],
       # "entropy_coeff":[0.0],
        #"discount":[0.1,0.5,0.9,0.95],
        "discount":[0.9],
        "rollout_threads": [1],
        "reward_scale": [10],
        "policy": ["MAAC"],
        "base_policy_type": ["primal6"],
        "n_agents": [4],
        "obj_density": [0.2],
        "env_size": [5],
        ###
        "maac_steps_per_update":[100],
        "maac_num_updates":[4],
        "maac_attend_heads":[4],
        "maac_batch_size": [1024],
        "env": ["independent_navigation-v0"]
    }

    grid1 = ParameterGrid(parmeter_grid1)
    
    for param in grid1:
        args = []
        args = ["--working_directory", work_dir,"--alternative_plot_dir", plot_dir]

        name = "MAAC_" + param["env"] \
            + "_disc_" + str(param["discount"]) \
            + "_rewardscale_" + str(param["reward_scale"]) \
            + "_minibatch_" + str(param["maac_batch_size"]) \
            + "_nupdates_" + str(param["maac_steps_per_update"]) \
            + "_attheads_" + str(param["maac_attend_heads"]) \
            + "_envsize_" + str(param["env_size"]) \
            + "_nagents_"+ str(param["n_agents"]) \
            + "_objdensity_"+ str(param["obj_density"]) \
            + "_policytype_"+ str(param["base_policy_type"]) \
            + "_seed_"+ str(param["seed"])


        args.extend(["--env_name", param["env"]]) #,"independent_navigation-v0"])
        args.extend(["--n_agents", str(param["n_agents"])])
        args.extend(["--map_shape", str(param["env_size"])])
        args.extend(["--obj_density", str(param["obj_density"])])

        args.extend(["--maac_buffer_length", str(int(1e5))])
        args.extend(["--policy", str(param["policy"])])
        args.extend(["--name", name])
        args.extend(["--maac_use_gpu"])
        args.extend(["--maac_share_actor"])
        args.extend(["--maac_reward_scale", str(param["reward_scale"])])
        args.extend(["--maac_n_rollout_threads", str(param["rollout_threads"])])
        args.extend(["--maac_steps_per_update", str(param["maac_steps_per_update"])])
        args.extend(["--maac_num_updates", str(param["maac_num_updates"])])
        args.extend(["--maac_attend_heads", str(param["maac_attend_heads"])])
        args.extend(["--maac_n_episodes", episodes])
        args.extend(["--maac_batch_size", str(param["maac_batch_size"])])
        args.extend(["--maac_base_policy_type", param["base_policy_type"]]) 
        args.extend(["--benchmark_frequency", str(int(100000))])
        args.extend(["--benchmark_num_episodes", str(int(50))])
        args.extend(["--benchmark_render_length", str(int(25))])
        args.extend(["--render_rate", str(int(50))])

        args.extend(["--maac_gamma", str(param["discount"])])

       # parser.add_argument("--maac_gamma", default=0.9, type=float)

        args.extend(["--seed", str(param["seed"])])
    
        main.main(args)



def maac_2A2_1():
    episodes = str(30000)
    # experiment_group_name = "TestMaac"
    # work_dir = experiment_group_name
    # plot_dir = experiment_group_name + "_Central_TestBases"

    experiment_group_name = "2A2"
    work_dir = '/home/james/Desktop/Gridworld/EXPERIMENTS/' + experiment_group_name
    plot_dir = '/home/james/Desktop/Gridworld/CENTRAL_TENSORBOARD/' + experiment_group_name #+ "_Central"
    #work_dir = experiment_group_name
    #plot_dir = experiment_group_name + "_Central"
 
    parmeter_grid1 = {
        "seed": [1],
        #"batch_sizes": [500,1000,2000],
       # "entropy_coeff":[0.0],
        #"discount":[0.1,0.5,0.9,0.95],
        "discount":[0.9],
        "rollout_threads": [1],
        "reward_scale": [10],
        "policy": ["MAAC"],
        "base_policy_type": ["primal6"],
        "n_agents": [4],
        "obj_density": [0.2],
        "env_size": [5],
        ###
        "maac_steps_per_update":[100],
        "maac_num_updates":[1],
        "maac_attend_heads":[4],
        "maac_batch_size": [1024],
        "env": ["independent_navigation-v0"],
        "step_r": [-0.1], #[-0.1, -0.4],
        "obstacle_collision_r": [-0.4], #[-0.015],
        "agent_collision_r":  [-0.4], #[-1.0],
        "goal_reached_r": [1.0], #[0.1],
        "finish_episode_r": [0.0]
    }

    grid1 = ParameterGrid(parmeter_grid1)
    
    for param in grid1:
        args = []
        args = ["--working_directory", work_dir,"--alternative_plot_dir", plot_dir]

        name = "2A2_maac_" \
            + "_arc_" + param["base_policy_type"] \
            + "_sr_" + str(param["step_r"]) \
            + "_ocr_" + str(param["obstacle_collision_r"]) \
            + "_acr_" + str(param["agent_collision_r"]) \
            + "_grr_" + str(param["goal_reached_r"]) \
            + "_fer_" + str(param["finish_episode_r"]) \
            + "_disc_" + str(param["discount"]) \
            + "_rewardscale_" + str(param["reward_scale"]) \
            + "_minibatch_" + str(param["maac_batch_size"]) \
            + "_nupdates_" + str(param["maac_steps_per_update"]) \
            + "_attheads_" + str(param["maac_attend_heads"]) \
            + "_envsize_" + str(param["env_size"]) \
            + "_nagents_"+ str(param["n_agents"]) \
            + "_objdensity_"+ str(param["obj_density"]) \
            + "_seed_"+ str(param["seed"])


        args.extend(["--env_name", param["env"]]) #,"independent_navigation-v0"])
        args.extend(["--n_agents", str(param["n_agents"])])
        args.extend(["--map_shape", str(param["env_size"])])
        args.extend(["--obj_density", str(param["obj_density"])])

        args.extend(["--use_custom_rewards"])
        args.extend(["--step_r", str(param["step_r"])])
        args.extend(["--obstacle_collision_r", str(param["obstacle_collision_r"])])
        args.extend(["--agent_collision_r", str(param["agent_collision_r"])])
        args.extend(["--goal_reached_r", str(param["goal_reached_r"])])
        args.extend(["--finish_episode_r", str(param["finish_episode_r"])])

        args.extend(["--maac_buffer_length", str(int(1e5))])
        args.extend(["--policy", str(param["policy"])])
        args.extend(["--name", name])
        args.extend(["--maac_use_gpu"])
        args.extend(["--maac_share_actor"])
        args.extend(["--maac_reward_scale", str(param["reward_scale"])])
        args.extend(["--maac_n_rollout_threads", str(param["rollout_threads"])])
        args.extend(["--maac_steps_per_update", str(param["maac_steps_per_update"])])
        args.extend(["--maac_num_updates", str(param["maac_num_updates"])])
        args.extend(["--maac_attend_heads", str(param["maac_attend_heads"])])
        args.extend(["--maac_n_episodes", episodes])
        args.extend(["--maac_batch_size", str(param["maac_batch_size"])])
        args.extend(["--maac_base_policy_type", param["base_policy_type"]]) 
        args.extend(["--benchmark_frequency", str(int(15000))])
        args.extend(["--benchmark_num_episodes", str(int(200))])
        args.extend(["--benchmark_render_length", str(int(25))])
        args.extend(["--render_rate", str(int(200))])
        args.extend(["--checkpoint_frequency", str(int(5000))])

        args.extend(["--maac_gamma", str(param["discount"])])

       # parser.add_argument("--maac_gamma", default=0.9, type=float)

        args.extend(["--seed", str(param["seed"])])
    
        main.main(args)

def maac_2A2_2():
    episodes = str(150000)
    # experiment_group_name = "TestMaac"
    # work_dir = experiment_group_name
    # plot_dir = experiment_group_name + "_Central_TestBases"

    experiment_group_name = "2A2"
    work_dir = '/home/james/Desktop/Gridworld/EXPERIMENTS/' + experiment_group_name
    plot_dir = '/home/james/Desktop/Gridworld/CENTRAL_TENSORBOARD/' + experiment_group_name #+ "_Central"
    #work_dir = experiment_group_name
    #plot_dir = experiment_group_name + "_Central"
 
    parmeter_grid1 = {
        "seed": [1],
        #"batch_sizes": [500,1000,2000],
       # "entropy_coeff":[0.0],
        #"discount":[0.1,0.5,0.9,0.95],
        "discount":[0.9],
        "rollout_threads": [1],
        "reward_scale": [10],
        "policy": ["MAAC"],
        "base_policy_type": ["primal6"],
        "n_agents": [4],
        "obj_density": [0.2],
        "env_size": [5],
        ###
        "maac_steps_per_update":[100],
        "maac_num_updates":[1],
        "maac_attend_heads":[4],
        "maac_batch_size": [1024],
        "env": ["independent_navigation-v0"],
        "step_r": [-0.1], #[-0.1, -0.4],
        "obstacle_collision_r": [-0.4], #[-0.015],
        "agent_collision_r":  [-0.4], #[-1.0],
        "goal_reached_r": [0.1], #[0.1],
        "finish_episode_r": [2.0]
    }

    grid1 = ParameterGrid(parmeter_grid1)
    
    for param in grid1:
        args = []
        args = ["--working_directory", work_dir,"--alternative_plot_dir", plot_dir]

        name = "2A2_maac_mixedr" \
            + "_arc_" + param["base_policy_type"] \
            + "_sr_" + str(param["step_r"]) \
            + "_ocr_" + str(param["obstacle_collision_r"]) \
            + "_acr_" + str(param["agent_collision_r"]) \
            + "_grr_" + str(param["goal_reached_r"]) \
            + "_fer_" + str(param["finish_episode_r"]) \
            + "_disc_" + str(param["discount"]) \
            + "_rewardscale_" + str(param["reward_scale"]) \
            + "_minibatch_" + str(param["maac_batch_size"]) \
            + "_nupdates_" + str(param["maac_steps_per_update"]) \
            + "_attheads_" + str(param["maac_attend_heads"]) \
            + "_envsize_" + str(param["env_size"]) \
            + "_nagents_"+ str(param["n_agents"]) \
            + "_objdensity_"+ str(param["obj_density"]) \
            + "_seed_"+ str(param["seed"])


        args.extend(["--env_name", param["env"]]) #,"independent_navigation-v0"])
        args.extend(["--n_agents", str(param["n_agents"])])
        args.extend(["--map_shape", str(param["env_size"])])
        args.extend(["--obj_density", str(param["obj_density"])])

        args.extend(["--use_custom_rewards"])
        args.extend(["--step_r", str(param["step_r"])])
        args.extend(["--obstacle_collision_r", str(param["obstacle_collision_r"])])
        args.extend(["--agent_collision_r", str(param["agent_collision_r"])])
        args.extend(["--goal_reached_r", str(param["goal_reached_r"])])
        args.extend(["--finish_episode_r", str(param["finish_episode_r"])])

        args.extend(["--maac_buffer_length", str(int(1e5))])
        args.extend(["--policy", str(param["policy"])])
        args.extend(["--name", name])
        args.extend(["--maac_use_gpu"])
        args.extend(["--maac_share_actor"])
        args.extend(["--maac_reward_scale", str(param["reward_scale"])])
        args.extend(["--maac_n_rollout_threads", str(param["rollout_threads"])])
        args.extend(["--maac_steps_per_update", str(param["maac_steps_per_update"])])
        args.extend(["--maac_num_updates", str(param["maac_num_updates"])])
        args.extend(["--maac_attend_heads", str(param["maac_attend_heads"])])
        args.extend(["--maac_n_episodes", episodes])
        args.extend(["--maac_batch_size", str(param["maac_batch_size"])])
        args.extend(["--maac_base_policy_type", param["base_policy_type"]]) 
        args.extend(["--benchmark_frequency", str(int(15000000))])
        args.extend(["--benchmark_num_episodes", str(int(200))])
        args.extend(["--benchmark_render_length", str(int(25))])
        args.extend(["--render_rate", str(int(200))])
        args.extend(["--checkpoint_frequency", str(int(5000))])

        args.extend(["--maac_gamma", str(param["discount"])])

       # parser.add_argument("--maac_gamma", default=0.9, type=float)

        args.extend(["--seed", str(param["seed"])])
    
        main.main(args)


def maac_2A2_3():
    episodes = str(100000)
    # experiment_group_name = "TestMaac"
    # work_dir = experiment_group_name
    # plot_dir = experiment_group_name + "_Central_TestBases"

    experiment_group_name = "2A2"
    work_dir = '/home/james/Desktop/Gridworld/EXPERIMENTS/' + experiment_group_name
    plot_dir = '/home/james/Desktop/Gridworld/CENTRAL_TENSORBOARD/' + experiment_group_name #+ "_Central"
    #work_dir = experiment_group_name
    #plot_dir = experiment_group_name + "_Central"
 
    parmeter_grid1 = {
        "seed": [1],
        #"batch_sizes": [500,1000,2000],
       # "entropy_coeff":[0.0],
        #"discount":[0.1,0.5,0.9,0.95],
        "discount":[0.9],
        "rollout_threads": [1],
        "reward_scale": [10],
        "policy": ["MAAC"],
        "base_policy_type": ["primal6"],
        "n_agents": [4],
        "obj_density": [0.2],
        "env_size": [5],
        ###
        "maac_steps_per_update":[100],
        "maac_num_updates":[1],
        "maac_attend_heads":[4],
        "maac_batch_size": [1024],
        "env": ["independent_navigation-v5_1","independent_navigation-v5_2"]
        # "step_r": [-0.1, -0.4],
        # "obstacle_collision_r": [-0.015],
        # "agent_collision_r": [-0.4],
        # "goal_reached_r": [0.1],
        # "finish_episode_r": [0.0]
    }

    grid1 = ParameterGrid(parmeter_grid1)
    
    for param in grid1:
        args = []
        args = ["--working_directory", work_dir,"--alternative_plot_dir", plot_dir]

        name = "2A2_maac_globalr_" \
            + "_arc_" + param["base_policy_type"] \
            + "_env_" + str(param["env"]) \
            + "_disc_" + str(param["discount"]) \
            + "_rewardscale_" + str(param["reward_scale"]) \
            + "_minibatch_" + str(param["maac_batch_size"]) \
            + "_nupdates_" + str(param["maac_steps_per_update"]) \
            + "_attheads_" + str(param["maac_attend_heads"]) \
            + "_envsize_" + str(param["env_size"]) \
            + "_nagents_"+ str(param["n_agents"]) \
            + "_objdensity_"+ str(param["obj_density"]) \
            + "_seed_"+ str(param["seed"])
            # + "_sr_" + str(param["step_r"]) \
            # + "_ocr_" + str(param["obstacle_collision_r"]) \
            # + "_acr_" + str(param["agent_collision_r"]) \
            # + "_grr_" + str(param["goal_reached_r"]) \
            # + "_fer_" + str(param["finish_episode_r"]) \

        args.extend(["--env_name", param["env"]]) #,"independent_navigation-v0"])
        args.extend(["--n_agents", str(param["n_agents"])])
        args.extend(["--map_shape", str(param["env_size"])])
        args.extend(["--obj_density", str(param["obj_density"])])

     #   args.extend(["--use_custom_rewards"])
        # args.extend(["--step_r", str(param["step_r"])])
        # args.extend(["--obstacle_collision_r", str(param["obstacle_collision_r"])])
        # args.extend(["--agent_collision_r", str(param["agent_collision_r"])])
        # args.extend(["--goal_reached_r", str(param["goal_reached_r"])])
        # args.extend(["--finish_episode_r", str(param["finish_episode_r"])])

        args.extend(["--maac_buffer_length", str(int(1e5))])
        args.extend(["--policy", str(param["policy"])])
        args.extend(["--name", name])
        args.extend(["--maac_use_gpu"])
        args.extend(["--maac_share_actor"])
        args.extend(["--maac_reward_scale", str(param["reward_scale"])])
        args.extend(["--maac_n_rollout_threads", str(param["rollout_threads"])])
        args.extend(["--maac_steps_per_update", str(param["maac_steps_per_update"])])
        args.extend(["--maac_num_updates", str(param["maac_num_updates"])])
        args.extend(["--maac_attend_heads", str(param["maac_attend_heads"])])
        args.extend(["--maac_n_episodes", episodes])
        args.extend(["--maac_batch_size", str(param["maac_batch_size"])])
        args.extend(["--maac_base_policy_type", param["base_policy_type"]]) 
        args.extend(["--benchmark_frequency", str(int(15000000))])
        args.extend(["--benchmark_num_episodes", str(int(200))])
        args.extend(["--benchmark_render_length", str(int(25))])
        args.extend(["--render_rate", str(int(200))])
        args.extend(["--checkpoint_frequency", str(int(5000))])

        args.extend(["--maac_gamma", str(param["discount"])])

       # parser.add_argument("--maac_gamma", default=0.9, type=float)

        args.extend(["--seed", str(param["seed"])])
    
        main.main(args)


def maac_2A2_4():
    episodes = str(150000)
    # experiment_group_name = "TestMaac"
    # work_dir = experiment_group_name
    # plot_dir = experiment_group_name + "_Central_TestBases"

    experiment_group_name = "2A2"
    work_dir = '/home/james/Desktop/Gridworld/EXPERIMENTS/' + experiment_group_name
    plot_dir = '/home/james/Desktop/Gridworld/CENTRAL_TENSORBOARD/' + experiment_group_name #+ "_Central"
    #work_dir = experiment_group_name
    #plot_dir = experiment_group_name + "_Central"
 
    parmeter_grid1 = {
        "seed": [1],
        #"batch_sizes": [500,1000,2000],
       # "entropy_coeff":[0.0],
        #"discount":[0.1,0.5,0.9,0.95],
        "discount":[0.9],
        "rollout_threads": [1],
        "reward_scale": [10],
        "policy": ["MAAC"],
        "base_policy_type": ["primal6"],
        "n_agents": [4],
        "obj_density": [0.2],
        "env_size": [5],
        ###
        "maac_steps_per_update":[100],
        "maac_num_updates":[1],
        "maac_attend_heads":[4],
        "maac_batch_size": [1024],
        "env": ["independent_navigation-v7_1"]
        # "step_r": [-0.1, -0.4],
        # "obstacle_collision_r": [-0.015],
        # "agent_collision_r": [-0.4],
        # "goal_reached_r": [0.1],
        # "finish_episode_r": [0.0]
    }

    grid1 = ParameterGrid(parmeter_grid1)
    
    for param in grid1:
        args = []
        args = ["--working_directory", work_dir,"--alternative_plot_dir", plot_dir]

        name = "2A2_maac_mixedr_" \
            + "_arc_" + param["base_policy_type"] \
            + "_env_" + str(param["env"]) \
            + "_disc_" + str(param["discount"]) \
            + "_rewardscale_" + str(param["reward_scale"]) \
            + "_minibatch_" + str(param["maac_batch_size"]) \
            + "_nupdates_" + str(param["maac_steps_per_update"]) \
            + "_attheads_" + str(param["maac_attend_heads"]) \
            + "_envsize_" + str(param["env_size"]) \
            + "_nagents_"+ str(param["n_agents"]) \
            + "_objdensity_"+ str(param["obj_density"]) \
            + "_seed_"+ str(param["seed"])
            # + "_sr_" + str(param["step_r"]) \
            # + "_ocr_" + str(param["obstacle_collision_r"]) \
            # + "_acr_" + str(param["agent_collision_r"]) \
            # + "_grr_" + str(param["goal_reached_r"]) \
            # + "_fer_" + str(param["finish_episode_r"]) \

        args.extend(["--env_name", param["env"]]) #,"independent_navigation-v0"])
        args.extend(["--n_agents", str(param["n_agents"])])
        args.extend(["--map_shape", str(param["env_size"])])
        args.extend(["--obj_density", str(param["obj_density"])])

     #   args.extend(["--use_custom_rewards"])
        # args.extend(["--step_r", str(param["step_r"])])
        # args.extend(["--obstacle_collision_r", str(param["obstacle_collision_r"])])
        # args.extend(["--agent_collision_r", str(param["agent_collision_r"])])
        # args.extend(["--goal_reached_r", str(param["goal_reached_r"])])
        # args.extend(["--finish_episode_r", str(param["finish_episode_r"])])

        args.extend(["--maac_buffer_length", str(int(1e5))])
        args.extend(["--policy", str(param["policy"])])
        args.extend(["--name", name])
        args.extend(["--maac_use_gpu"])
        args.extend(["--maac_share_actor"])
        args.extend(["--maac_reward_scale", str(param["reward_scale"])])
        args.extend(["--maac_n_rollout_threads", str(param["rollout_threads"])])
        args.extend(["--maac_steps_per_update", str(param["maac_steps_per_update"])])
        args.extend(["--maac_num_updates", str(param["maac_num_updates"])])
        args.extend(["--maac_attend_heads", str(param["maac_attend_heads"])])
        args.extend(["--maac_n_episodes", episodes])
        args.extend(["--maac_batch_size", str(param["maac_batch_size"])])
        args.extend(["--maac_base_policy_type", param["base_policy_type"]]) 
        args.extend(["--benchmark_frequency", str(int(300000))])
        args.extend(["--benchmark_num_episodes", str(int(200))])
        args.extend(["--benchmark_render_length", str(int(25))])
        args.extend(["--render_rate", str(int(200))])
        args.extend(["--checkpoint_frequency", str(int(5000))])

        args.extend(["--maac_gamma", str(param["discount"])])

       # parser.add_argument("--maac_gamma", default=0.9, type=float)

        args.extend(["--seed", str(param["seed"])])
    
        main.main(args)

def maac_4_2():
    episodes = str(100000)
    # experiment_group_name = "TestMaac"
    # work_dir = experiment_group_name
    # plot_dir = experiment_group_name + "_Central_TestBases"

    experiment_group_name = "4_2"
    work_dir = '/home/james/Desktop/Gridworld/EXPERIMENTS/' + experiment_group_name
    plot_dir = '/home/james/Desktop/Gridworld/CENTRAL_TENSORBOARD/' + experiment_group_name #+ "_Central"
    #work_dir = experiment_group_name
    #plot_dir = experiment_group_name + "_Central"
 
    parmeter_grid1 = {
        "seed": [1],
        #"batch_sizes": [500,1000,2000],
       # "entropy_coeff":[0.0],
        #"discount":[0.1,0.5,0.9,0.95],
        "discount":[0.9],
        "rollout_threads": [1],
        "reward_scale": [10],
        "policy": ["MAAC"],
        "base_policy_type": ["primal7"],
        "n_agents": [2,4,8,16],
        "obj_density": [0.0],
        "env_size": [7],
        ###
        "maac_steps_per_update":[100],
        "maac_num_updates":[1],
        "maac_attend_heads":[4],
        "maac_batch_size": [1024],
        "env": ["independent_navigation-v0"],
        "step_r": [-0.1], #[-0.01, -0.1, -0.4],
        "obstacle_collision_r": [-0.4], #[-0.015],
        "agent_collision_r": [-0.4], #[-1.0], 
        "goal_reached_r": [0.5], #[0.1],
        "finish_episode_r": [2.0]
    }

    grid1 = ParameterGrid(parmeter_grid1)
    
    for param in grid1:
        args = []
        args = ["--working_directory", work_dir,"--alternative_plot_dir", plot_dir]

        name = "4_2_maac" \
            + "_arc_" + param["base_policy_type"] \
            + "_env_" + str(param["env"]) \
            + "_disc_" + str(param["discount"]) \
            + "_rewardscale_" + str(param["reward_scale"]) \
            + "_minibatch_" + str(param["maac_batch_size"]) \
            + "_nupdates_" + str(param["maac_steps_per_update"]) \
            + "_attheads_" + str(param["maac_attend_heads"]) \
            + "_envsize_" + str(param["env_size"]) \
            + "_nagents_"+ str(param["n_agents"]) \
            + "_objdensity_"+ str(param["obj_density"]) \
            + "_sr_" + str(param["step_r"]) \
            + "_ocr_" + str(param["obstacle_collision_r"]) \
            + "_acr_" + str(param["agent_collision_r"]) \
            + "_grr_" + str(param["goal_reached_r"]) \
            + "_fer_" + str(param["finish_episode_r"]) \
            + "_seed_"+ str(param["seed"])

        args.extend(["--env_name", param["env"]]) #,"independent_navigation-v0"])
        args.extend(["--n_agents", str(param["n_agents"])])
        args.extend(["--map_shape", str(param["env_size"])])
        args.extend(["--obj_density", str(param["obj_density"])])

        args.extend(["--use_custom_rewards"])
        args.extend(["--step_r", str(param["step_r"])])
        args.extend(["--obstacle_collision_r", str(param["obstacle_collision_r"])])
        args.extend(["--agent_collision_r", str(param["agent_collision_r"])])
        args.extend(["--goal_reached_r", str(param["goal_reached_r"])])
        args.extend(["--finish_episode_r", str(param["finish_episode_r"])])

        args.extend(["--maac_buffer_length", str(int(1e5))])
        args.extend(["--policy", str(param["policy"])])
        args.extend(["--name", name])
        args.extend(["--maac_use_gpu"])
        args.extend(["--maac_share_actor"])
        args.extend(["--maac_reward_scale", str(param["reward_scale"])])
        args.extend(["--maac_n_rollout_threads", str(param["rollout_threads"])])
        args.extend(["--maac_steps_per_update", str(param["maac_steps_per_update"])])
        args.extend(["--maac_num_updates", str(param["maac_num_updates"])])
        args.extend(["--maac_attend_heads", str(param["maac_attend_heads"])])
        args.extend(["--maac_n_episodes", episodes])
        args.extend(["--maac_batch_size", str(param["maac_batch_size"])])
        args.extend(["--maac_base_policy_type", param["base_policy_type"]]) 
        args.extend(["--benchmark_frequency", str(int(300000))])
        args.extend(["--benchmark_num_episodes", str(int(200))])
        args.extend(["--benchmark_render_length", str(int(25))])
        args.extend(["--render_rate", str(int(200))])
        args.extend(["--checkpoint_frequency", str(int(5000))])

        args.extend(["--maac_gamma", str(param["discount"])])

       # parser.add_argument("--maac_gamma", default=0.9, type=float)

        args.extend(["--seed", str(param["seed"])])
    
        main.main(args)