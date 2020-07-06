from sklearn.model_selection import ParameterGrid
from A2C2 import main

def a2c2_test_p1():
    n_episodes = 100000
    experiment_group_name = "A2C2_CN"
    work_dir = "EXPERIMENTS/" + experiment_group_name
    plot_dir = "CENTRAL_TENSORBOARD/" + experiment_group_name

 
    parmeter_grid1 = {
        "env_name": ["cooperative_navigation-v1"],
        "n_agents": [2, 4],
        "network_decoder_type": ["mlp"],
        "comm_zero": [False],
        "share_critic": [False],
        "share_actor": [True], #[False],
        "share_comm_network": [False]
    }


    grid1 = ParameterGrid(parmeter_grid1)
    
    for param in grid1:
        args = []
        args = ["--working_directory", work_dir, "--alternative_plot_dir", plot_dir]
        args.extend(["--n_episodes", str(n_episodes)])
       # args.extend(["--n_workers", str(workers)])
       # args.extend(["--map_shape", tuple((5,5))])
        name = "CN_"+ param["env_name"][-2:] \
            + "_agnts" + str(param["n_agents"]) + "_PS_cr_" + str(param["share_critic"]) \
            + "_PS_ac_" + str(param["share_actor"]) + "_PS_cNet_" + str(param["share_comm_network"]) \
            + "_comm_zero_" + str(param["comm_zero"])

        args.extend(["--n_agents", str(param["n_agents"])])
        args.extend(["--name", name])
        args.extend(["--env_name", param["env_name"]])
        args.extend(["--network_decoder_type", param["network_decoder_type"]])
        if param["share_critic"]: args.extend(["--share_critic"])
        if param["share_actor"]: args.extend(["--share_actor"])
        if param["share_comm_network"]: args.extend(["--share_comm_network"])
       # args.extend(["--share_actor", param["share_actor"]])
       # args.extend(["--share_comm_network", param["share_comm_network"]])
        if param["comm_zero"]: args.extend(["--comm_zero"])
        #args.extend(["--comm_zero", param["comm_zero"]])

        #print("param[share_critic] is: {}  type is {}".format(param["share_critic"], type(param["share_critic"])))

        #print("Args in test_a2c2 {}".format(args))
    
        main(args)

if __name__ == "__main__":
    a2c2_test_p1()
