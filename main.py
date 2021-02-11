'''Some code modified from: https://github.com/IC3Net/IC3Net/blob/master/comm.py '''
import argparse
from Env.make_env import make_env
from Agents.make_policy import make_policy 
from ic3Net_trainer import Trainer
from utils.logger import Logger
import config
import sys
import os
import numpy as np

def main(args):
    parser = argparse.ArgumentParser("Experiment parameters")

    #Environment:
    parser.add_argument("--map_shape", default = 5, type=int)
    parser.add_argument("--n_agents", default = 1, type=int)
    parser.add_argument("--obj_density", default = 0.0, type=float)
    parser.add_argument("--view_d", default = 2, type=int)
    parser.add_argument("--env_name", default = "cooperative_navigation-v0", type = str)
    parser.add_argument("--use_custom_rewards", default = False, action='store_true')
    parser.add_argument("--step_r", default = -10, type=float)
    parser.add_argument("--agent_collision_r", default = -10, type=float)
    parser.add_argument("--obstacle_collision_r", default = -10, type=float)
    parser.add_argument("--goal_reached_r", default = -10, type=float)
    parser.add_argument("--finish_episode_r", default = -10, type=float)
    parser.add_argument("--block_r", default = -10, type=float)
    parser.add_argument("--custom_env_ind", default= -1, type=int)

    #Policy:
    parser.add_argument("--policy", default="TEST", type =str)

    #           PPO Paramares:
    parser.add_argument("--ppo_hidden_dim", default= 120, type=int)
    parser.add_argument("--ppo_lr_a", default= 0.001, type=float)
    parser.add_argument("--ppo_lr_v", default= 0.001, type=float)
    parser.add_argument("--ppo_base_policy_type", default= "mlp", type=str)
    parser.add_argument("--ppo_recurrent", default= False, action='store_true')
    parser.add_argument("--ppo_heur_block", default= False, action='store_true')
    parser.add_argument("--ppo_heur_valid_act", default= False, action='store_true')
    parser.add_argument("--ppo_heur_no_prev_state", default= False, action='store_true')
    parser.add_argument("--ppo_workers", default= 1, type= int)
    parser.add_argument("--ppo_rollout_length", default= 32, type=int)
    parser.add_argument("--ppo_share_actor", default= False,action='store_true')
    parser.add_argument("--ppo_share_value", default= False,action='store_true')
    parser.add_argument("--ppo_iterations", default= 7000, type=int)
    parser.add_argument("--ppo_k_epochs", default= 1, type=int)
    parser.add_argument("--ppo_eps_clip", default= 0.2, type=float)
    parser.add_argument("--ppo_minibatch_size", default= 32, type=int)
    parser.add_argument("--ppo_entropy_coeff", default = 0.01, type=float)
    parser.add_argument("--ppo_value_coeff", default= 0.5, type=float)
    parser.add_argument("--ppo_discount", default= 0.95, type=float)
    parser.add_argument("--ppo_gae_lambda", default=1.0, type=float)
    parser.add_argument("--ppo_use_gpu", default= False, action='store_true') #ppo_curr_n_updates
    parser.add_argument("--ppo_curr_n_updates", default = 5, type=int)
    parser.add_argument("--ppo_bc_iteration_prob", default = 0.0, type=float)
    parser.add_argument("--ppo_continue_from_checkpoint", default = False, action='store_true')

    ################################################################################################
    #           IC3Net: #All parameters from https://github.com/IC3Net/IC3Net/blob/master/comm.py 
    #
    parser.add_argument("--hid_size", default= 120, type =int)
    parser.add_argument("--recurrent", default= False, action = 'store_true')
    parser.add_argument("--detach_gap", default = 10, type = int)
    parser.add_argument("--comm_passes", default= 1, type =int)
    parser.add_argument('--share_weights', default=False, action='store_true', 
                    help='Share weights for hops')
    #parser.add_argument("--comm_mode", default = 1, type = int,
    #help= "if mode == 0 -- no communication; mode==1--ic3net communication; mode ==2 -- commNet communication")
    parser.add_argument("--comm_mode", default = "avg", type = str, help="Average or sum the hidden states to obtain the comm vector")
    parser.add_argument("--hard_attn", default=True,action='store_false', help="to communicate or not. If hard_attn == False, no comm")
    parser.add_argument("--comm_mask_zero", default=False,action='store_true', help="to communicate or not. If hard_attn == False, no comm")
    parser.add_argument('--comm_action_one', default=False, action='store_true',
                    help='Always communicate.')
    parser.add_argument("--ic3_base_policy_type", default= "mlp", type =str)

    ################################################################################################
    #           MAAC parampeters from : https://github.com/shariqiqbal2810/MAAC
    #
    parser.add_argument("--maac_buffer_length", default=int(1e6), type=int)
    parser.add_argument("--maac_n_episodes", default=50000, type=int)
    parser.add_argument("--maac_n_rollout_threads", default=6, type=int)
    parser.add_argument("--maac_steps_per_update", default=100, type=int)
    parser.add_argument("--maac_num_updates", default=4, type=int,
                        help="Number of updates per update cycle")
    parser.add_argument("--maac_batch_size",
                        default=1024, type=int, 
                        help="Batch size for training")
    parser.add_argument("--maac_pol_hidden_dim", default=128, type=int)
    parser.add_argument("--maac_critic_hidden_dim", default=128, type=int)
    parser.add_argument("--maac_attend_heads", default=4, type=int)
    parser.add_argument("--maac_pi_lr", default=0.001, type=float)
    parser.add_argument("--maac_q_lr", default=0.001, type=float)
    parser.add_argument("--maac_tau", default=0.001, type=float)
    parser.add_argument("--maac_gamma", default=0.9, type=float)
    parser.add_argument("--maac_reward_scale", default=100., type=float)
    parser.add_argument("--maac_use_gpu", action='store_true')
    parser.add_argument("--maac_share_actor", action='store_true')
    parser.add_argument("--maac_base_policy_type", default = 'cnn_old') #Types: mlp, cnn_old, cnn_new

    #Training:
    parser.add_argument("--n_workers", default = 1, type=int,
    help="The number of parallel environments sampled from")
    parser.add_argument("--n_steps", default = 1, type = int, 
    help= "For AC type policies")
    #parser.add_argument("--n_steps", default = 1, type = int, 

    parser.add_argument("--device", default= 'cuda', type=str)
    parser.add_argument("--iterations", default = int(3*1e6), type = int)
    parser.add_argument("--lr", default=0.001, type= float)
   # parser.add_argument("--lr_step", default = -1, type=int)
   # parser.add_argument("lr_")
    parser.add_argument("--discount", default = 1.0, type=float)
    parser.add_argument("--lambda_", default= 1.0, type=float)
    parser.add_argument("--value_coeff", default= 0.01, type=float, help="Value function update coefficient")
    parser.add_argument("--entropy_coeff", default= 0.05, type=float, help="Entropy regularization coefficient")
    parser.add_argument("--model", default="mlp", type= str)
    #parser.add_argument("--hidden_size", default= 250, type= int)
    parser.add_argument("--seed", default= 2, type= int)
    #Using same convention/parameters as https://github.com/IC3Net/IC3Net:
    # parser.add_argument('--num_epochs', default=100, type=int,
    #                 help='number of training epochs')
    #parser.add_argument('--iterations', default=100, type=int,
     #   help='number of training epochs')
    #parser.add_argument('--epoch_size', type=int, default=10,
     #                   help='number of update iterations in an epoch')
    parser.add_argument('--batch_size', type=int, default=2000,
                        help='number of steps before each update (per thread)')
    #End here...

    #Saving and rendering
    #parser.add_argument("--working_directory", default='/home/james/Desktop/Gridworld/single_agent', type = str)
    #parser.add_argument("--working_directory", default='/home/james/Desktop/Gridworld/single_agent', type = str)
    parser.add_argument("--working_directory", default='none', type = str)
    #parser.add_argument("--model_path", default = '/home/james/Desktop/Gridworld/Models', type = str) #TODO: Add
    #parser.add_argument("--render_path", default = '/home/james/Desktop/Gridworld/Render', type = str) #TODO: Add functionality to rendering.py
    
    parser.add_argument("--mean_stats", default= 1, type = int, help="The number of iterations over which stats are averaged")
    parser.add_argument("--checkpoint_frequency", default = int(10e3), type=int)
    parser.add_argument("--print_frequency", default = int(5e1), type=int)
    parser.add_argument("--replace_checkpoints", default = True, type=bool)
    parser.add_argument("--render_rate", default= int(5e2 - 10), type=int)
    parser.add_argument("--render_length", default = 7, type= int, help="Number of episodes of rendering to save")
    #parser.add_argument("--note", default = "NO NOTES", type= str, help="Add note to describe experiment")
    parser.add_argument("--name", default="NO_NAME", type=str, help="Experiment name")
    parser.add_argument("--alternative_plot_dir", default="none", help = "Creates a single folder to store tensorboard plots for comparison")
    parser.add_argument("--benchmark_frequency", default= int(3000), type=int, help="Frequency (iterations or episodes) with which to evalue the greedy policy.")
    parser.add_argument("--benchmark_num_episodes", default= int(500), type=int)
    parser.add_argument("--benchmark_render_length", default= int(100), type=int)
    #parser.add_argument("--render_save_episodes", default = 30, type= int, help="The last number of episode renders to save to mp4 file")
    #parser.add_argument("--render_after", default=500, type=int)

   # parser.add_argument("--", type=str)

   # cmd_args = []
    #cmd_args.append(["--working_directory", "/home/james/Desktop/Gridworld/test1", "--alternative-plot_dir", "/home/james/Desktop/Gridworld/CENTRAL_PLOT"])
    #for a in cmd_args:

    args = parser.parse_args(args)
    args.map_shape = (args.map_shape, args.map_shape)

    config.set_global_seed(args.seed)
    env = make_env(args)

    print("Running: \n {}".format(env.summary()))
    if args.policy == 'IC3':
        policy = make_policy(args, env)

        trainer = Trainer(args, policy, env)

        logger = Logger(args, env.summary(), policy.summary(), policy)

        #print("Base policy type: {}".format(args.ic3_base_policy_type))
        for iteration in range(args.iterations):
            print("IC3 iteration: {} of {}".format(iteration, args.iterations))
            batch, stats, render_frames = trainer.sample_batch(render=logger.should_render())
            stats["value_loss"], stats["action_loss"] = policy.update(batch)

            stats["iterations"] = iteration
            terminal_t_info = [inf for i,inf in enumerate(batch.misc) if inf["terminate"]]
            
            avg_comm = np.average([np.average(inf['comm_action']) for inf in batch.misc])
            logger.log(stats, terminal_t_info, render_frames, commActions=avg_comm)

            if iteration % args.checkpoint_frequency==0 and iteration != 0:
                path = logger.checkpoint_dir + "/checkpoint_"+ str(iteration)
                policy.save(path) 

            if iteration % args.benchmark_frequency == 0 and iteration != 0:
                trainer.benchmark_ic3(policy, logger, args.benchmark_num_episodes, args.benchmark_render_length, iteration)
        #Benchmark
        trainer.benchmark_ic3(policy, logger, args.benchmark_num_episodes, args.benchmark_render_length, iteration)
        #Checkpoint
        path = logger.checkpoint_dir + "/checkpoint_"+ str(args.iterations)
        policy.save(path)
    elif args.policy == 'MAAC':
        from utils.logger import Maac_Logger
        from Agents.MAAC import run
        logger = Maac_Logger(args)
        run(args, logger)
    elif args.policy == 'A2C':
        raise NotImplementedError 
        # from Agents.Ind_A2C import run_a2c
        #run_a2c(args)
    elif args.policy == 'PPO':
        from Agents.PPO import run
        run(args)
    elif args.policy == 'CURR_PPO':
        from Agents.PPO_CurriculumTrain import run
        run(args)
    else:
        raise Exception("Policy type not implemented")

if __name__ == "__main__":
    main(sys.argv[1:])
    
            



