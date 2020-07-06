import argparse

import Experiments.experiments as exp
import Experiments.maac_experiments as exp2
import Experiments.test_a2c2_ex as exp3
import Experiments.test_maac as exp4
import Experiments.test_ic3 as exp5
import Experiments.cn_Maac_Ic3 as exp6
import Experiments.test_a2c as exp7
import Experiments.test_ppo as exp8
import Experiments.exp_ppo_curr as exp9
import Experiments.ppo_hyp_param as exp10
import Experiments.test_ppo2 as exp11

if __name__ == "__main__":
    #parameter_tuning()
    #test()
    #multi_agent_comparison_1()
   # multi_agent_comparison_2()
    #single_agent_comp_shared_value_f()
    #singel_agentP0_3_part1()
    #singel_agentP0_3_part2()
    #singel_agentP0_3_part3()
    #singel_agentP0_3_1()
    #test3()

    parser = argparse.ArgumentParser("Run Experiment")
    parser.add_argument("--name", type=str)
    #args = parser.parse_args()
    #args = parser.parse_args(["--name", "cn_maac_ic3"])
 #   args =  parser.parse_args(["--name", "test_maac_new"])
  #  args =  parser.parse_args(["--name", "test_ic3"]) 
    #args =  parser.parse_args(["--name", "test_a2c"]) 
    args =  parser.parse_args(["--name", "test_ppo2"])
    #args =  parser.parse_args(["--name", "exp_ppo_curr"])
   # args =  parser.parse_args(["--name", "ppo_hyp_param"])
    experiments = [exp, exp2, exp3, exp4, exp5, exp6, exp7, exp8, exp9, exp10, exp11]
    flag = True
    for ex in experiments:
        if hasattr(ex, args.name):
            e = getattr(ex, args.name)
            e()
            flag = False
    if flag:
        print("No experiment with that name")

    # elif hasattr(exp2, args.name):
    #     e2 = getattr(exp2, args.name)
    #     e2()
    # elif hasattr(exp3, args.name):
    #     a2c2 = getattr(exp3, args.name)
    #     a2c2()
    # elif hasattr(exp4, args.name):
    #     e4 = getattr(exp4, args.name)
    #     e4()
    # elif hasattr(exp5, args.name):
    #     e5 = getattr(exp5, args.name)
    #     e5()
    # elif hasattr(exp6, args.name):
    #     e6 = getattr(exp6, args.name)
    #     e6()
    # else:
    #     print("No experiment with that name")
