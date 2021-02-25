import argparse

import Experiments.ppo_experiments as exp12
import Experiments.maac_experiments as exp13
import Experiments.ic3_experiments as exp14
#import Agents.BC.data_generator as exp15
#import Agents.BC.BC as exp16
#import Experiments.mstar_experiments as exp17
#import Experiments.final_experiments as exp18

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Run Experiment")
    parser.add_argument("--name", type=str)
    args = parser.parse_args(["--name", "primal_test"])
    #args = parser.parse_args(["--name", "ic3net_test"])
    #experiments = [exp12, exp13, exp14, exp15, exp16, exp17, exp18]
    experiments = [exp12, exp13, exp14]
    #args = parser.parse_args()
    flag = True
    for ex in experiments:
        if hasattr(ex, args.name):
            print("Running: {}".format(args.name))
            e = getattr(ex, args.name)
            e()
            flag = False
    if flag:
        print("No experiment with that name")

