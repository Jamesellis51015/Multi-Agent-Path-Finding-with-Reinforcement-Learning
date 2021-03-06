# Muli-Agent Path Finding with Reinforcement Learning

## This repository contains:
- ### Multi-Agent Gridworld Environment: A basic gridworld implementation where agents can collide with each other as well as obstacles. Agents have to navigate to their goal locations. Majority of the code for the environment can be found in Env/grid_env.py and Env/env.py

![Alt text](gridworld_example.png?raw=true "")

- ### PPO code in Agents/PPO folder and Agents/PPO_IL.py.

- ### MAAC code in Agents/MAAC.py and Agents/maac_utils/

- ### IC3Net code in Agents/ic3Net.py and ic3Net_trainer.py

- ### ODM* code in utils/ODM_star.py

# How to run examples:

## Create a conda 'gridworld' environment using the environment.yml file:
```bash
conda env create -f environment.yml
conda activate gridworld
```
## Manually contol an agent on a gridworld:
Run gridworld_manual.py with arguments for the map size [x=y=map_shape], the number of agents and obstacle density:
```bash
python gridworld_manual.py --map_shape 10 --n_agents 2 --obj_density 0.2
```
To see a command-line printout of the observations add --verbose

Follow comand instuctions to make agents move. 
For example: 0 1 1 2 s    => agent 0 move up ; agent 1 move right ; step

## Run ODM* on an environment:
```bash
python odmstar_example.py --map_shape 32 --n_agents 20 --obj_density 0.2 --inflation 1.1
```
## Train a PPO agent:
```bash
python run_experiments.py --name ppo_test
```
## Train a IC3Net agent:
```bash
python run_experiments.py --name ic3net_test
```
## Train a MAAC agent:
```bash
python run_experiments.py --name maac_test
```
## Train a PRIMAL agent:
```bash
python run_experiments.py --name primal_test
```
## Code Credits:

Rendering of Gridworld environment:

```
@misc{gym_minigrid,
  author = {Chevalier-Boisvert, Maxime and Willems, Lucas and Pal, Suman},
  title = {Minimalistic Gridworld Environment for OpenAI Gym},
  year = {2018},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/maximecb/gym-minigrid}},
}
```

Code in Agents/MAAC.py and Agents/maac_utils/ modified from: https://github.com/shariqiqbal2810/MAAC
 
```
@InProceedings{pmlr-v97-iqbal19a,
  title =    {Actor-Attention-Critic for Multi-Agent Reinforcement Learning},
  author =   {Iqbal, Shariq and Sha, Fei},
  booktitle =    {Proceedings of the 36th International Conference on Machine Learning},
  pages =    {2961--2970},
  year =     {2019},
  editor =   {Chaudhuri, Kamalika and Salakhutdinov, Ruslan},
  volume =   {97},
  series =   {Proceedings of Machine Learning Research},
  address =      {Long Beach, California, USA},
  month =    {09--15 Jun},
  publisher =    {PMLR},
  pdf =      {http://proceedings.mlr.press/v97/iqbal19a/iqbal19a.pdf},
  url =      {http://proceedings.mlr.press/v97/iqbal19a.html},
}

```
Code in Agents/maac_utils/vec_env/ from Open AI baselines: https://github.com/openai/baselines/tree/master/baselines/common

Code in Agents/ic3Net.py modified from: https://github.com/IC3Net/IC3Net
```
@article{singh2018learning,
  title={Learning when to Communicate at Scale in Multiagent Cooperative and Competitive Tasks},
  author={Singh, Amanpreet and Jain, Tushar and Sukhbaatar, Sainbayar},
  journal={arXiv preprint arXiv:1812.09755},
  year={2018}
}
```




