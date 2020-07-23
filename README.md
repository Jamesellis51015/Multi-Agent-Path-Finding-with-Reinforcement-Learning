# Multiagent-Gridworld

## Work In Progress!

A basic gridworld implementation where agents can collide with each other as well as obstacles. Agents have to navigate to their goal locations. Majority of the code for the environment can be found in Env/grid_env.py and Env/env.py

![Alt text](gridworld_example.png?raw=true "")

Requirements:
- Many requirements. Best would be to make a conda environment from environment.yml

For an example run manual_test.py and type commands according to instructions. (Sometimes fails first time, if its a problem with command inputs, try again.)

To make own gridworld, make new class in env.py which inherits Grid_Env class from grid_env.py and implement reset() and get_rewards() accoring to new environment dynamics. Custom environments instead of randomly generated ones can also be specified by a csv file (see Env/custom folder for examples). 

Credits for rendering goes to:

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





