

from Agents.PG_Shared import PG_Shared
from Agents.PG_Separate import PG_Separate
from Agents.AC_Separate import AC_Separate
from Agents.AC_Shared import  AC_Shared
from Agents.AC_One_Step import AC_One_Step
#from Agents.AC_testing import AC_Test
from Agents.ic3Net import IC3Net
#from Agents.MAAC import MAAC

def make_policy(args, env):
    if args.policy == 'PG_IND':
        pass
    elif args.policy == 'MAAC':
        raise Exception("make_policy should not be called")
        #policy = MAAC(args, env.observation_space, env.action_space)
    elif args.policy == 'IC3':
        policy = IC3Net(args, env.observation_space, env.action_space)
        return policy
    elif args.policy == 'PG_Shared':
        policy = PG_Shared(args, env.observation_space, env.action_space)
        return policy
    elif args.policy == 'PG_Separate':
        policy = PG_Separate(args, env.observation_space, env.action_space)
        return policy
    elif args.policy == 'AC_Shared':
        policy = AC_Shared(args, env.observation_space, env.action_space)
        return policy
    elif args.policy == 'AC_Separate':
        policy = AC_Separate(args, env.observation_space, env.action_space)
        return policy
    elif args.policy == 'AC_One_Step':
        policy = AC_One_Step(args, env.observation_space, env.action_space)
        return policy
    # elif args.policy == 'AC_Test':
    #     policy = AC_Test(args, env.observation_space, env.action_space)
    #     return policy
    else:
        raise NotImplementedError


    