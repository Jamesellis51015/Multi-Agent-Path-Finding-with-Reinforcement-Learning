




def make_env(args):
    # Important env characteristics to set:
    # -map size
    # -objects density
    # -collisions
    # -observation space
    # -reward function 
    # -observability (Fully or partial) and view size
    # -
    
    if args.env_name == 'independent_navigation-v0':

        from Env.env import Independent_NavigationV0
        env = Independent_NavigationV0(args)
        return env

    elif args.env_name == 'independent_navigation-v1':
        from Env.env import Independent_NavigationV1
        env = Independent_NavigationV1(args)
        return env
    elif args.env_name == 'independent_navigation-v2':
        from Env.env import Independent_NavigationV2
        env = Independent_NavigationV2(args)
        return env
    elif args.env_name == 'independent_navigation-v3':
        from Env.env import Independent_NavigationV3
        env = Independent_NavigationV3(args)
        return env
    elif args.env_name == 'independent_navigation-v4_1':
        from Env.env import Independent_NavigationV4_1 as hldr
        env = hldr(args)
        return env
    elif args.env_name == 'independent_navigation-v4_2':
        from Env.env import Independent_NavigationV4_2 as hldr
        env = hldr(args)
        return env
    elif args.env_name == 'independent_navigation-v4_3':
        from Env.env import Independent_NavigationV4_3 as hldr
        env = hldr(args)
        return env
    elif args.env_name == "predator_pray":
        pass
    elif args.env_name == "cooperative_navigation-v0":
        from Env.env import Cooperative_Navigation_V0 
        env = Cooperative_Navigation_V0(args)
        return env
    elif args.env_name == "cooperative_navigation-v1":
        from Env.env import Cooperative_Navigation_V1 
        env = Cooperative_Navigation_V1(args)
        return env
    elif args.env_name == "narrow_corridor-v0":
        from Env.env import Narrow_CorridorV0 
        env = Narrow_CorridorV0(args)
        return env
    else:
        raise NotImplementedError