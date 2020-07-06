import numpy as np

def flat_np_lst(dict_inp, flat = True, npy = True):
    """Converts a dict of np observation to a np-arr of flattened obs"""
    inp_lst =  list(dict_inp.values())
    if flat: inp_lst = [i.flatten() for i in inp_lst]
    if npy: inp_lst = np.stack(inp_lst)
    return inp_lst


def flat_np_lst_env_stack(dict_inp, flat = True, npy = True):
    return np.stack([flat_np_lst(i, flat=flat) for i in dict_inp])





def lst_to_dict(inp_lst):
    return {i:l for i,l in enumerate(inp_lst)}

def wrap_actions(actions):
  act_out = []
  for act in actions:
    act = np.array(act)
    act_shape = list(act.shape)
    act_shape = act_shape[:-1]
    act = np.argmax(act, axis = -1).reshape(act_shape)
    act_out.append(lst_to_dict(list(act)))
  return act_out

