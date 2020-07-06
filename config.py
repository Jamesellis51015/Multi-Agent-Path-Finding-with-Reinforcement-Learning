'''A file for global variables '''

import torch
import numpy as np


#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

print("Device = {}".format(device))


def set_global_seed(seed = None):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

