import random
import numpy as np
import torch

def set_seed(seed: int, env=None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if env is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)


    