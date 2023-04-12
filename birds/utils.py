import numpy as np
import torch
import random

def fix_seed(seed=None):
    if seed is None:
        seed = np.random.randint(0, 1000)
    print(f"Fixing seed to {seed}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
