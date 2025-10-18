import os, random, numpy as np, torch

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

def class_weights_from_counts(counts):
    # counts: list of ints aligned to class index
    import numpy as np, torch
    cw = np.sum(counts) / (len(counts) * np.array(counts))
    return torch.tensor(cw, dtype=torch.float32)
