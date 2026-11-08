r""" Helper functions """
import random

import torch
import numpy as np


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def fix_randseed(seed):
    r""" Set random seeds for reproducibility """
    if seed is None:
        seed = int(random.random() * 1e5)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def mean(x):
    return sum(x) / len(x) if len(x) > 0 else 0.0


def worker_init_fn(worker_id):
    r""" Per-worker reseed so NumPy/random streams differ across DataLoader workers """
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def to_cuda(batch):
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(DEVICE)
    return batch


def to_cpu(tensor):
    return tensor.detach().clone().cpu()
