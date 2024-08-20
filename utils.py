import os
import torch

def create_directories(base_dir):
    paths = ['lib/weights', 'lib/images', 'lib/dataset']
    for path in paths:
        os.makedirs(os.path.join(base_dir, path), exist_ok=True)

def check_cuda():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
