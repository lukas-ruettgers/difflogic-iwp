import random

import torch
import numpy as np

import subprocess
subprocess.run(["python", "./execution_setup/set_default_directories.py"], check=True)

import library
import importlib
importlib.reload(library)
from library.config import parse_args, set_baseline_args
from library.datasets import load_dataset
from library.models import get_model, get_model_size_in_MB
from library.train import train
from library.measurements import results

num_threads = torch.get_num_threads()
torch.set_num_threads(num_threads)
torch.set_num_interop_threads(num_threads)
print(f"Number of threads used: {torch.get_num_threads()}")

if __name__ == '__main__':
    assert torch.cuda.is_available(), "CUDA is not available"
    args = parse_args()
    
    # Set args for predefined models in 2024 paper
    set_baseline_args(args)
    print('================ ARGUMENTS =====================')
    for key in sorted(vars(args)):
        print(f"{key}: {getattr(args, key)}")
    print()

    # Reproducibility
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # MODEL
    print('================ MODEL =========================')
    model = get_model(args)
    print(f"Model: {model}")
    model_size = get_model_size_in_MB(model)
    print(f"Model size: {model_size:.2f} MB")
    print()

    # DATA
    print('================ DATASET =======================')
    train_loader, validation_loader, test_loader, num_train_batches, transform = load_dataset(args)
    print()

    trkw = dict(
        test_loader=test_loader, 
        valid_loader=validation_loader,
        train_loader=train_loader,
    )

    # TRAINING
    print('================ TRAINING ======================')
    train(args, model, transform, **trkw)
    print()

    print('================ RESULTS =======================')
    results(args)
        