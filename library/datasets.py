import math
import os, sys
import numpy as np
import random

import torch
import torchvision

from torchvision.transforms import v2 

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from execution_setup.directories import DIR_DATA


class CachedDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=lambda x:x):
        self.data_dir = data_dir
        self.transform = transform
        self.files = sorted([f for f in os.listdir(data_dir) if f.endswith('.pt')])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(os.path.join(self.data_dir, self.files[idx]), weights_only=True)
        img, label = data['img'], data['label']
        img = self.transform(img)
        return img, label
    
# --- Check and precompute if needed ---
def preprocess_dataset(dataset_class, data_dir, save_dir, num_samples_required=None, transform=torchvision.transforms.ToTensor()):
    os.makedirs(save_dir, exist_ok=True)
    existing_files = [f for f in os.listdir(save_dir) if f.endswith('.pt')]
    base_dataset = None
    if num_samples_required is None:
        base_dataset = dataset_class(root=data_dir, train=True, transform=transform, download=True)
        num_samples_required = len(base_dataset)

    print(f"[INFO] Found {len(existing_files)} preprocessed samples in {save_dir}.")
    if len(existing_files) >= num_samples_required:
        print(f"These are sufficiently many. Skipping generation.")
        return

    if base_dataset is None:
        base_dataset = dataset_class(root=data_dir, train=True, transform=transform, download=True)
    total_generated = len(existing_files)
    print(f"[INFO] Generating {num_samples_required} samples with augmentations to {save_dir}...")

    idx_counter = len(existing_files)

    # Loop until we have enough
    while total_generated < num_samples_required:
        for i in range(len(base_dataset)):
            img, label = base_dataset[i]
            # torch.manual_seed(random.randint(0, 99999))  # ensure randomness
            # augmented = augment_pipeline(img)
            save_path = os.path.join(save_dir, f'{idx_counter:06d}.pt')
            # subdir = f'{idx_counter // 10000:05d}'  # group every 10k files
            # os.makedirs(os.path.join(save_dir, subdir), exist_ok=True)
            # save_path = os.path.join(save_dir, subdir, f'{idx_counter:06d}.pt')
            # torch.save({'img': augmented, 'label': label}, save_path)
            torch.save({'img': img, 'label': label}, save_path)
            idx_counter += 1
            total_generated += 1
            if total_generated >= num_samples_required:
                break

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def load_dataset(args):
    g = torch.Generator()
    g.manual_seed(args.seed)

    if 'cifar-10' != args.dataset and 'cifar-100' != args.dataset:
        raise NotImplementedError(args.dataset)
    transform_list, augment_transform_list = get_transforms(args)
    transforms_test = v2.Compose(transform_list)
    transforms_train = v2.Compose(augment_transform_list + transform_list)
    transforms_train_preprocessing = v2.Compose(transform_list)

    dataset_class = get_dataset_class(args)
    cifar_data_dir = os.path.join(DIR_DATA, args.dataset)
    save_dir = f'{cifar_data_dir}_{args.encoding}'
    preprocess_dataset(dataset_class, cifar_data_dir, save_dir, num_samples_required=None, transform=transforms_train_preprocessing)

    if args.preprocess_once:
        train_transforms_augmentations = lambda x:x
        if args.augment:
            train_transforms_augmentations = v2.Compose(augment_transform_list)
        train_set = CachedDataset(save_dir, transform=train_transforms_augmentations)
    else:
        train_set = dataset_class(cifar_data_dir, train=True, download=True, transform=transforms_train)
    test_set = dataset_class(cifar_data_dir, train=False, transform=transforms_test)

    train_set_size = math.ceil((1 - args.valid_set_size) * len(train_set))
    valid_set_size = len(train_set) - train_set_size
    train_set, validation_set = torch.utils.data.random_split(train_set, [train_set_size, valid_set_size])

    num_workers=4
    # Set num_workers=0 if you encounter Pin memory errors on your GPU
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=False, drop_last=True, num_workers=num_workers, worker_init_fn=seed_worker, generator=g)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=args.batch_size, shuffle=False, pin_memory=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, pin_memory=True, drop_last=True)

    return train_loader, validation_loader, test_loader, train_set_size // args.batch_size, lambda x: x

def get_dataset_class(args):
    if args.dataset == 'cifar-10':
        return torchvision.datasets.CIFAR10
    elif args.dataset == 'cifar-100':
        return torchvision.datasets.CIFAR100
    

def temperature_encoding(n_thresholds):
    return lambda x: torch.cat([(x > (i + 1) / (n_thresholds + 1)).float() for i in range(n_thresholds)], dim=0)

def get_transforms(args):
    transform_list = [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ]
    augment_transform_list = []
    if args.augment:
        augment_transform_list = [
            v2.RandomResizedCrop(size=(32, 32), scale=(0.5, 1.0), ratio=(0.8, 1.2), antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
        ]
    
    
    # Binary encoding
    encoding_resolutions = [3, 7, 15, 23, 31]

    encoding_map = {
        'real-input': lambda x: x,
        **{f'{n}-thresholds': temperature_encoding(n) for n in encoding_resolutions}
    }

    encoding = args.encoding  # Everything after 'c10-' or 'c100-'
    if encoding not in encoding_map:
        raise ValueError(f"Unknown encoding: {encoding}. Available: {list(encoding_map.keys())}")

    threshold_transform = encoding_map[encoding]
    transform_list.append(v2.Lambda(threshold_transform))

    return transform_list, augment_transform_list

def input_dim_of_dataset(args):
    spatial_size = in_size_of_dataset(args)
    channel_size = num_channels_of_dataset(args)
    return spatial_size * spatial_size * channel_size

def num_channels_of_dataset(args):
    if 'cifar' in args.dataset:
        res = 3
        if '3-thresholds' in args.encoding and not '23-thresholds' in args.encoding: 
            res *= 3
        elif '7-thresholds' in args.encoding:
            res *= 7
        elif '15-thresholds' in args.encoding:
            res *= 15
        elif '23-thresholds' in args.encoding:
            res *= 23
        elif '31-thresholds' in args.encoding:
            res *= 31
        return res

def in_size_of_dataset(args):
    if 'cifar' in args.dataset:
        return 32
    
def class_count_of_dataset(args):
    if 'cifar-100' in args.dataset:
        return 100
    if 'cifar-10' in args.dataset:
        return 10