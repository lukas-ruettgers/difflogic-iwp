import os, sys
import shutil
from pathlib import Path
import fnmatch
import zarr
import wandb
import time
import numpy as np
import torch

from .config import HISTOGRAM_BINS
from .models import get_layers, get_cuda_func, LogicLayer, LogicLayerIWP

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from execution_setup.directories import DIR_RESULTS_TEMP, DIR_RESULTS

def setup_logging(args, model):
    # Init wandb metrics
    wandb_metrics = init_wandb(args, model)

    # Init zarr metrics
    zarr_metrics = init_zarr(args, model)
    return wandb_metrics, zarr_metrics

def init_zarr(args, model):
    zarr_metrics = None
    if len(args.log_verbose) > 0 : # string not empty
        N = 1 + (args.num_iterations-1) // args.ext_eval_freq
        N_eval = 1 + (args.num_iterations-1) // args.eval_freq
        logic_layers = get_layers(args, model)
        L = len(logic_layers)
        W = args.k
        K = 4 if args.iwp else 16
        zarr_shape = (N,L,W,K)
        zarr_chunk = (1,1,W,K)
            
        # print(f"N={N}, L={L}, W={W}, K={K}")
        zarr_activations = None
        zarr_grads_t = None
        zarr_test_cont = None
        zarr_test_disc = None
        zarr_valid_disc = None
        zarr_features = None
        zarr_timing_backward = None
        zarr_timing_forward = None
        if 'act' in args.log_verbose:
            zarr_activations = initialize_weight_storage(
                shape=zarr_shape, run_name = args.wandb_name, arr_name=f'activations', chunks=zarr_chunk
            )
        if 'gr-t' in args.log_verbose:
            zarr_grads_t = initialize_weight_storage(
                shape=(L), chunks=(1), run_name = args.wandb_name, arr_name=f'grads_t'
            )
        if 'eval' in args.log_verbose:
            zarr_test_cont = initialize_weight_storage(
                shape=(N_eval), chunks=(1), run_name = args.wandb_name, arr_name=f'test_cont'
            )
            zarr_test_disc = initialize_weight_storage(
                shape=(N_eval), chunks=(1), run_name = args.wandb_name, arr_name=f'test_disc'
            )
            zarr_valid_disc = initialize_weight_storage(
                shape=(N_eval), chunks=(1), run_name = args.wandb_name, arr_name=f'valid_disc'
            )
        if 'features' in args.log_verbose:
            zarr_features = initialize_weight_storage(
                shape=(L+1,HISTOGRAM_BINS), chunks=(HISTOGRAM_BINS), run_name = args.wandb_name, arr_name=f'features'
            )
        if 'timing' in args.log_verbose:
            zarr_timing_forward = initialize_weight_storage(
                shape=(args.n_timing_measurements), chunks=(1), run_name = args.wandb_name, arr_name=f'timing_forward'
            )
            zarr_timing_backward = initialize_weight_storage(
                shape=(args.n_timing_measurements), chunks=(1), run_name = args.wandb_name, arr_name=f'timing_backward'
            )

        zarr_metrics = dict(
            a=zarr_activations,
            gr=zarr_grads_t,
            test_cont=zarr_test_cont,
            test_disc=zarr_test_disc,
            valid_disc=zarr_valid_disc,
            features=zarr_features,
            timing_forward=zarr_timing_forward,
            timing_backward=zarr_timing_backward,
        )
    return zarr_metrics

# Weights and biases
def init_wandb(args, model):
    wandb_name = time.strftime("%Y%m%d%H%M%S")

    assert wandb.run is None
    args.wandb_name = wandb_name

    if args.no_logging:
        return

    # Build config dict
    wandb_config = vars(args)
    wandb.init(
        project="difflogic-iwp", 
        name=wandb_name,
        config=wandb_config
    )
    
    wandb_metrics = dict()
    wandb_metrics['valid/acc_eval_best']=0
    wandb_metrics['valid/acc_train_best']=0
    wandb_metrics['test/acc_eval_best']=0
    wandb_metrics['test/acc_train_best']=0
    wandb_metrics['train/acc_eval_best']=0
    wandb_metrics['train/acc_train_best']=0
    
    if 'timing' in args.wandb_verbose:
        setup_perf_timing(args, model)
    return wandb_metrics

def clean_wandb():
    # Clean current directory wandb folder
    try:
        if os.path.exists("wandb"):
            shutil.rmtree("wandb")
            print("Cleaned local wandb directory")
    except Exception as e:
        print(f"Clean up local wandb folder failed: {e}")

def setup_perf_timing(args, model):
    logic_layers = get_layers(args, model)
    L = len(logic_layers)
    N = args.n_timing_measurements * L
    Func = get_cuda_func(args)
    
    Func.timing_measurements = N
    Func.timing_measurements_factor = 1 / N
    Func.backward_count = 0
    Func.backward_time = 0
    Func.backward_times = []

def log_timing_statistics(args, forward_time_zarr, backward_time_zarr):
    if 'timing' in args.log_verbose: 
        forward_times = np.array(forward_time_zarr)
        forward_times_mean = np.mean(forward_times)
        forward_times_std = np.std(forward_times)
        backward_times = np.array(backward_time_zarr)
        backward_times_mean = np.mean(backward_times)
        backward_times_std = np.std(backward_times)
        
        timing_metrics = {
            'timing/forward_mean': forward_times_mean,
            'timing/forward_std': forward_times_std,
            'timing/backward_mean': backward_times_mean,
            'timing/backward_std': backward_times_std,
        }
        # print(timing_metrics)
        if timing_metrics is not None and wandb.run is not None:
            wandb.log(timing_metrics)

def log_activations(x, model, log_features_wandb, log_features_zarr, zarr_features=None):
    k = 0
    wandb_metrics = dict()
    if log_features_zarr:
        x_np = x.clone().detach().cpu().numpy().flatten()  
        hist, _ = np.histogram(x_np, bins=HISTOGRAM_BINS, range=(0.0, 1.0))
        zarr_features[0,:] = hist
    if log_features_wandb:
        wandb_metrics[f"features/std/after_{k:02d}_layers"] = x.std().item()
        wandb_metrics[f"features/abs_mean/after_{k:02d}_layers"] = x.abs().mean().item()
        wandb_metrics[f"features/hist/after_{k:02d}_layers"] = wandb.Histogram(x.cpu().numpy())
                
    for m in model:
        x = m(x)
        if isinstance(m, (LogicLayer, LogicLayerIWP)):
            k += 1
            if log_features_zarr:
                x_np = x.clone().detach().cpu().numpy().flatten()  
                hist, _ = np.histogram(x_np, bins=HISTOGRAM_BINS, range=(0.0, 1.0))
                zarr_features[k,:] = hist
            if log_features_wandb:
                wandb_metrics[f"features/std/after_{k:02d}_layers"] = x.std().item()
                wandb_metrics[f"features/abs_mean/after_{k:02d}_layers"] = x.abs().mean().item()
                wandb_metrics[f"features/hist/after_{k:02d}_layers"] = wandb.Histogram(x.cpu().numpy())
    return zarr_features, wandb_metrics

def log_weights(logic_layers, subsample_size = 1000):
    wandb_metrics = dict()
    for layer_idx, logic_layer in enumerate(logic_layers):
        # WEIGHTS
        weights = logic_layer.weights.detach().cpu()

        nan_mask = torch.isnan(weights)
        if nan_mask.any():
            print(f"WARNING: {nan_mask.sum()} ({(100 * nan_mask.sum() / nan_mask.numel()):2.2f}%) NaN values detected in weights at layer {layer_idx}")
            # grads = grads[~nan_mask]
        
        inf_mask = torch.isinf(weights)
        if inf_mask.any():
            print(f"WARNING: {nan_mask.sum()} ({(100 * nan_mask.sum() / nan_mask.numel()):2.2f}%) Inf values detected in weights at layer {layer_idx}")

        valid_weights = weights[~(nan_mask | inf_mask)]

        sample_size = valid_weights.numel()
        logged_weights = valid_weights
        if sample_size > subsample_size:  # Only subsample if tensor is large
            indices = torch.randperm(sample_size)[:subsample_size]
            logged_weights = valid_weights.flatten()[indices]
        
        wandb_metrics[f"weights/layer_{layer_idx}_hist"] = wandb.Histogram(logged_weights)
        wandb_metrics[f"weights/layer_{layer_idx}_std"] = valid_weights.std()
        wandb_metrics[f"weights/layer_{layer_idx}_mean"] = valid_weights.mean()
    return wandb_metrics

def log_gradients(logic_layers, zarr_grads_t):
    for layer_idx, logic_layer in enumerate(logic_layers):
        grads = logic_layer.weights.grad.detach().cpu()
        nan_mask = torch.isnan(grads)
        if nan_mask.any():
            print(f"WARNING: {nan_mask.sum()} ({(100 * nan_mask.sum() / nan_mask.numel()):2.2f}%) NaN values detected in grads at layer {layer_idx}")
        
        inf_mask = torch.isinf(grads)
        if inf_mask.any():
            print(f"WARNING: {inf_mask.sum()} ({(100 * inf_mask.sum() / inf_mask.numel()):2.2f}%) Inf values detected in grads at layer {layer_idx}")
        
        valid_mask = ~(nan_mask | inf_mask)
        if hasattr(logic_layer, 'grad_mask'):
            grad_mask = logic_layer.grad_mask.detach().cpu()
            valid_mask &= grad_mask

        valid_grads = grads[valid_mask]
        if valid_grads.numel() == 0:
            grads_mean = -1
        else:
            grads_mean = np.abs(valid_grads).mean().numpy()
        # print(f"------ Grads at layer {layer_idx} ------------\n {grads_mean}\n")
        zarr_grads_t[layer_idx] = grads_mean
    return zarr_grads_t

def results(args):
    if wandb.run is not None:
        wandb.finish()

        # Copy results from temp back to results dir
        if DIR_RESULTS_TEMP != DIR_RESULTS:
            copy_results(src=DIR_RESULTS_TEMP, dst=DIR_RESULTS, run_name=args.wandb_name)

def copy_results(src, dst, run_name):
    source_dir = os.path.join(src, run_name)
    target_dir = os.path.join(dst, run_name)
    print(f"Copying results from {source_dir} to {target_dir} ...")
    cuttree_filtered(
        src=source_dir,
        dst=target_dir,
        overwrite=True
    )

def copytree_filtered(src, dst, exclude_pattern="*.tmp", overwrite=True):
    os.makedirs(dst, exist_ok=True)
    for root, _, files in os.walk(src):
        # Compute destination path
        rel_path = os.path.relpath(root, src)
        dst_root = os.path.join(dst, rel_path)
        os.makedirs(dst_root, exist_ok=True)

        for file in files:
            if fnmatch.fnmatch(file, exclude_pattern):
                continue  # Skip excluded files
            src_file = os.path.join(root, file)
            dst_file = os.path.join(dst_root, file)

            if not overwrite and os.path.exists(dst_file):
                continue  # Skip existing files if not overwriting

            shutil.copy2(src_file, dst_file)  # copy2 preserves metadata

def cuttree_filtered(src, dst, exclude_pattern="*.tmp", overwrite=True):
    copytree_filtered(src, dst, exclude_pattern=exclude_pattern, overwrite=overwrite)
    # Remove src dir
    src = Path(src)
    if src.exists() and src.is_dir():
        shutil.rmtree(src)

# Logging
def initialize_weight_storage(shape, run_name, arr_name, chunks=None, results_dir=DIR_RESULTS):
    """Initialize zarr array"""
    results_path = Path(os.path.join(results_dir, run_name))
    results_path.mkdir(parents=True, exist_ok=True)
    
    weights_store = zarr.open(
        results_path / arr_name,
        mode='w',
        shape=shape,
        chunks=chunks,
        dtype=np.float32,
        fill_value=np.nan
    )
    return weights_store
