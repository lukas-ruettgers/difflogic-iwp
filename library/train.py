import random
import torch
from tqdm import tqdm
import wandb
import time
import copy

import difflogic
import importlib
importlib.reload(difflogic)

from .config import DEVICE
from .models import get_layers, setup_hooks, regularization
from .measurements import setup_logging, log_activations, log_gradients, log_weights, log_timing_statistics

def get_optimizer(args, model):
    if args.optim == 'adam':
        return torch.optim.Adam(model.parameters(), lr=args.learning_rate)

# Training
def compute_loss(pred, y):
    return torch.nn.CrossEntropyLoss()(pred, y)

def process_batch(x, y, model, transform, optim, accumulate_steps=1, backward=False):
    """
    Process a single batch: forward pass, loss computation, backward pass, and optimization.
    
    Args:
        x: Input batch
        y: Target batch
        model: Neural network model
        transform: Data transformation function
        optim: Optimizer
        args: Arguments object
        device: Device for computation
    
    Returns:
        tuple: (loss_value, train_accuracy)
    """    
    # Forward pass
    pred = model(transform(x))
        
    # Compute loss
    loss = compute_loss(pred, y)
    scaled_loss = loss / accumulate_steps

    # Compute accuracy
    train_acc = (pred.argmax(-1) == y).float().mean().item()
    
    # Backward pass and optimization
    scaled_loss.backward()
    if backward:
        optim.step()
        for module in model.modules():
            if hasattr(module, 'reset_fixed_weights'):
                module.reset_fixed_weights()
    
    return loss.item(), train_acc

def run_evaluation_and_checkpoint(args, model, best_accuracies, test_loader, valid_loader):
    valid_accuracy_eval_mode = 0
    valid_accuracy_train_mode = 0
    test_accuracy_eval_mode = 0
    test_accuracy_train_mode = 0
    if args.valid_set_size != 0:
        valid_accuracy_eval_mode = eval_on_loader(model, valid_loader, train_mode=False)
        valid_accuracy_train_mode = eval_on_loader(model, valid_loader, train_mode=True)
    test_accuracy_eval_mode = eval_on_loader(model, test_loader, train_mode=False)
    test_accuracy_train_mode = eval_on_loader(model, test_loader, train_mode=True)

    r = {
        'valid/acc_eval': valid_accuracy_eval_mode,
        'valid/acc_train': valid_accuracy_train_mode,
        'test/acc_eval': test_accuracy_eval_mode,
        'test/acc_train': test_accuracy_train_mode,
    }

    # Update best wandb metrics
    r['valid/acc_eval_best']=max(best_accuracies['valid/acc_eval'], valid_accuracy_eval_mode)
    r['valid/acc_train_best']=max(best_accuracies['valid/acc_train'], valid_accuracy_train_mode)
    r['test/acc_eval_best']=max(best_accuracies['test/acc_eval'], test_accuracy_eval_mode)
    r['test/acc_train_best']=max(best_accuracies['test/acc_train'], test_accuracy_train_mode)
    
    return r

def update_metrics_and_log(
    loss, train_acc, i, args, model, zarr_metrics, wandb_metrics, train_loader=None, test_loader=None, valid_loader=None):
    if args.no_logging:
        return
    wandb_metrics["train/loss_train"] = loss
    wandb_metrics['train/acc_train_batch'] = train_acc
    
    # Eval valid and test metrics
    if i % args.eval_freq == 0 and (i!=0 or args.eval_initial):
        best_accuracies = dict()
        best_accuracies['valid/acc_eval'] = wandb_metrics['valid/acc_eval_best']
        best_accuracies['valid/acc_train'] = wandb_metrics['valid/acc_train_best']
        best_accuracies['test/acc_eval'] = wandb_metrics['test/acc_eval_best']
        best_accuracies['test/acc_train'] = wandb_metrics['test/acc_train_best']
        m = run_evaluation_and_checkpoint(args, model, best_accuracies, test_loader=test_loader, valid_loader=valid_loader)
        wandb_metrics.update(m)
        if 'eval' in args.log_verbose:
            zarr_test_cont = zarr_metrics['test_cont']
            zarr_test_disc = zarr_metrics['test_disc']
            zarr_valid_disc = zarr_metrics['valid_disc']

            t_eval = i // args.eval_freq
            zarr_test_cont[t_eval] = m['test/acc_train']
            zarr_test_disc[t_eval] = m['test/acc_eval']
            zarr_valid_disc[t_eval] = m['valid/acc_eval']

    
    if ((i % args.ext_eval_freq == 0 and (i!=0 or args.eval_initial)) or i == args.log_timestamp):
        if i % args.ext_eval_freq == 0 and 'eval' in args.wandb_verbose:
            # Eval discretization gap on training dataset
            wandb_metrics[f"train/acc_eval"] = eval_on_loader(model, train_loader, train_mode=False, subsample_size=args.train_eval_subsample_size)
            wandb_metrics[f"train/acc_eval_best"] = max(wandb_metrics[f"train/acc_eval_best"], wandb_metrics[f"train/acc_eval"])
            
            wandb_metrics[f"train/acc_train"] = eval_on_loader(model, train_loader, train_mode=True, subsample_size=args.train_eval_subsample_size)
            wandb_metrics[f"train/acc_train_best"] = max(wandb_metrics[f"train/acc_train_best"], wandb_metrics[f"train/acc_train"])

        logic_layers = get_layers(args, model)        

        # WEIGHTS        
        log_weights_wandb = 'w' in args.wandb_verbose
        if log_weights_wandb:
            wandb_metrics_cur = log_weights(logic_layers)
            wandb_metrics.update(wandb_metrics_cur)
        
        # GRADIENTS
        log_gradients_zarr = (i == args.log_timestamp and 'gr-t' in args.log_verbose)
        if log_gradients_zarr:
            zarr_grads_t = zarr_metrics['gr']
            zarr_grads_t = log_gradients(logic_layers, zarr_grads_t)
                
        # ACTIVATIONS OVER LAYERS
        log_features_wandb = 'features' in args.wandb_verbose
        log_features_zarr = 'features' in args.log_verbose and i==args.log_timestamp
        zarr_features = zarr_metrics['features'] if log_features_zarr else None
        if log_features_wandb or log_features_zarr:
            with torch.no_grad():
                x, _ = next(iter(train_loader))
                x = x.to(DEVICE, non_blocking=True)
                zarr_features, wandb_metrics_cur = log_activations(x, model, log_features_wandb, log_features_zarr, zarr_features, wandb_metrics)
                wandb_metrics.update(wandb_metrics_cur)

    if args.eval_initial:
        i+=1
    elif i==0:
        return
    if wandb.run is not None:
        wandb.log(wandb_metrics, step=i)

def run_timing_measurements(args, model, transform, train_loader, forward_time_zarr, backward_time_zarr):
    optim = get_optimizer(args, model)
    optim.zero_grad()
    train_loader_iter = iter(train_loader)
    for i in range(args.n_timing_measurements+1):
        try:
            x, y = next(train_loader_iter)
        except StopIteration:
            # Reset iterator when we reach the end of the dataset
            train_loader_iter = iter(train_loader)
            x, y = next(train_loader_iter)
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        # Forward pass
        torch.cuda.synchronize()
        start = time.perf_counter()
        pred = model(transform(x))
        torch.cuda.synchronize()
        end = time.perf_counter()
        forward_time = 1000000 * (end - start)
        
        # Compute loss
        loss = compute_loss(pred, y)
        
        # Backward pass and optimization
        torch.cuda.synchronize()
        start = time.perf_counter()
        loss.backward()
        torch.cuda.synchronize()
        end = time.perf_counter()
        backward_time = 1000000 *(end - start)

        optim.step()
        for module in model.modules():
            if hasattr(module, 'reset_fixed_weights'):
                module.reset_fixed_weights()
        if i==0: 
            # Skip first run, outlier because of cold cache   
            continue
        forward_time_zarr[i-1] = forward_time
        backward_time_zarr[i-1] = backward_time
    log_timing_statistics(args, forward_time_zarr, backward_time_zarr)

def train(args, model, transform, train_loader, test_loader=None, valid_loader=None):
    wandb_metrics, zarr_metrics = setup_logging(args, model)
    if 'timing' in args.log_verbose:
        model_copy = copy.deepcopy(model).to(DEVICE)
        run_timing_measurements(args, model_copy, transform, train_loader, 
                                zarr_metrics['timing_forward'], 
                                zarr_metrics['timing_backward'])
        del model_copy         
        torch.cuda.empty_cache()       

    train_loader_iter = iter(train_loader)
    optim = get_optimizer(args, model)
    optim.zero_grad()

    N_BATCHES_PER_BACKWARD = args.batches_per_backward

    model = model.to(DEVICE)
    setup_hooks(args, model)
    for i in tqdm(
            range(args.num_iterations),
            total=args.num_iterations,
    ):  
        loss_per_backward = 0.0
        train_acc_per_backward = 0.0
        for j in range(N_BATCHES_PER_BACKWARD):
            # Load data
            try:
                x, y = next(train_loader_iter)
            except StopIteration:
                # Reset iterator when we reach the end of the dataset
                train_loader_iter = iter(train_loader)
                x, y = next(train_loader_iter)
            
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            # Process data
            backward = (j+1)==N_BATCHES_PER_BACKWARD
            
            regularization(args, model, regularization_active=True)
            
            loss, train_acc = process_batch(x, y, model, transform, optim, accumulate_steps=N_BATCHES_PER_BACKWARD, backward=backward)
            
            regularization(args, model, regularization_active=False)
            
            loss_per_backward += loss
            train_acc_per_backward += train_acc


        loss_per_backward /= N_BATCHES_PER_BACKWARD
        train_acc_per_backward /= N_BATCHES_PER_BACKWARD
        update_metrics_and_log(
            loss_per_backward, train_acc_per_backward, i, args, model, zarr_metrics, wandb_metrics, 
            train_loader=train_loader,
            test_loader=test_loader, 
            valid_loader=valid_loader,
        )

        optim.zero_grad()
    

# Evaluation
def eval(model, test_x, test_y, mode):
    orig_mode = model.training
    with torch.no_grad():
        model.train(mode=mode)
        res = 0
        for i in range(len(test_y) // 1_000):
            res += (model(test_x[i*1000:(i+1)*1000]).argmax(-1) == test_y[i*1000:(i+1)*1000]).float().mean().item()
        res = res / (len(test_y) // 1_000)
        model.train(mode=orig_mode)
    return res

def accuracy(y_pred: torch.Tensor, y: torch.Tensor) -> float:
    return (y_pred.argmax(-1) == y).to(torch.float32).mean().item()

# Uniform eval method for both train and test
def eval_on_loader(model: torch.nn.Module, loader: torch.utils.data.DataLoader, train_mode: bool, subsample_size: int = None) -> float:
    device = DEVICE
    if loader is None:
        return -1

    orig_mode = model.training
    with torch.no_grad():
        model.train(mode=train_mode)
        
        if subsample_size is not None:
            # Streaming approach with random batch skipping
            accuracies = []
            samples_processed = 0
            
            # Estimate skip probability based on dataset size
            dataset_size = len(loader.dataset) if hasattr(loader.dataset, '__len__') else None
            batch_size = loader.batch_size
            
            if dataset_size is not None:
                # Calculate probability of keeping each batch
                total_batches = len(loader)
                target_batches = min(total_batches, (subsample_size + batch_size - 1) // batch_size)
                keep_prob = target_batches / total_batches
            else:
                # Conservative estimate if dataset size is unknown
                keep_prob = min(1.0, subsample_size / (batch_size * 1000))
            
            for batch_idx, (x, y) in enumerate(loader):
                # Stop if we've processed enough samples
                if samples_processed >= subsample_size:
                    break
                
                # Randomly decide whether to process this batch
                if random.random() < keep_prob:
                    if train_mode:
                        acc = accuracy(model(x.to(device=device, non_blocking=True)), 
                                    y.to(device, non_blocking=True))
                    else:
                        acc = accuracy(model(x.to(device=device, non_blocking=True).round()), 
                                    y.to(device, non_blocking=True))
                        
                    accuracies.append(acc)
                    samples_processed += len(y)
                
                # Adaptive adjustment of keep_prob if we're not getting enough samples
                if batch_idx > 0 and batch_idx % 100 == 0:
                    expected_samples = (batch_idx + 1) * batch_size * keep_prob
                    if samples_processed < expected_samples * 0.5:
                        keep_prob = min(1.0, keep_prob * 1.5)
            
            if len(accuracies) == 0:
                return -1
            
            res = torch.mean(torch.tensor(accuracies))
        else:
            # No subsampling - evaluate on entire dataset
            if train_mode:
                res = torch.mean(
                    torch.tensor([
                        accuracy(model(x.to(device=device, non_blocking=True)), y.to(device, non_blocking=True))
                        for x, y in loader
                    ])
                )
            else:
                res = torch.mean(
                    torch.tensor([
                        accuracy(model(x.to(device=device, non_blocking=True)).round(), y.to(device, non_blocking=True))
                        for x, y in loader
                    ])
                )
            
        model.train(mode=orig_mode)
    return res.item()
