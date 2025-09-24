from math import sqrt
import torch
import torch.nn as nn

from .config import DEVICE
from .datasets import input_dim_of_dataset, num_channels_of_dataset, in_size_of_dataset, class_count_of_dataset

from difflogic import GroupSum, LogicLayer, LogicLayerIWP, LogicLayerCudaFunction, IWPLogicLayerCudaFunction

def get_model(args):
    in_dim = input_dim_of_dataset(args)
    in_channels = num_channels_of_dataset(args)
    in_size = in_size_of_dataset(args)
    class_count = class_count_of_dataset(args)

    args.in_dim = in_dim
    args.in_channels = in_channels
    args.in_size = in_size
    args.class_count = class_count
    
    llkw = dict(
        weight_init_choice=args.weights_init,
        sigma=args.weights_init_variance,  
    )
    if args.iwp:
        llkw.update(
            act_fn=args.act_fn,
            skip_grad=args.skip_grad,
            init_shift=args.init_shift,
            init_shift_direction=args.init_shift_direction,
            random_outage=args.random_outage,
            random_outage_prob=args.random_outage_prob,
            run_op_on_iwp=args.run_op_on_iwp,
            run_original_cuda=args.run_original_cuda,
        )
    if args.architecture=='lgn':
        model = get_lgn(args, llkw)
    elif args.architecture == 'cnn':
        model = get_cnn(args)

    return model

def get_lgn(args, llkw):
    logic_layers = []
    logic_layer = LogicLayer
    if args.iwp:
        logic_layer = LogicLayerIWP

    k = args.k
    l = args.depth * args.depth_scale
    
    llkw.update(
        connections=args.connections,
        device=DEVICE,
    )
    
    logic_layers.append(torch.nn.Flatten())
    logic_layers.append(logic_layer(in_dim=args.in_dim, out_dim=k, **llkw))
    if args.resconnect:
        for j in range(l - 2):
            llkw.update(
                res_connect_fraction = (j+1)/(l-1)
            )
            logic_layers.append(logic_layer(in_dim=k, out_dim=k, **llkw))
        # Allow to permute all channels in the final logic_layers before Group Sum
        
        llkw.update(
            res_connect_fraction = 0
        )
        logic_layers.append(logic_layer(in_dim=k, out_dim=k, **llkw))
    else:
        for _ in range(l - 2):
            logic_layers.append(logic_layer(in_dim=k, out_dim=k, **llkw))
        k_last = k 
        if args.c100_scale_width:
            k_last *= 10
            args.softmax_temperature *= sqrt(10)
        logic_layers.append(logic_layer(in_dim=k, out_dim=k_last, **llkw))
    
    gskw = dict(
        k=args.class_count,
        tau=args.softmax_temperature,
        device=DEVICE,
        in_dim=args.k,
    )
    model = torch.nn.Sequential(
        *logic_layers,
        GroupSum(**gskw)
    )
    return model

def get_cnn(args):
    in_channels = args.in_channels 
    in_height = args.in_size
    in_width = args.in_size
    base_channels = args.k
    num_blocks = args.depth_scale * 2  # total number of conv blocks
    num_classes = args.class_count
    
    layers = []
    current_channels = in_channels
    out_channels = base_channels

    for block in range(num_blocks):
        # Convolution + ReLU
        layers.append(nn.Conv2d(current_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU(inplace=True))

        # Optional: add MaxPool every 2 blocks to reduce spatial dims
        if block % 2 == 1:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        current_channels = out_channels
        out_channels *= 2  # double channels at each stage (optional)

    # Global pooling to reduce to (B, C, 1, 1)
    layers.append(nn.AdaptiveAvgPool2d((1, 1)))
    layers.append(nn.Flatten())  # flatten to (B, C)

    # Classifier (could be single or multi-layered)
    classifier = []
    classifier.append(nn.Linear(current_channels, args.k))
    classifier.append(nn.ReLU())
    classifier.append(nn.Linear(args.k, num_classes))

    for l in layers:
        if isinstance(l, (nn.Linear, nn.Conv2d)):
            l.weights = l.weight
    for l in classifier:
        if isinstance(l, (nn.Linear, nn.Conv2d)):
            l.weights = l.weight

    model = nn.Sequential(
        *layers,
        *classifier
    )
    return model

def get_layers(args, model):
    if args.architecture=='lgn':
        return [m for m in model if isinstance(m, (LogicLayer, LogicLayerIWP))]
    elif args.architecture=='cnn':
        return [m for m in model if isinstance(m, (nn.Linear, nn.Conv2d))]

def get_model_size_in_MB(model: nn.Module) -> float:
    total_bytes = 0
    for param in model.parameters():
        total_bytes += param.nelement() * param.element_size()
    return total_bytes / (1024 ** 2)  # Convert to MB

def get_cuda_func(args):
    if args.iwp:
        return IWPLogicLayerCudaFunction
    else:
        return LogicLayerCudaFunction

def regularization(args, model, regularization_active):
    logic_layers = get_layers(args, model)
    for l in logic_layers:
        l.exert_regularization = regularization_active

    if regularization_active:
        # Dropout
        if args.dropout_prob > 0:
            dropout(args,model)

def dropout(args, model):
    logic_layers = get_layers(args, model)
    if 'lgn' != args.architecture:
        raise NotImplementedError()
    dropout_mask_in = (torch.rand(logic_layers[0].in_dim) < args.dropout_prob)
    affected_indices = torch.nonzero(dropout_mask_in, as_tuple=False).squeeze(1)
    # print(f"Indices affected by dropout: {len(affected_indices)}")
    for layer in logic_layers:
        affected_indices = layer.pass_dropout(affected_indices)
    
    # print(f"Indices affected by dropout: {len(affected_indices)}")
    final_dropout_mask = torch.zeros(logic_layers[-1].weights.shape[0], dtype=torch.bool, device=logic_layers[-1].weights.device)
    final_dropout_mask[affected_indices] = True

    # Register new dropout mask (deleted after backward)
    logic_layers[-1].dropout_mask = final_dropout_mask

def setup_random_outage(args, model):
    logic_layers = get_layers(args, model)
    def apply_random_outage(module, input, output):
        if module.random_outage is None or not module.exert_regularization:
            return output
        
        # Select affected output indices
        mask = (torch.rand(output.shape[1:], device=module.device) < module.random_outage_prob)
        
        # Replace outputs
        output_intervened = output.clone()
        if module.random_outage == 'const0':
            output_intervened[:,mask] = 0
        elif module.random_outage == 'const1':
            output_intervened[:,mask] = 1
        elif module.random_outage == 'const0.5':
            output_intervened[:,mask] = 0.5
        elif module.random_outage == 'uniform':
            output_intervened[:,mask] = torch.rand_like(output_intervened[:,mask])
        elif module.random_outage == 'bernoulli':
            output_intervened[:,mask] = (torch.rand_like(output_intervened[:,mask]) > 0.5).float()
        
        # Mask gradients of affected indices accordingly
        def mask_gradients(grad):
            return grad * (~mask)
        
        output_intervened.register_hook(mask_gradients)
        return output_intervened
    
    for l in logic_layers:
        l.outage_hook_forward = l.register_forward_hook(apply_random_outage)

def setup_dropout(args, model):
    logic_layers = get_layers(args, model)
    def mask_outputs(module, input, output):
        if not module.exert_regularization:
            # Avoid dropout in evaluation
            return output

        assert hasattr(module, 'dropout_mask'), "dropout_mask not initialized."
        mask = module.dropout_mask
        if output.shape[1:] != mask.shape:
            # First dim is batch size
            return output
        masked_output = output * (~mask) 

        def mask_gradients(grad):
            return grad * (~mask)
        
        masked_output.register_hook(mask_gradients)

        return masked_output
    logic_layers[-1].dropout_hook_forward = logic_layers[-1].register_forward_hook(mask_outputs)

def setup_hooks(args, model):
    # Random outage
    if args.random_outage_prob > 0:
        setup_random_outage(args, model)

    # Register dropout hooks
    if args.dropout_prob > 0:
        setup_dropout(args, model)
