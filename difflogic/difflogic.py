import torch
import math
import difflogic_cuda, difflogic_cuda_iwp
import numpy as np
from .functional import bin_op_s, get_unique_connections, GradFactor, SinSkipGrad, SigmoidStraightThrough, LinearStraightThrough, bin_gate, weight_init
import itertools
import time

########################################################################################################################


class LogicLayer(torch.nn.Module):
    """
    The core module for differentiable logic gate networks. Provides a differentiable logic gate layer.
    """
    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            device: str = 'cuda',
            grad_factor: float = 1.,
            implementation: str = None,
            connections: str = 'random',
            weight_init_choice='gauss',
            sigma = 1.0,
            res_connect_fraction=0.0,
    ):
        """
        :param in_dim:      input dimensionality of the layer
        :param out_dim:     output dimensionality of the layer
        :param device:      device (options: 'cuda' / 'cpu')
        :param grad_factor: for deep models (>6 layers), the grad_factor should be increased (e.g., 2) to avoid vanishing gradients
        :param implementation: implementation to use (options: 'cuda' / 'python'). cuda is around 100x faster than python
        :param connections: method for initializing the connectivity of the logic gate net
        """
        super().__init__()
        self.weight_init_choice = weight_init_choice
        self.weights = torch.nn.parameter.Parameter(weight_init(
            (out_dim, 16),
            choice=self.weight_init_choice,
            sigma=sigma,
            device=device
        ))
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.device = device
        self.grad_factor = grad_factor

        """
        The CUDA implementation is the fast implementation. As the name implies, the cuda implementation is only 
        available for device='cuda'. The `python` implementation exists for 2 reasons:
        1. To provide an easy-to-understand implementation of differentiable logic gate networks 
        2. To provide a CPU implementation of differentiable logic gate networks 
        """
        self.implementation = implementation
        if self.implementation is None and device == 'cuda':
            self.implementation = 'cuda'
        elif self.implementation is None and device == 'cpu':
            self.implementation = 'python'
        assert self.implementation in ['cuda', 'python'], self.implementation

        self.connections = connections
        assert self.connections in ['random', 'unique'], self.connections
        self.indices = self.get_connections(self.connections, device)

        self.num_neurons = out_dim
        self.num_weights = out_dim

        # Grad mask (to avoid logging gradients of artificially fixed weights)
        # False = fixed, True = learned
        mask = torch.ones_like(self.weights, dtype=torch.bool)
        self.register_buffer('grad_mask', mask)

        # Res connections (This adjusts self.indices, invoke before computing the inverse connections!)
        self.res_connect_fraction = res_connect_fraction
        self.num_res_connections = int(math.ceil(res_connect_fraction*self.out_dim))
        if self.num_res_connections > 0:
            assert self.in_dim >= self.num_res_connections
            self.setup_res_connections()
        
        # Inverse connections index
        if self.implementation == 'cuda':
            """
            Defining additional indices for improving the efficiency of the backward of the CUDA implementation.
            """
            given_x_indices_of_y = [[] for _ in range(in_dim)]
            indices_0_np = self.indices[0].cpu().numpy()
            indices_1_np = self.indices[1].cpu().numpy()
            for y in range(out_dim):
                given_x_indices_of_y[indices_0_np[y]].append(y)
                given_x_indices_of_y[indices_1_np[y]].append(y)
            self.given_x_indices_of_y_start = torch.tensor(
                np.array([0] + [len(g) for g in given_x_indices_of_y]).cumsum(), device=device, dtype=torch.int64)
            self.given_x_indices_of_y = torch.tensor(
                [item for sublist in given_x_indices_of_y for item in sublist], dtype=torch.int64, device=device)
        
    # Dropout
    def pass_dropout(self, affected_input_indices):
        if isinstance(affected_input_indices, list):
            affected_input_indices = torch.tensor(affected_input_indices, dtype=torch.long, device=self.device)
        starts = self.given_x_indices_of_y_start[affected_input_indices]
        ends = self.given_x_indices_of_y_start[affected_input_indices + 1]
        counts = ends - starts

        total = counts.sum()

        # Create a flat index array of all affected outputs
        # Step 1: repeat start indices according to how many outputs each input connects to
        repeated_starts = torch.repeat_interleave(starts, counts)

        # Step 2: create relative offsets within each connection block
        relative_offsets = torch.arange(total, device=self.device) - \
                        torch.repeat_interleave(counts.cumsum(0) - counts, counts)

        # Step 3: compute final indices into self.given_x_indices_of_y
        affected_output_indices = self.given_x_indices_of_y[repeated_starts + relative_offsets]
        affected_output_indices = torch.unique(affected_output_indices)

        return affected_output_indices

    def remove_dropout_mask(self):
        if hasattr(self, "dropout_mask"):
            del self.dropout_mask

    # Residual connections
    def setup_res_connections(self):
        # Fix connections
        a,b = self.indices
        a[0:self.num_res_connections] = torch.arange(self.num_res_connections, device=self.device).long()
        self.indices = a, b

        # Fix gates
        self.grad_mask[0:self.num_res_connections, :] = False
        self.fix_res_connect_gates()

    def fix_res_connect_gates(self):
        # Fix logic gate to feedforward A
        with torch.no_grad():
            self.weights[0:self.num_res_connections, :] = -10
            self.weights[0:self.num_res_connections, 3] = 10

    def reset_fixed_weights(self):
        self.fix_res_connect_gates()
        self.remove_dropout_mask()

    def apply_grad_mask(self):
        # Apply mask to gradients after backward pass
        if self.w_00 is not None and self.weights.grad is not None:
            self.weights.grad *= self.grad_mask

    #### Original forward code, without PackBitsTensor for simplicity
    def forward(self, x):
        if self.grad_factor != 1.:
            x = GradFactor.apply(x, self.grad_factor)

        if self.implementation == 'cuda':
            return self.forward_cuda(x)
        elif self.implementation == 'python':
            return self.forward_python(x)
        else:
            raise ValueError(self.implementation)

    def forward_python(self, x):
        assert x.shape[-1] == self.in_dim, (x[0].shape[-1], self.in_dim)

        if self.indices[0].dtype == torch.int64 or self.indices[1].dtype == torch.int64:
            print(self.indices[0].dtype, self.indices[1].dtype)
            self.indices = self.indices[0].long(), self.indices[1].long()
            print(self.indices[0].dtype, self.indices[1].dtype)

        a, b = x[..., self.indices[0]], x[..., self.indices[1]]
        if self.training:
            x = bin_op_s(a, b, torch.nn.functional.softmax(self.weights, dim=-1))
        else:
            weights = torch.nn.functional.one_hot(self.weights.argmax(-1), 16).to(torch.float32)
            x = bin_op_s(a, b, weights)
        return x

    def forward_cuda(self, x):
        if self.training:
            assert x.device.type == 'cuda', x.device
        assert x.ndim == 2, x.ndim

        x = x.transpose(0, 1)
        x = x.contiguous()

        assert x.shape[0] == self.in_dim, (x.shape, self.in_dim)

        a, b = self.indices

        if self.training:
            w = torch.nn.functional.softmax(self.weights, dim=-1).to(x.dtype)
            x = LogicLayerCudaFunction.apply(
                x, a, b, w, self.given_x_indices_of_y_start, self.given_x_indices_of_y
            ).transpose(0, 1)
        else:
            w = torch.nn.functional.one_hot(self.weights.argmax(-1), 16).to(x.dtype)
            with torch.no_grad():
                x = LogicLayerCudaFunction.apply(
                    x, a, b, w, self.given_x_indices_of_y_start, self.given_x_indices_of_y
                ).transpose(0, 1)
        
        return x

    def get_connections(self, connections, device='cuda'):
        assert self.out_dim * 2 >= self.in_dim, 'The number of neurons ({}) must not be smaller than half of the ' \
                                                'number of inputs ({}) because otherwise not all inputs could be ' \
                                                'used or considered.'.format(self.out_dim, self.in_dim)
        if connections == 'random':
            c = torch.randperm(2 * self.out_dim) % self.in_dim
            c = torch.randperm(self.in_dim)[c]
            c = c.reshape(2, self.out_dim)
            a, b = c[0], c[1]
            a, b = a.to(torch.int64), b.to(torch.int64)
            a, b = a.to(device), b.to(device)
            return a, b
        elif connections == 'unique':
            return get_unique_connections(self.in_dim, self.out_dim, device)
        else:
            raise ValueError(connections)

    def extra_repr(self):
        return '{}, {}{}{}{}'.format(self.in_dim, self.out_dim, 
            ', train' if self.training else 'eval',
            f', weights_init={self.weight_init_choice}',
            f', num_resconnections={self.num_res_connections}' if self.num_res_connections > 0 else '',
        )

class LogicLayerIWP(LogicLayer):
    """
    Logic Layer with input-wise Parametrization
    The core module for differentiable logic gate networks. Provides a differentiable logic gate layer.
    """
    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            device: str = 'cuda',
            grad_factor: float = 1.,
            implementation: str = None,
            connections: str = 'random',
            weight_init_choice='gauss',
            sigma = 1.0,
            act_fn = torch.sigmoid,
            init_shift=0.0,
            init_shift_direction=None,
            res_connect_fraction=0.0,
            skip_grad=False,
            random_outage=None,
            random_outage_prob=0.0,
            run_op_on_iwp=False,
            run_original_cuda=False,
    ):
        self.act_fn_str = act_fn
        super().__init__(in_dim, out_dim, 
            device=device,
            weight_init_choice=weight_init_choice,
            sigma=sigma,
            grad_factor=grad_factor,
            implementation=implementation,
            connections=connections,
            res_connect_fraction=res_connect_fraction,
            )

        # Binary output estimator
        self.act_fn = None
        if self.act_fn_str == "SIN01":
            self.act_fn = self.sin_act
        elif self.act_fn_str == 'sigmoid':
            self.act_fn = torch.sigmoid
        elif self.act_fn_str == 'sigmoid-st':
            self.act_fn = self.sigmoid_st
        elif self.act_fn_str == 'sin-st':
            self.act_fn = self.sin_st
        elif self.act_fn_str == 'linear':
            self.act_fn = self.linear_act

        self.run_op_on_iwp = run_op_on_iwp
        # Input-wise parametrization, only 4 per neuron needed
        if not self.run_op_on_iwp:
            self.weights = torch.nn.parameter.Parameter(weight_init(
                (out_dim, 4),
                choice='gauss',
                sigma=sigma,
                device=device
            ))
        
            # Reinit grad mask
            mask = torch.ones_like(self.weights, dtype=torch.bool)
            self.register_buffer('grad_mask', mask)

            # init/Residual initialization
            self.init_shift = init_shift
            self.init_shift_direction = init_shift_direction
            if self.init_shift != 0:
                assert init_shift_direction is not None, init_shift_direction
                self.shift_init(self.weight_init_choice, self.init_shift, self.init_shift_direction)

            # Res connections
            # Indices remain unchanged, but weights were reinitialized, so fix them again
            if self.num_res_connections>0:
                self.grad_mask[0:self.num_res_connections, :] = False
                self.fix_res_connect_gates()

        # Skip gradients
        self.skip_grad = skip_grad

        # Dropout
        self.random_outage=random_outage
        self.random_outage_prob=random_outage_prob
        
    
    # Custom binary output estimators
    def sin_act(self, x):
        return 0.5 + 0.5 * torch.sin(x)

    def sin_st(self, x):
        return SinSkipGrad.apply(x)

    def linear_act(self, x):
        return LinearStraightThrough.apply(x)
    
    def sigmoid_st(self, x):
        return SigmoidStraightThrough.apply(x)

    # Override the emulation of residual connections
    def fix_res_connect_gates(self):
        # Fix logic gate to feedforward A
        with torch.no_grad():
            if self.act_fn_str == "SIN01":
                self.weights[0:self.num_res_connections, 0] = -torch.pi/2
                self.weights[0:self.num_res_connections, 1] = -torch.pi/2
                self.weights[0:self.num_res_connections, 2] = torch.pi/2
                self.weights[0:self.num_res_connections, 3] = torch.pi/2
            elif self.act_fn_str == 'sigmoid':
                self.weights[0:self.num_res_connections, 0] = -10
                self.weights[0:self.num_res_connections, 1] = -10
                self.weights[0:self.num_res_connections, 2] = 10
                self.weights[0:self.num_res_connections, 3] = 10
            else:
                raise NotImplementedError(self.act_fn_str)
    
    
    # Heavy Tail Initializations for the IWP
    def shift_init(self, init, shift, shift_combination):
        with torch.no_grad():
            M = self.weights.shape[0]
            if init is None or init == 'ri':
                for i in range(4):
                    if shift_combination[i] == '0':
                        self.weights[..., i] -= shift
                    else:
                        self.weights[..., i] += shift
            else:
                if init == 'ri':
                    K = 1
                    sign_patterns = torch.tensor([
                        [-1, -1, 1, 1], # A
                    ], dtype=self.weights.dtype)
                if init == 'and-or':
                    K = 2
                    sign_patterns = torch.tensor([
                        [-1, 1, 1, 1], # OR
                        [-1, -1, -1, 1], # AND
                    ], dtype=self.weights.dtype)
                elif init == 'and-or-ri':
                    K = 4
                    sign_patterns = torch.tensor([
                        [-1, 1, 1, 1], # OR
                        [-1, -1, -1, 1], # AND
                        [-1, -1, 1, 1], # A
                        [-1, 1, -1, 1], # B
                    ], dtype=self.weights.dtype)
                elif init == 'uniform':
                    K = 16
                    bool_outputs = torch.tensor(list(itertools.product([0, 1], repeat=4)), dtype=torch.float32)
                    sign_patterns = 2 * bool_outputs - 1
                else:
                    raise NotImplementedError(init)
                probs = np.full(K, 1 / K)
                choices = np.random.choice(K, size=M, p=probs)
                signs_tensor = sign_patterns.to(self.device, dtype=self.weights.dtype)[choices]
                self.weights += shift * signs_tensor  
                

    def extra_repr(self):
        if self.run_op_on_iwp:
            return super().extra_repr()
        return super().extra_repr() + '{}{}{}{}'.format( 
            f', init_shift={self.init_shift} ({self.init_shift_direction})' if self.init_shift != 0 else '',
            f', act_fn={self.act_fn_str}',
            f', skip_grad' if self.skip_grad else '',
            f', random_outage={self.random_outage} (prob={self.random_outage_prob})' if self.random_outage is not None else '',
        )

    # Adjust forward pass
    def forward_python(self, x):
        assert x.shape[-1] == self.in_dim, (x[0].shape[-1], self.in_dim)

        if self.indices[0].dtype == torch.int64 or self.indices[1].dtype == torch.int64:
            # NOTE: torch.int64 == torch.long.
            # print(self.indices[0].dtype, self.indices[1].dtype)
            self.indices = self.indices[0].long(), self.indices[1].long()
            # print(self.indices[0].dtype, self.indices[1].dtype)

        a, b = x[..., self.indices[0]], x[..., self.indices[1]]
        if self.training:
            x = bin_gate(a, b, self.act_fn(self.weights))
        else:
            weights = (self.act_fn(self.weights) >= 0.5).to(x.dtype)
            x = bin_gate(a, b, weights)
        return x

    def forward_cuda(self, x):
        if self.training:
            assert x.device.type == 'cuda', x.device
        assert x.ndim == 2, x.ndim

        x = x.transpose(0, 1)
        x = x.contiguous()

        assert x.shape[0] == self.in_dim, (x.shape, self.in_dim)

        a, b = self.indices

        if self.run_op_on_iwp:
            if self.training:
                w = torch.nn.functional.softmax(self.weights, dim=-1).to(x.dtype)
                x = LogicLayerCudaFunction.apply(
                    x, a, b, w, self.given_x_indices_of_y_start, self.given_x_indices_of_y
                ).transpose(0, 1)
            else:
                w = torch.nn.functional.one_hot(self.weights.argmax(-1), 16).to(x.dtype)
                with torch.no_grad():
                    x = LogicLayerCudaFunction.apply(
                        x, a, b, w, self.given_x_indices_of_y_start, self.given_x_indices_of_y
                    ).transpose(0, 1)
        else:
            if self.training:
                w = self.act_fn(self.weights).to(x.dtype)
                x = IWPLogicLayerCudaFunction.apply(
                    x, a, b, w, self.given_x_indices_of_y_start, self.given_x_indices_of_y
                ).transpose(0, 1)
            else:
                with torch.no_grad():
                    w = torch.round(self.act_fn(self.weights)).to(x.dtype)
                    x = IWPLogicLayerCudaFunction.apply(
                        x, a, b, w, self.given_x_indices_of_y_start, self.given_x_indices_of_y
                    ).transpose(0, 1)
        
        return x
    
########################################################################################################################


class GroupSum(torch.nn.Module):
    """
        The GroupSum module.
    """
    def __init__(self, k: int, tau: float = 1., device='cuda', n_splits: int = 1, for_each_split: bool = False, 
        use_bias: bool = False, 
        use_weights: bool = False, 
        gs_flip: int = 0, 
        in_dim=None):
        """
        :param k: number of intended real valued outputs, e.g., number of classes
        :param tau: the (softmax) temperature tau. The summed outputs are divided by tau.
        :param device: The device on which the computations will be performed.
        :param n_splits: The number of splits.
        :param for_each_split: Indicates whether to use individual sums for each split mode.
        :param use_bias: Whether to add bias to the output logits.
        :param use_weights: Whether to use learnable weights for each activation.
        :param in_dim: The input dimension (last dimension of input tensor).
        """
        super().__init__()
        self.k = k
        self.tau = tau
        self.device = device
        self.n_splits = n_splits
        self.for_each_split = for_each_split
        self.use_bias = use_bias
        self.use_weights = use_weights
        self.in_dim = in_dim
        self.gs_flip = gs_flip

        if self.use_bias:
            if self.for_each_split:
                # Bias shape: (n_splits, k) for individual splits
                self.bias = torch.nn.parameter.Parameter(torch.zeros(self.n_splits, self.k, device=device))
            else:
                # Bias shape: (k,) for summed across splits
                self.bias = torch.nn.parameter.Parameter(torch.zeros(self.k, device=device))
        else:
            self.register_parameter('bias', None)
        if self.use_weights:
            if self.in_dim is None:
                raise ValueError("in_dim must be provided when use_weights=True")
            self.weights = torch.nn.parameter.Parameter(torch.ones(self.in_dim, device=device))
        else:
            self.register_parameter('weights', None)

        if self.gs_flip > 0:
            assert in_dim is not None
            flip_mask = ((torch.arange(in_dim, device=device)) % self.gs_flip == 0)
            self.register_buffer('flip_mask', flip_mask)

    def forward(self, x):
        """
        Perform the forward computation of the group sum operation.

        :param x: The input tensor.
        :return: The output tensor after the group sum operation.
        """
        if self.use_weights and self.weights is not None:
            assert x.shape[-1] == self.weights.shape[-1], f"x.shape: {x.shape}, self.weights.shape: {self.weights.shape}" 
            x = x * self.weights

        assert x.shape[-1] % self.k == 0, (x.shape, self.k)
        assert x.shape[-1] % self.n_splits == 0, (x.shape, self.n_splits)

        if self.gs_flip > 0:
            x = self.flip(x)

        if self.for_each_split:
            logits = x.reshape(*x.shape[:-1], self.n_splits, self.k, x.shape[-1] // (self.k * self.n_splits)).sum(-1) / self.tau
        else:
            assert x.shape[-1] % (self.k*self.n_splits) == 0, (x.shape, self.k, self.n_splits)
            logits = x.reshape(*x.shape[:-1], self.n_splits, self.k, x.shape[-1] // (self.k * self.n_splits)).sum(-1).sum(-2) / self.tau

        if self.use_bias and self.bias is not None:
            assert logits.shape[-1] == self.bias.shape[-1], f"logits.shape: {logits.shape}, self.bias.shape: {self.bias.shape}" 
            logits = logits + self.bias

        # print(f"GS Logits: {logits}")
        return logits

    def flip(self, x):
        # Flip every kth input
        assert x.shape[-1] == self.flip_mask.shape[-1], (x.shape, self.flip_mask.shape)
        
        return x * (~self.flip_mask) + (1-x)*self.flip_mask


    def extra_repr(self):
        return 'k={}, tau={}, device={}, {}{}{}{}{}'.format(
            self.k, self.tau, self.device,
            'ns={}, '.format(self.n_splits) if self.n_splits > 1 else '',
            'for_each_split, ' if self.for_each_split else '',
            'bias, ' if self.use_bias else 'no_bias, ',
            'weights' if self.use_weights else 'no_weights',
            ', gs_flip={}'.format(self.gs_flip) if self.gs_flip != 0 else '',
        )

########################################################################################################################

class LogicLayerCudaFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, a, b, w, given_x_indices_of_y_start, given_x_indices_of_y):
        ctx.save_for_backward(x, a, b, w, given_x_indices_of_y_start, given_x_indices_of_y)
        return difflogic_cuda.forward(x, a, b, w)

    @staticmethod
    def backward(ctx, grad_y):
        cls = LogicLayerCudaFunction

        if hasattr(cls, 'backward_count') and cls.backward_count < cls.timing_measurements:
            torch.cuda.synchronize()
            start = time.perf_counter()

        x, a, b, w, given_x_indices_of_y_start, given_x_indices_of_y = ctx.saved_tensors
        grad_y = grad_y.contiguous()

        grad_w = grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x = difflogic_cuda.backward_x(x, a, b, w, grad_y, given_x_indices_of_y_start, given_x_indices_of_y)
        if ctx.needs_input_grad[3]:
            grad_w = difflogic_cuda.backward_w(x, a, b, grad_y)
        if hasattr(cls, 'backward_count') and cls.backward_count < cls.timing_measurements:
            torch.cuda.synchronize()
            end = time.perf_counter()
            duration_ns = 1000000 *(end - start)
            cls.backward_count += 1
            cls.backward_time += duration_ns * cls.timing_measurements_factor
            cls.backward_times.append(duration_ns)
            # print(f"Backward ns: {duration_ns}")
        
        return grad_x, None, None, grad_w, None, None, None


class IWPLogicLayerCudaFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, a, b, w, given_x_indices_of_y_start, given_x_indices_of_y):
        ctx.save_for_backward(x, a, b, w, given_x_indices_of_y_start, given_x_indices_of_y)
        return difflogic_cuda_iwp.iwp_forward(x, a, b, w)

    @staticmethod
    def backward(ctx, grad_y):
        cls = IWPLogicLayerCudaFunction

        if hasattr(cls, 'backward_count') and cls.backward_count < cls.timing_measurements:
            torch.cuda.synchronize()
            start = time.perf_counter()
        
        x, a, b, w, given_x_indices_of_y_start, given_x_indices_of_y = ctx.saved_tensors
        grad_y = grad_y.contiguous()

        grad_w = grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x = difflogic_cuda_iwp.iwp_backward_x(x, a, b, w, grad_y, given_x_indices_of_y_start, given_x_indices_of_y)
        if ctx.needs_input_grad[3]:
            grad_w = difflogic_cuda_iwp.iwp_backward_w(x, a, b, grad_y)
        if hasattr(cls, 'backward_count') and cls.backward_count < cls.timing_measurements:
            torch.cuda.synchronize()
            end = time.perf_counter()
            duration_ns = 1000000 *(end - start)
            cls.backward_count += 1
            cls.backward_time += duration_ns * cls.timing_measurements_factor
            cls.backward_times.append(duration_ns)
            # print(f"Backward ns: {duration_ns}")
        
        return grad_x, None, None, grad_w, None, None, None



########################################################################################################################
