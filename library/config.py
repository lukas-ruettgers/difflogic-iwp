import argparse
from math import sqrt

SEEDS = [0, 304011, 362799]

DEVICE = 'cuda'

# Plotting configs
HISTOGRAM_BINS = 20

def parse_args():
    parser = argparse.ArgumentParser(description='Configure reparametrized logic gate network.')
    # Experimental setup
    parser.add_argument('--seed', '-s', type=int, default=0, help='seed (default: 0)')

    # LGN model architecture
    parser.add_argument('--architecture', '-a', type=str, default='lgn', choices=['lgn', 'cnn'], help='Which model architecture to train.')
    parser.add_argument('--connections', type=str, default='random', choices=['random', 'unique'])
    parser.add_argument('--tau', '-t', type=float, default=10, help='the softmax temperature tau')
    parser.add_argument('--c100_scale_temp', type=int, default=None, help='Whether to scale the tuned temperature when employing the CIFAR-10 model for CIFAR-100.')
    parser.add_argument('--c100_scale_width', type=int, default=None, help='Whether to scale width and keep the tuned temperature when employing the CIFAR-10 model for CIFAR-100.')
    parser.add_argument('--num_neurons', '-k', type=int)
    parser.add_argument('--num_layers', '-l', type=int)
    ## Parametrization
    ### IWP
    parser.add_argument('--iwp', action='store_true', help='Use reparametrization')
    parser.add_argument('--op', action='store_false', dest='iwp', help='Use original parametrization')
    parser.set_defaults(iwp=True)
    parser.add_argument('--act_fn', type=str, default="SIN01")
    parser.add_argument('--skip_grad', type=bool, default=False)
    parser.add_argument('--init_shift', type=float, default=0.0)
    parser.add_argument('--init_shift_direction', type=str, default='0101', help='Heavy-tail shift directions')
    parser.add_argument('--run_op_on_iwp', type=bool, default=False)
    parser.add_argument('--run_original_cuda', type=bool, default=False)

    ## Weight Initialization
    parser.add_argument('--weights_init', type=str, default='gauss', choices=['gauss', 'ri', 'and-or', 'and-or-ri', 'uniform'], help='Weight initialization')
    parser.add_argument('--weights_init_variance', type=float, default=1.0, help='Weight initialization variance')

    # Dataset
    parser.add_argument('--dataset', '-d', type=str, choices=[
        'cifar-10',
        'cifar-100',
    ], required=True, help='Dataset')
    ## Preprocessing
    parser.add_argument('--preprocess_once', action='store_true', help='Preprocess the binary encoding of each image once.')
    parser.add_argument('--not_preprocess_once', dest='preprocess_once', action='store_false', help='Disable preprocessing once')
    parser.set_defaults(preprocess_once=True)
    parser.add_argument('--encoding', type=str, choices=[
        'real-input',
        '3-thresholds',
        '7-thresholds',
        '15-thresholds',
        '23-thresholds',
        '31-thresholds',
    ], required=True, default='3-thresholds', help='Specify the binary encoding of inputs.')
    ## Batches
    parser.add_argument('--batch-size', '-bs', type=int, default=100, help='Batch size (default=100)')
    parser.add_argument('--batches_per_backward', '-bpb', type=int, default=1, help='Number of batches accumulated per backward')
    ## Augmentation
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--no_augment', dest='augment', action='store_false')
    parser.set_defaults(augment=True)
    
    # Model scaling
    parser.add_argument('--depth_scale', '-dep', type=int, default=1)

    # Training
    parser.add_argument('--learning-rate', '-lr', type=float, default=0.01, help='Learning rate (default: 0.01)')
    parser.add_argument('--num-iterations', '-ni', type=int, default=250_000, help='Number of iterations (default: 250_000)')
    parser.add_argument('--grad-factor', type=float, default=1.)
    parser.add_argument('--optim', '-o', type=str, default='adam', choices=['adam'], help='Optimizer')
    
    # Evaluation
    parser.add_argument('--valid-set-size', '-vss', type=float, default=0., help='Fraction of the train set used for validation (default: 0.)')
    parser.add_argument('--eval-freq', '-ef', type=int, default=1_000, help='Evaluation frequency (default: 1_000)')
    parser.add_argument('--ext-eval-freq', '-eef', type=int, default=5_000, help='Exhaustive evaluation frequency (default: 5_000)')
    parser.add_argument('--eval_initial', type=bool, default=False)
    parser.add_argument('--train_eval_subsample_size', type=int, default=1000, help='Subsample size for evaluation on training dataset')
    
    # Regularization
    parser.add_argument('--resconnect', action='store_true', type=bool)
    parser.set_defaults(resconnect=False)
    parser.add_argument('--random_outage', type=str, default=None, choices=['const0','const1','const0.5','uniform','bernoulli'], help='Random gate intervention strategy') 
    parser.add_argument('--random_outage_prob', type=float, default=0) 
    parser.add_argument('--dropout_prob', type=float, default=0)

    # Logging
    parser.add_argument('--no_logging', type=bool, action=argparse.BooleanOptionalAction, help='Disable all logging')
    parser.add_argument('--log_verbose', type=str, default='', help="Concatenate all measurements you want to log: ['timing', 'act', 'gr-t', 'features', 'eval']")
    parser.add_argument('--log_timestamp', type=int, default=0, help='At what training step to log verbosely')
    parser.add_argument('--n_timing_measurements', type=int, default=20, help='The number of timing measurements to conduct')
    parser.add_argument('--wandb_verbose', type=str, default='', choices=['eval'], help='What to log to wandb')

    args = parser.parse_args()
    return args

def set_baseline_args(args):
    # Reproduce CIFAR-10 M architecture from Petersen et al. (2022)
    args.valid_set_size = 0.1
    args.learning_rate = 0.01
    args.softmax_temperature = 100
    args.depth = 4
    args.k = 128000

    # Decide model scaling for CIFAR-100
    if args.dataset == 'cifar-100':
        if args.c100_scale_temp is None:
            if 'classic' in args.architecture:
                args.c100_scale_temp = 1
        if args.c100_scale_width is None:
            if 'deep' in args.architecture:
                args.c100_scale_width = 1

        if args.c100_scale_temp:
            args.softmax_temperature *= sqrt(0.1)

    # Decrease batch size for larger models to meet memory constraints
    if args.depth_scale >= 10:
        args.batch_size //= 2
        args.batches_per_backward *= 2
    if args.depth_scale >= 20:
        args.batch_size //= 2
        args.batches_per_backward *= 2
    
    # Parametrization
    if args.iwp:
        # IWP
        args.weights_init_variance = 0.25
        if args.weights_init == 'ri':
            if args.init_shift == 0:
                # Default value
                if args.act_fn == 'SIN01':
                    args.init_shift = 1.2
                elif args.act_fn == 'sigmoid':
                    args.init_shift = 3

            if args.init_shift_direction is None:
                args.init_shift_direction = '0101'
    else:
        # OP
        if args.weights_init == 'ri':
            # OP Res Init
            args.weights_init_variance = 5.0
        else:
            # OP with normal weight initialization
            args.weights_init_variance = 1.0

