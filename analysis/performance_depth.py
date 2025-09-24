from config import * 

def parse_args():
    parser = ArgumentParser(description='Train logic gate network on the various datasets.')
    parser.add_argument('-o', '--object', type=str, default='valid_disc', help='Name of the zarr array where the measurements are stored')
    parser.add_argument('-d', '--dataset', type=str, default='cifar-100')
    parser.add_argument('-m', '--method', type=str, required=True)
    parser.add_argument('-i', '--identifier', type=str, default=None)
    args = parser.parse_args()
    return args


def performance_depth(arrays, labels, plot_name, depths=None, title=None, results_dir=DIR_ANALYSIS, x_label=r'Depth Scale Factor', y_label=r'Accuracy (\%)', legend_above_plot=True, fig_size=DEFAULT_FIG_SIZE, use_log_scale_x=False, use_log_scale_y=False, rotate_y_label=False):
    """
    Plot validation accuracy vs depth scaling for different model architectures
    in a professional style suitable for NeurIPS publication.
    
    Parameters:
    -----------
    arrays : list of list/array
        Each sub-array contains validation accuracies for different depth scales
    labels : list of str
        Labels for each architecture/model
    """
    
    # Set publication-quality style
    plt.style.use('seaborn-v0_8-whitegrid')
    palette = sns.color_palette("colorblind", n_colors=len(arrays))
    sns.set_palette(palette)
    
    # Create figure with appropriate size for publication (single column)
    fig, ax = plt.subplots(figsize=fig_size, dpi=300)
    
    # Convert to numpy arrays for easier handling
    # arrays = [np.array(arr, dtype=np.float32) for arr in arrays]
    # print(arrays)
    n_models = len(arrays)

    # Create depth scaling factors (x-axis)
    max_length = max(len(arr) for arr in arrays)
    depth_scales = np.arange(1, max_length + 1)  # 1x, 2x, 3x, 4x, 5x scaling
    
    # Plot each architecture's performance vs depth scaling
    for idx, accuracies in enumerate(arrays):
        # Create x-values for this specific architecture
        arch_depth_scales = np.arange(1, len(accuracies) + 1)
        if depths is not None:
            arch_depth_scales = np.array([depths[i] for i in range(len(accuracies))])
        
        is_list_data = isinstance(accuracies[0], (list))
        if is_list_data:
            # Compute std and mean of array and convert to tuple oneself
            accuracy_statistics = [(np.mean(np.array(arr)[~np.isnan(np.array(arr))]), np.std(np.array(arr)[~np.isnan(np.array(arr))])) for arr in accuracies]
            accuracies = accuracy_statistics
            # Ensure that: is_tuple_data = True

        is_tuple_data = isinstance(accuracies[0], (tuple))
        if is_tuple_data:
            # Split into means and stds
            means = np.array([val[0] for val in accuracies], dtype=np.float32)
            stds = np.array([val[1] for val in accuracies], dtype=np.float32)

            valid_mask = ~np.isnan(means)
            means = means[valid_mask]
            stds = stds[valid_mask]
            arch_depth_scales = arch_depth_scales[valid_mask]
            # Shaded area for std deviation
            ax.fill_between(arch_depth_scales,
                            means - stds,
                            means + stds,
                            alpha=0.2,
                            color=palette[idx],
                            linewidth=0)
            
            # Plot the mean line
            marker = 'o'
            linestyle = None
            if (idx % 2) == 1:
                marker = 's'
                linestyle = 'dashed'
            ax.plot(arch_depth_scales, means,
                    linewidth=3.5,
                    linestyle=linestyle,
                    alpha=0.8,
                    label=labels[idx],
                    marker=marker,
                    markersize=6,
                    markerfacecolor='white',
                    markeredgewidth=2)
        else:
            accuracies = np.array(accuracies)
            valid_mask = ~np.isnan(accuracies)
            # Plot with professional styling - always use markers for discrete data points
            marker = 'o'
            linestyle=None
            if (idx % 2) == 1:
                marker = 's'
                linestyle='dashed'
            ax.plot(arch_depth_scales[valid_mask], accuracies[valid_mask],
                    linewidth=3.5,
                    linestyle=linestyle,
                    alpha=0.8,
                    label=labels[idx],
                    marker=marker,
                    markersize=6,
                    markerfacecolor='white',
                    markeredgewidth=2)

    # Professional formatting
    ax.set_xlabel(f'{x_label}', fontsize=14, fontweight='bold')
    if rotate_y_label or len(y_label) < 5:
        ax.set_ylabel(f'{y_label}', fontsize=14, fontweight='bold', rotation=0, labelpad=10)
    else:
        ax.set_ylabel(f'{y_label}', fontsize=14, fontweight='bold')
    if title is not None:
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Set log scale if requested
    if use_log_scale_x:
        ax.set_xscale('log')
        
        ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
        ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs='auto', numticks=12))
        ax.xaxis.set_minor_formatter(plt.NullFormatter())  # Optional: hide minor tick labels
    else:
        # Set x-axis to show integer depth scales only
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        x_max = max_length
        if depths is not None:
            x_max = depths[max_length-1]
        ax.set_xlim(0.8, x_max + 0.2)  # Add some padding

    if use_log_scale_y:
        ax.set_yscale('log')
        
        ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs='auto', numticks=12))
        ax.yaxis.set_minor_formatter(plt.NullFormatter())
        
    # Format y-axis for accuracy (typically 0-100% or 0-1)
    all_values = np.concatenate([arr for arr in arrays])
    valid_mask = ~np.isnan(all_values)
    if len(all_values[valid_mask]) > 0:
        y_min, y_max = np.min(all_values[valid_mask]), np.max(all_values[valid_mask])
        
        if not use_log_scale_y:
            if y_max <= 1.0:
                y_range = y_max - y_min
                ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
                def percentage_formatter(x, pos):
                    return f'{x*100:.1f}%'
                ax.yaxis.set_major_formatter(FuncFormatter(percentage_formatter))
            else:
                y_range = y_max - y_min
                ax.set_ylim(y_min - 0.05 * y_range, y_max + 0.05 * y_range)


    # # Customize legend
    legend = ax.legend(loc='best', frameon=True, fancybox=True,
                      shadow=True, ncol=1, fontsize=10)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)

    # if legend_above_plot:
    #     # Above plot
    #     legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
    #                    ncol=n_models, fontsize=10, frameon=False)
    # else:
    #     # Below plot
    #     legend = ax.legend(
    #         loc='upper center',
    #         bbox_to_anchor=(0.5, -0.05),  # ↓ move it below the x-axis
    #         ncol=n_models,
    #         fontsize=10,
    #         frameon=False
    #     )
    
    # Grid styling
    # ax.grid(True, axis='y', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.grid(True, axis='x', linestyle='--', linewidth=0.5, alpha=0.3)
    ax.set_axisbelow(True)
    
    # Spine styling
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color('black')
    
    # Tick formatting
    ax.tick_params(axis='both', which='major', labelsize=12, width=1.2)
    
    # Add minor ticks for better readability
    # ax.tick_params(axis='x', which='minor', length=3, width=0.8)
    
    # Tight layout for publication
    plt.tight_layout()
    
    # Save plots
    full_path = os.path.join(results_dir, f"{plot_name}.png")
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    if SAVE_PGF:
        plt.savefig(f'{results_dir}/{plot_name}.pgf', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    if SAVE_PDF:
        plt.savefig(f'{results_dir}/{plot_name}.pdf', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    if SAVE_PNG:
        plt.savefig(f'{results_dir}/{plot_name}.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    print(f"Saved to: {full_path}")

def performance_depth_skip(arrays, labels, segments, plot_name, depths=None, title=None, results_dir=DIR_ANALYSIS, x_label=r'Depth Scale Factor', y_label=r'Accuracy (\%)', legend_above_plot=True, fig_size=DEFAULT_FIG_SIZE, use_log_scale_x=False, use_log_scale_y=False, rotate_y_label=False):
    """
    Plot validation accuracy vs depth scaling for different model architectures
    in a professional style suitable for NeurIPS publication.
    
    Parameters:
    -----------
    arrays : list of list/array
        Each sub-array contains validation accuracies for different depth scales
    labels : list of str
        Labels for each architecture/model
    """

    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    palette = sns.color_palette("colorblind", n_colors=len(arrays))
    sns.set_palette(palette)

    # Create broken y-axis figure
    fig, (ax_upper, ax_lower) = plt.subplots(2, 1,
                                             sharex=True,
                                             figsize=fig_size,
                                             dpi=300,
                                             gridspec_kw={'height_ratios': [1, 2]})
    
    axes = [ax_lower, ax_upper]
    lower_segment, upper_segment = segments

    # X values
    max_length = max(len(arr) for arr in arrays)
    depth_scales = np.arange(1, max_length + 1)
    n_models = len(arrays)

    for idx, accuracies in enumerate(arrays):
        is_tuple_data = isinstance(accuracies[0], (tuple, list))
        arch_depth_scales = np.arange(1, len(accuracies) + 1)
        if depths is not None:
            arch_depth_scales = np.array([depths[i] for i in range(len(accuracies))])

        if is_tuple_data:
            means = np.array([val[0] for val in accuracies], dtype=np.float32)
            stds = np.array([val[1] for val in accuracies], dtype=np.float32)
        else:
            means = np.array(accuracies, dtype=np.float32)
            stds = np.zeros_like(means)

        valid_mask = ~np.isnan(means)
        means = means[valid_mask]
        stds = stds[valid_mask]
        arch_depth_scales = arch_depth_scales[valid_mask]

        marker = 'o' if idx % 2 == 0 else 's'
        linestyle = None if idx % 2 == 0 else 'dashed'

        for ax in axes:
            ymin, ymax = ax.get_ylim() if ax.has_data() else (None, None)
            segment = upper_segment if ax == ax_upper else lower_segment

            # Mask values for this segment
            seg_mask = (means >= segment[0]) & (means <= segment[1])
            if not np.any(seg_mask):
                continue

            x_vals = arch_depth_scales[seg_mask]
            y_vals = means[seg_mask]
            y_errs = stds[seg_mask]

            ax.fill_between(x_vals,
                            y_vals - y_errs,
                            y_vals + y_errs,
                            alpha=0.2,
                            color=palette[idx],
                            linewidth=0)
            ax.plot(x_vals, y_vals,
                    linewidth=3.5,
                    linestyle=linestyle,
                    alpha=0.8,
                    label=labels[idx],
                    marker=marker,
                    markersize=6,
                    markerfacecolor='white',
                    markeredgewidth=2,
                    color=palette[idx])

    # Set y-limits
    ax_lower.set_ylim(*lower_segment)
    ax_upper.set_ylim(*upper_segment)

    # Hide spines
    ax_upper.spines['bottom'].set_visible(False)
    ax_lower.spines['top'].set_visible(False)

    # Break marks
    d = .01  # size of diagonal break lines in axes coordinates
    # Arguments for line style
    kwargs = dict(transform=ax_upper.transAxes, color='k', clip_on=False)
    # Top (upper axis) break marks
    ax_upper.plot((-d, +d), (-d, +d), **kwargs)        # Left diagonal
    ax_upper.plot((1 - d, 1 + d), (-d, +d), **kwargs)   # Right diagonal
    # Bottom (lower axis) break marks
    kwargs.update(transform=ax_lower.transAxes)
    ax_lower.plot((-d, +d), (1 - d, 1 + d), **kwargs)   # Left diagonal
    ax_lower.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # Right diagonal

    # Axis formatting
    ax_lower.set_xlabel(x_label, fontsize=14, fontweight='bold')
    if rotate_y_label:
        ax_lower.set_ylabel(y_label, fontsize=14, fontweight='bold', rotation=0, labelpad=10)
    else:
        ax_lower.set_ylabel(y_label, fontsize=14, fontweight='bold')

    if title:
        ax_upper.set_title(title, fontsize=14, fontweight='bold', pad=20)

    if use_log_scale_x:
        for ax in axes:
            ax.set_xscale('log')
            ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
            ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs='auto', numticks=12))
            ax.xaxis.set_minor_formatter(NullFormatter())
    else:
        ax_lower.xaxis.set_major_locator(MaxNLocator(integer=True))
        x_max = max_length
        if depths is not None:
            x_max = depths[max_length-1]
        ax_lower.set_xlim(0.8, x_max + 0.2)

    # Legend
    ax_upper.legend(loc='upper left', fontsize=10, frameon=True)

    # Grid, spines, and layout
    for ax in axes:
        ax.grid(True, axis='x', linestyle='--', linewidth=0.5, alpha=0.3)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
            spine.set_color('black')
        ax.tick_params(axis='both', which='major', labelsize=12, width=1.2)

    plt.tight_layout()

    # Save
    full_path = os.path.join(results_dir, f"{plot_name}.png")
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    if SAVE_PGF:
        plt.savefig(f'{results_dir}/{plot_name}.pgf', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    if SAVE_PDF:
        plt.savefig(f'{results_dir}/{plot_name}.pdf', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    if SAVE_PNG:
        plt.savefig(f'{results_dir}/{plot_name}.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')

    print(f"Saved to: {full_path}")

def create_performance_depth_plot(args):
    labels = None
    depths = None
    title = ''
    y_label = None
    args.experiment_notes = ''
    if args.dataset.startswith('cifar-100'):
        args.experiment_notes = 'CIFAR-100'
        title = 'CIFAR-100'
    
    y_label = r'Accuracy (\%)'
    x_label = r'Depth Scale Factor'
    if args.object == 'train_cont':
        title += ', Training Accuracy (cont.)'
    elif args.object == 'train_disc':
        title += ', Training Accuracy (disc.)'
    elif args.object == 'valid_cont':
        title += ', Validation Accuracy (cont.)'
    elif args.object == 'valid_disc':
        title += ', Validation Accuracy (disc.)'
    elif args.object == 'train_time':
        title += ', Training Time'
        y_label = 'minutes'

    title = None
    use_log_scale_x = False
    use_log_scale_y = False
    rotate_y_label = False
    segments = None
    depths = [1,2,3,4,5]
    if args.dataset == 'cifar-100':
        if args.object == 'test_disc':
            if args.identifier == 'example':
                depths = [1,2,3,4,5]
                arrays = [
                    [
                        # Model 1
                        [0.2915, 0.2734, 0.2760],
                        [0.2729, 0.2924, 0.2894],
                        [0.2976, 0.2967, 0.2983],
                        [0.2990, 0.2992, 0.3036],
                        [0.3005, 0.2995, 0.3046]
                        ],  
                    [
                        # Model 2
                        [0.2636, 0.2662, 0.2643],
                        [0.2723, 0.2750, 0.2753],
                        [0.2808, 0.2777, 0.2795],
                        [0.2817, 0.2792, 0.2797],
                        [0.2818, 0.2836, 0.2791]
                        ],
                ]
                labels = [
                "Reparametrized",
                "Original",
                ]   
                
    args.name = args.method + "/" + args.dataset + "/" 
    if args.identifier is not None and len(args.identifier) > 0:
        args.name += args.identifier + '/'
    args.name += args.object 
    if args.experiment_notes is not None and len(args.experiment_notes) > 0:
        args.name += '_' + args.experiment_notes
    kwargs = dict(
        plot_name=args.name, 
        depths=depths, 
        title=title, 
        y_label=y_label, 
        x_label=x_label, 
        use_log_scale_y=use_log_scale_y, 
        use_log_scale_x=use_log_scale_x, 
        rotate_y_label=rotate_y_label,
    )
    if segments is not None:
        performance_depth_skip(arrays, labels, segments, **kwargs)
    else:
        performance_depth(arrays, labels, **kwargs)

def performance_total(arrays, labels, plot_name, title=None, results_dir=DIR_ANALYSIS, y_label='Accuracy (%)', acc_types=None, fig_size=DEFAULT_FIG_SIZE):
    """
    Plot training accuracy (continuous), training accuracy (discrete), and validation accuracy
    for different model architectures in a professional bar chart style suitable for NeurIPS publication.
    
    Parameters:
    -----------
    arrays : list of list/array
        Each sub-array contains 3 accuracies: [train_continuous, train_discrete, validation]
    labels : list of str
        Labels for each architecture/model
    """
    n_models = len(arrays)
    n_acc_types = 3

    # Set publication-quality style
    plt.style.use('seaborn-v0_8-whitegrid')
    colors = sns.color_palette("colorblind", n_colors=len(arrays))
    # sns.set_palette(palette)
    
    # Create figure with appropriate size for publication (single column)
    fig, ax = plt.subplots(figsize=fig_size, dpi=300)
    
    # Convert to numpy arrays for easier handling
    arrays = np.array(arrays, dtype=np.float32)
    
    # Accuracy type labels
    if acc_types is None:
        acc_types = ['Train (Continuous)', 'Train (Discrete)', 'Validation (Discrete)']
    
    # Set up bar chart parameters
    bar_width = 0.2
    group_width = n_models * bar_width
    group_spacing = 2 * bar_width
    x = np.arange(n_acc_types) * (group_width + group_spacing)
    
    # Create bars for each accuracy type
    for model_idx, acc_values in enumerate(arrays):
        offset = (model_idx - (n_models - 1) / 2) * bar_width  # center groups
        x_pos = x + offset

        bars = ax.bar(x_pos, acc_values,
                      width=bar_width,
                      label=labels[model_idx],
                      color=colors[model_idx],
                      edgecolor='white',
                      linewidth=1.0,
                      alpha=0.9)

        # Add value labels
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                if height <= 1.0:
                    label = f'{height*100:.1f}%'
                else:
                    label = f'{height:.1f}%'
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                        label, ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # Professional formatting
    # ax.set_xlabel('Accuracy Type', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'{y_label}', fontsize=12, fontweight='bold')
    if title is not None:
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Set x-axis labels and positions for accuracy types
    ax.set_xticks(x)
    ax.set_xticklabels(acc_types, fontsize=10)
    
    # Format y-axis for accuracy
    all_values = np.concatenate([arr for arr in arrays])
    valid_mask = ~np.isnan(all_values)
    
    if len(all_values[valid_mask]) > 0:
        y_min, y_max = np.min(all_values[valid_mask]), np.max(all_values[valid_mask])
        
        # Check if values are in [0,1] range (convert to percentage) or already percentages
        if y_max <= 1.0:
            # Scale y-axis appropriately
            y_range = y_max - y_min
            ax.set_ylim(max(0, y_min - 0.1 * y_range), y_max + 0.15 * y_range)
            
            # Use FuncFormatter for proper percentage formatting
            def percentage_formatter(x, pos):
                return f'{x*100:.1f}%'
            ax.yaxis.set_major_formatter(FuncFormatter(percentage_formatter))
        else:
            # Assume already in percentage
            y_range = y_max - y_min
            ax.set_ylim(max(0, y_min - 0.05 * y_range), y_max + 0.1 * y_range)
    
    # Customize legend
    legend = ax.legend(loc='best', frameon=True, fancybox=True, shadow=True, 
                      ncol=1, fontsize=10)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)
    
    # Grid styling
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, axis='y')
    ax.grid(True, alpha=0.01, linestyle='-', linewidth=0.5, axis='x')
    ax.set_axisbelow(True)
    
    # Spine styling
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color('black')
    
    # Tick formatting
    ax.tick_params(axis='y', which='both', labelsize=10, width=1.2)
    ax.tick_params(axis='x', which='minor', length=0, bottom=False)
    # ax.tick_params(axis='x', which='minor', length=3, width=0.8)
    
    # Tight layout for publication
    plt.tight_layout()
    
    # Save plots
    full_path = os.path.join(results_dir, f"{plot_name}.png")
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    if SAVE_PGF:
        plt.savefig(f'{results_dir}/{plot_name}.pgf', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    if SAVE_PDF:
        plt.savefig(f'{results_dir}/{plot_name}.pdf', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    if SAVE_PNG:
        plt.savefig(f'{results_dir}/{plot_name}.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"Saved to: {full_path}")

def performance_total_multi(arrays, labels, plot_name, title=None, results_dir=DIR_ANALYSIS, y_label=r'$\mathrm{Accuracy} (\%)', acc_types=None, normalize=True, legend_above_plot=True, legend_box=False, fig_size=DEFAULT_FIG_SIZE, display_std_label=True):
    """
    Plot grouped bar charts with optional error bars for multiple metrics.
    
    Parameters:
    -----------
    arrays : list of lists or tuples
        Each element represents one model's metric values.
        Each value can be:
            - a scalar (mean)
            - a tuple (mean, std)
    labels : list of str
        Names of the models.
    plot_name : str
        Output filename (without extension).
    title : str or None
        Plot title.
    results_dir : str
        Directory to save plots.
    y_label : str or None
        Global y-axis label (optional; often not appropriate due to mixed units).
    acc_types : list of str
        Names of each metric.
    """

    # Use publication-quality style
    plt.style.use('seaborn-v0_8-whitegrid')

    n_models = len(arrays)
    n_metrics = len(arrays[0])
    if acc_types is None:
        acc_types = [f'Metric {i+1}' for i in range(n_metrics)]


    # fig_size = (1.8 * n_metrics, 4)
    scale_factor = max(3,n_metrics) / 3
    fig_size = (fig_size[0] * scale_factor, fig_size[1] * scale_factor)
    fig, ax = plt.subplots(figsize=fig_size, dpi=300)
    colors = sns.color_palette("colorblind", n_colors=n_models)

    width_scale_factor = n_models / 2
    # bar_width = 0.8 / n_models 
    bar_width = 0.4
    group_positions = np.arange(n_metrics) * width_scale_factor

    is_list_data = isinstance(arrays[0], (list))
    if is_list_data:
        # Compute std and mean of array and convert to tuple oneself
        arrays_statistics = [[(np.mean(np.array(arr)[~np.isnan(np.array(arr))]), np.std(np.array(arr)[~np.isnan(np.array(arr))])) for arr in array] for array in arrays]
        arrays = arrays_statistics
        # Ensure that: is_tuple_data = True

    # Iterate over models
    for model_idx in range(n_models):
        heights = []
        errors = []
        true_values = []
        std_provided = []
        for metric_idx in range(n_metrics):
            val = arrays[model_idx][metric_idx]
            if isinstance(val, tuple):
                std_provided.append(True)
                mean, std = val
            else:
                std_provided.append(False)
                mean, std = (val, 0.0)
            true_values.append((mean, std))
            
        # Normalize per group (metric)
        for metric_idx in range(n_metrics):
            metric_vals = []
            for m in range(n_models):
                v = arrays[m][metric_idx]
                metric_vals.append(v[0] if isinstance(v, tuple) else v)

            max_val = max(metric_vals)
            if normalize:
                norm_val = true_values[metric_idx][0] / max_val if max_val != 0 else 0
                norm_std = true_values[metric_idx][1] / max_val if max_val != 0 else 0
            else:
                norm_val = true_values[metric_idx][0]
                norm_std = true_values[metric_idx][1]

            heights.append(norm_val)
            errors.append(norm_std)

        # Bar positions
        offsets = (model_idx - (n_models - 1) / 2) * bar_width
        x_pos = group_positions + offsets

        errors_cleaned = []
        for e in errors:
            if e == 0 or e is None:
                errors_cleaned.append(np.nan)
            else:
                errors_cleaned.append(e)

        # Plot bars
        ax.bar(x_pos, heights,
               yerr=errors_cleaned,
               width=bar_width,
               color=colors[model_idx],
               edgecolor='white',
               capsize=3,
               alpha=0.9,
               label=labels[model_idx])

        # Annotate with true values
        for idx, (x, height, std_norm) in enumerate(zip(x_pos, heights, errors)):
            mean, std = true_values[idx]
            if isinstance(mean, float):
                if std_provided[idx]:
                    if mean <= 1.0 and any(pat in acc_types[idx].lower() for pat in ['cont','disc','acc']):
                        label = rf'${mean*100:.1f}$'
                        if display_std_label:
                            label += rf'$\pm {std*100:.1f}\% $'
                    else:
                        label = rf'${mean:.2f}$'
                        if display_std_label:
                            label += rf' $\pm {std:.2f}$'
                    label_y = height + std_norm 
                    if display_std_label:
                        label_y += 0.02
                else:
                    label = rf'{mean*100:.1f}\%' if (mean <= 1.0) else f'{mean:.2f}'
                    label_y = height
                ax.text(x, label_y, label, ha='center', va='bottom', fontsize=8)

    # Axes formatting
    ax.set_xticks(group_positions)
    xlabelsize=13
    if n_metrics > 3:
        xlabelsize=11

    ax.set_xticklabels(acc_types, fontsize=xlabelsize)
    if normalize:
        ax.set_ylim(0, 1.2)
        ax.set_ylabel(r'Relative Scale (per metric)', fontsize=13, fontweight='bold')
    else:
        all_values = np.concatenate([arr for arr in arrays])
        valid_mask = ~np.isnan(all_values)
        if len(all_values[valid_mask]) > 0:
            y_min, y_max = np.min(all_values[valid_mask]), np.max(all_values[valid_mask])
            
            # Check if values are in [0,1] range (convert to percentage) or already percentages
            if y_max <= 1.0:
                # Scale y-axis appropriately
                y_range = y_max - y_min
                ax.set_ylim(max(0, y_min - 0.1 * y_range), y_max + 0.15 * y_range)
                
                # Use FuncFormatter for proper percentage formatting
                def percentage_formatter(x, pos):
                    return f'{x*100:.1f}%'
                ax.yaxis.set_major_formatter(FuncFormatter(percentage_formatter))
            else:
                # Assume already in percentage
                y_range = y_max - y_min
                ax.set_ylim(max(0, y_min - 0.05 * y_range), y_max + 0.1 * y_range)
    
        ax.set_ylabel(rf'{y_label}', fontsize=13, fontweight='bold')

    if title:
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    # Grid superfluous since values are provided explicitly.
    ax.grid(False)
    
    # Legend
    legend_fontsize = 13
    legend_shift = 0.1
    if n_models <= 2:
        legend_shift += 0.02
        legend_fontsize = 15 
    elif n_models >= 4:
        legend_fontsize = 11
    if legend_above_plot:
        # Above plot
        legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05 + legend_shift),
                       ncol=n_models, fontsize=legend_fontsize, frameon=False)
    elif legend_box:
        legend = ax.legend(loc='best', frameon=True, fancybox=True,
                      shadow=True, ncol=1, fontsize=10)
    else:
        # Below plot
        legend = ax.legend(
            loc='upper center',
            bbox_to_anchor=(0.5, 0.0-legend_shift),  # ↓ move it below the x-axis
            ncol=n_models,
            fontsize=legend_fontsize,
            frameon=False
        )

    # Spines and ticks
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color('black')
    # ax.tick_params(axis='y', which='major', labelsize=10)

    # plt.tight_layout()

    full_path = os.path.join(results_dir, f"{plot_name}.png")
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    if SAVE_PGF:
        plt.savefig(f'{results_dir}/{plot_name}.pgf', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    if SAVE_PDF:
        plt.savefig(f'{results_dir}/{plot_name}.pdf', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    if SAVE_PNG:
        plt.savefig(f'{results_dir}/{plot_name}.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"Saved to: {os.path.join(results_dir, f'{plot_name}.png')}")

def create_total_performance_plot(args):
    args.experiment_notes = ''
    arrays = None
    labels = None
    title = None
    acc_types = None
    multi_metric = False
    display_std_label = True
    normalize = False
    legend_above_plot=True
    legend_box=False
    if args.dataset.startswith('cifar-100'):
        args.experiment_notes = 'CIFAR-100'
        title = 'CIFAR-100'
    
    y_label = r'Accuracy (\%)'
    
    if args.dataset == 'cifar-100':
        if args.identifier == 'example':
            arrays = [
                [(31.69, 4.21), (75.20, 1.03)],
                [(27.61, 1.87), (72.14, 1.21)],
                [(25.93, 2.17), (72.22, 0.95)],
            ]
            labels = ['SIN', 'SIG-ST', 'LIN-ST']
            title = None
            acc_types = [r'Forward pass [$\mu$s]', r'Backward pass [$\mu$s]']
            multi_metric = True
            normalize=True
            legend_above_plot=True
            
    args.name = args.method + "/" + args.dataset + "/" + args.identifier 
    if args.experiment_notes is not None and len(args.experiment_notes) > 0:
        args.name += '_' + args.experiment_notes
    if multi_metric:
        performance_total_multi(arrays, labels, plot_name=args.name, title=title, y_label=y_label, acc_types=acc_types, normalize=normalize, legend_above_plot=legend_above_plot, legend_box=legend_box, display_std_label=display_std_label)
    else:
        performance_total(arrays, labels, plot_name=args.name, title=title, y_label=y_label, acc_types=acc_types)

if __name__=='__main__':
    args = parse_args()
    args.experiment_notes = ''
    if args.dataset.startswith('cifar-100'):
        args.experiment_notes = 'CIFAR-100'
    
    if args.method == 'performance_depth':
        create_performance_depth_plot(args)        
    elif args.method == 'performance_bars':
        create_total_performance_plot(args)
    else:
        raise NotImplementedError(args.method)