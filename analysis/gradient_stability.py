from config import * 

def parse_args():
    parser = ArgumentParser(description='Train logic gate network on the various datasets.')
    parser.add_argument('-o', '--object', type=str, default='grads_initial')
    parser.add_argument('-n', '--notes', type=str, default=None)
    args = parser.parse_args()
    return args

def stability(arrays, labels, fname='gradient_stability', plot_dir=DIR_ANALYSIS, title=None, distinct_linestyles=True, fig_size=DEFAULT_FIG_SIZE, legend_above_plot=True):
    """
    Plot gradient norms over layers of a neural network in a professional style
    suitable for NeurIPS publication.
    
    Parameters:
    -----------
    arrays : list of list/array
        Each sub-array contains gradient norms for one layer over time
    """
    
    # Set publication-quality style
    plt.style.use('seaborn-v0_8-whitegrid')
    palette = sns.color_palette("colorblind", n_colors=len(arrays))
    sns.set_palette(palette)
    
    # Create figure with appropriate size for publication (single column)
    fig, ax = plt.subplots(figsize=fig_size, dpi=300)
    
    # Convert to numpy arrays for easier handling
    arrays = [np.array(arr) for arr in arrays]
    n_models = len(arrays)

    # Create time steps
    max_length = max(len(arr) for arr in arrays)
    time_steps = np.arange(max_length)
    
    # Plot each layer's gradient norms
    for idx, gradient_norms in enumerate(arrays):
        # Handle arrays of different lengths by padding with NaN
        if len(gradient_norms) < max_length:
            padded_norms = np.full(max_length, np.nan)
            padded_norms[:len(gradient_norms)] = gradient_norms
            gradient_norms = padded_norms
            
        # Plot with professional styling
        # op = labels[idx] in ["OP Res Init", "OP"]
        # iwp = 'IWP' in labels[idx]
        # nap = 'NAP' in labels[idx]
        op = 'OP' in labels[idx]
        ri = 'RI' in labels[idx]
        linestyle = 'solid'
        if distinct_linestyles:
            if op and not ri:
                linestyle = 'dotted'
            if not op and ri:
                linestyle = 'dashed'
            if op and ri:
                linestyle = 'dashdot'
        
        markers = ['o', 's', 'p', 'h', 'D', 'v', '^']
        ax.plot(time_steps, gradient_norms, 
                linewidth=3.5, 
                linestyle=linestyle,
                alpha=0.8,
                label=labels[idx],
                color=palette[idx],
                marker=markers[idx] if not distinct_linestyles else None,
                markevery=len(gradient_norms)//10,
                markersize=5)
    
    # Professional formatting
    ax.set_xlabel('Layer', fontsize=15, fontweight='bold')
    ax.set_ylabel('Gradient Norm', fontsize=15, fontweight='bold')
    if title is not None:
        ax.set_title(title, fontsize=17, fontweight='bold', pad=20)
    
    # Set y-axis to log scale if values span multiple orders of magnitude
    grad_values = np.concatenate([arr[~np.isnan(arr)] for arr in arrays])
    
    if len(grad_values) > 0:
        value_range = np.max(grad_values) / np.max([np.min(grad_values), 1e-10])
        if value_range > 100:
            ax.set_yscale('log')
            ax.set_ylabel('Gradient Norm (log scale)', fontsize=15, fontweight='bold')
    
    # Customize legend
    legend = ax.legend(loc='best', frameon=True, fancybox=True, 
                      shadow=True, ncol=1, fontsize=14)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)
    # if legend_above_plot:
    #     # Above plot
    #     legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
    #                    ncol=n_models, fontsize=14, frameon=False)
    # else:
    #     # Below plot
    #     legend = ax.legend(
    #         loc='upper center',
    #         bbox_to_anchor=(0.5, -0.05),  # ↓ move it below the x-axis
    #         ncol=n_models,
    #         fontsize=14,
    #         frameon=False
    #     )

    # Grid styling
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Spine styling
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color('black')
    
    # Tick formatting
    ax.tick_params(axis='both', which='major', labelsize=10, width=1.2)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
    
    # Tight layout for publication
    plt.tight_layout()

    full_path = os.path.join(plot_dir, f"{fname}.png")
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    if SAVE_PGF:
        plt.savefig(f'{results_dir}/{fname}.pgf', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    if SAVE_PDF:
        plt.savefig(f'{results_dir}/{fname}.pdf', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    if SAVE_PNG:
        plt.savefig(f'{results_dir}/{fname}.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"Saved plot to {full_path}.")
    print(f"Saved plot to {full_path}.")

def get_eids(args):
    # Example
    if args.notes == 'example':
        eids = [
            ("run_name1", "Model 1"),
            ("run_name2", "Model 2"),
        ]
        raise NotImplementedError("This is only an example.")
    return eids    

if __name__=='__main__':
    args = parse_args()
    eids = get_eids(args)

    arrays = [load_zarr(args.object, plot_dir=os.path.join(DIR_ANALYSIS, eid)) for eid, _ in eids]
    labels = [label for _, label in eids]

    name = args.method + "/" + args.notes + "/" + args.object
    stability(arrays, labels, fname=name, plot_dir = DIR_ANALYSIS, title=None)
    