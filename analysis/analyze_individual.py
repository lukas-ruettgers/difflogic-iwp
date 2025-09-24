from config import * 

latex_labels = [
    "0",                  # 0
    "A ∧ B",              # 1
    "¬(A → B)",           # 2
    "A",                  # 3
    "¬(B → A)",           # 4
    "B",                  # 5
    "A ⊕ B",              # 6
    "A ∨ B",              # 7
    "¬(A ∨ B)",           # 8
    "¬(A ⊕ B)",           # 9
    "¬B",                 # 10
    "B → A",              # 11
    "¬A",                 # 12
    "A → B",              # 13
    "¬(A ∧ B)",           # 14
    "1"                   # 15
]

def parse_args():
    parser = ArgumentParser(description='Train logic gate network on the various datasets.')
    parser.add_argument('-eid', '--experiment_id', type=str, default=None)
    parser.add_argument('-o', '--object', type=str, required=True)
    parser.add_argument('-m', '--method', type=str, required=True)
    parser.add_argument('-n', '--notes', type=str, default=None)
    args = parser.parse_args()
    return args

def plot_activation_histogram_heatmap(a, plot_name, title='Intermediate Feature Distribution Over Layers', plot_dir=DIR_ANALYSIS, normalize=True, fig_size=DEFAULT_FIG_SIZE):
    """
    Generate a heatmap of histogram activations over time steps.

    Parameters:
    -----------
    a : np.ndarray
        Array of shape (N, 100) representing activation histograms over time.
    plot_name : str
        Filename prefix to save the figure.
    title : str
        Title of the plot.
    plot_dir : str
        Directory to save the plots.
    normalize : bool
        Whether to normalize each histogram to sum to 1.
    """
    # Set professional style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    a = np.array(a, dtype=np.float32)
    # Normalize histograms if needed
    if normalize:
        a = a / (a.sum(axis=1, keepdims=True) + 1e-8)  # Avoid division by zero

    N, num_bins = a.shape
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)

    # Create figure
    fig, ax = plt.subplots(figsize=fig_size, dpi=300)

    # Use imshow with proper aspect ratio and extent
    X, Y = np.meshgrid(np.arange(N + 1), bin_edges)
    im = ax.pcolormesh(X, Y, a.T, cmap='plasma', shading='auto')


    # Axis labels and title
    if N <= 10:
        step = 1
    else:
        step = (N // 10) + 1
    ax.set_xticks(np.arange(N) + 0.5)           # Shift by 0.5 to center on each bin
    tick_labels = [str(i) if i % step == 0 else '' for i in range(N)]
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel('Logic Layer', fontsize=15, fontweight='bold')
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_yticklabels(['0', '0.5', '1'], fontsize=13)
    ax.set_ylabel('Activation Value', fontsize=15, fontweight='bold')

    if title:
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, pad=0.01, aspect=30)
    cbar.set_label('Density', fontsize=15, fontweight='bold')
    cbar.ax.tick_params(labelsize=13)
    cbar.solids.set_rasterized(False)

    # Axis formatting
    ax.tick_params(axis='both', which='major', labelsize=13, width=1.2)
    ax.tick_params(axis='x', which='minor', length=0)
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color('black')

    # Grid styling
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5, axis='y')
    ax.grid(True, alpha=0.01, linestyle='-', linewidth=0.5, axis='x')
    ax.set_axisbelow(True)

    # Tight layout for publication
    plt.tight_layout()

    # Save the figure
    full_path = os.path.join(plot_dir, f"{plot_name}.png")
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    if SAVE_PGF:
        plt.savefig(f'{plot_dir}/{plot_name}.pgf', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    if SAVE_PDF:
        plt.savefig(f'{plot_dir}/{plot_name}.pdf', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    if SAVE_PNG:
        plt.savefig(f'{plot_dir}/{plot_name}.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()

    print(f"Saved heatmap to: {full_path}")

if __name__=='__main__':
    args = parse_args()

    if args.experiment_id is None:
        raise NotImplementedError("The folder name of the run must be provided")

    a = load_zarr(args.object, DIR_RESULTS=os.path.join(DIR_RESULTS, args.experiment_id))

    fname = args.method + "/" + args.object + "/"
    if args.notes is not None:
        fname += args.notes + '_' 
    fname += args.experiment_id
    if 'feature-histogram' in args.method:
        plot_activation_histogram_heatmap(a, plot_name=fname, title=None)
    else:
        raise NotImplementedError(args.method)