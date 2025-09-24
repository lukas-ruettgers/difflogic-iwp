from config import * 

def partials_dgi_dA(B):
    """Returns an array of ∂gᵢ/∂A for i = 0 to 15, evaluated at B."""
    return np.array([
        0,
        B,
        1 - B,
        1,
        -B,
        0,
        1 - 2 * B,
        1 - B,
        -1 + B,
        -1 + 2 * B,
        0,
        B,
        -1,
        -1 + B,
        -B,
        0
    ])


def parse_args():
    parser = ArgumentParser(description='Empirical histograms.')
    parser.add_argument('-o', '--object', type=str, required=True)
    parser.add_argument('-N', '--N', type=int, default=10000)
    args = parser.parse_args()
    return args

def compute_transformation_histogram(N, sigma, y, plot_name, fig_size=DEFAULT_FIG_SIZE):
    """
    Sample N pairs of normal random variables and compute (e^X - e^Y)/(e^X + e^Y)
    
    Parameters:
    N: number of samples
    sigma: standard deviation of the normal distribution
    """
    
    # Sample N pairs of i.i.d. normal random variables with variance sigma^2
    samples = np.random.normal(loc=0, scale=sigma, size=(N, 16))
    
    # Extract X and Y
    exp_samples = np.exp(samples)
    exp_samples_sum = np.sum(exp_samples, axis=-1)
    derivatives = partials_dgi_dA(B=y)
    weighted_exp_samples = exp_samples * derivatives
    weighted_exp_samples = np.sum(weighted_exp_samples, axis=-1)
    
    # Compute (e^X - e^Y) / (e^X + e^Y)
    transformation = (weighted_exp_samples) / (exp_samples_sum)
    
    # Create professional histogram plot with seaborn styling
    fig, ax = plt.subplots(figsize=fig_size, dpi=300)
    
    # Use seaborn's color palette for professional appearance
    # colors = sns.color_palette("husl", n_colors=3)
    colors = sns.color_palette("colorblind", n_colors=3)
    
    histogram_color = colors[0]
    mean_color = colors[1]
    std_color = colors[2]
    
    mean_val = np.mean(transformation)
    std_val = np.std(transformation)
    
    # Add standard deviation shaded area first
    std_label = rf'$\pm$1 Std Dev ({std_val:.3f})'
    std_label = rf'$\pm$1 Std Dev'
    ax.axvspan(mean_val - std_val, mean_val + std_val,
            color=std_color, alpha=0.2,
            label=std_label)
    # Optional: Add dashed lines at ±1 std dev
    # ax.axvline(mean_val - std_val, color=std_color, linestyle=':', linewidth=1.5)
    # ax.axvline(mean_val + std_val, color=std_color, linestyle=':', linewidth=1.5)
    
    # Create histogram with professional styling
    n, bins, patches = ax.hist(transformation, bins=50, density=True,
                              alpha=0.8, color=histogram_color, edgecolor='white',
                              linewidth=0.8)
    
    # Add mean line with contrasting color
    # accent_color = sns.color_palette("husl", n_colors=3)[1]  # Get a contrasting color
    ax.axvline(mean_val, color=mean_color, linestyle='--', linewidth=2.5,
               label=rf'Mean = {mean_val:.3f}', alpha=0.9)
    
    
    # Professional formatting
    # ax.set_xlabel('Transformation Value', fontsize=12, fontweight='bold')
    ax.set_ylabel(r'Density', fontsize=13, fontweight='bold')
    # ax.set_title(f'Empirical Distribution (N = {N:,}, σ = {sigma})', fontsize=14, fontweight='bold', pad=20)
    # Use FuncFormatter for proper percentage formatting
    def percentage_formatter(x, pos):
        return rf'{x:.1f}\%'
    ax.yaxis.set_major_formatter(FuncFormatter(percentage_formatter))

    # Set axis limits for better presentation
    ax.set_xlim(-1.05, 1.05)
    
    # Customize legend with seaborn styling
    legend = ax.legend(loc='center right', frameon=True, fancybox=True, 
                      shadow=True, fontsize=12)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)
    
    # Grid styling (seaborn already provides nice grids)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Spine styling
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color('black')
    
    # Tick formatting
    ax.tick_params(axis='both', which='major', labelsize=11, width=1.2)
    
    # Tight layout for publication
    plt.tight_layout()
    
    # Save the figure
    full_path = os.path.join(DIR_ANALYSIS, f"{plot_name}.png")
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    if SAVE_PGF:
        plt.savefig(f'{DIR_ANALYSIS}/{plot_name}.pgf', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    if SAVE_PDF:
        plt.savefig(f'{DIR_ANALYSIS}/{plot_name}.pdf', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    if SAVE_PNG:
        plt.savefig(f'{DIR_ANALYSIS}/{plot_name}.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()

    print(f"Saved plot to: {full_path}")
    
    # # Print statistics in professional format
    # print("\n" + "=" * 60)
    # print(f"EMPIRICAL STATISTICS (N = {N:,}, σ = {sigma})")
    # print("=" * 60)
    # print(f"Mean:     {mean_val:10.6f}")
    # print(f"Std Dev:  {std_val:10.6f}")
    # print(f"Minimum:  {np.min(transformation):10.6f}")
    # print(f"Maximum:  {np.max(transformation):10.6f}")
    # print(f"Range:    {np.max(transformation) - np.min(transformation):10.6f}")
    # print("=" * 60)

def sin_transformation(arr):
    return 0.5 + 0.5 * np.sin(arr)

def sigmoid_transformation(arr):
    return 1/(1+np.exp(-arr))

def sin_histogram(N, sigma, shift, plot_name, act='sin', fig_size=DEFAULT_FIG_SIZE):
    """
    Sample N pairs of normal random variables and compute (e^X - e^Y)/(e^X + e^Y)
    
    Parameters:
    N: number of samples
    sigma: standard deviation of the normal distribution
    """
    
    # Sample N pairs of i.i.d. normal random variables with variance sigma^2
    samples_pos = np.random.normal(loc=shift, scale=sigma, size=(N))
    samples_neg = np.random.normal(loc=-shift, scale=sigma, size=(N))

    if act=='sin':
        transformation_pos = sin_transformation(samples_pos)
        transformation_neg = sin_transformation(samples_neg)
    elif act == 'sigmoid':
        transformation_pos = sigmoid_transformation(samples_pos)
        transformation_neg = sigmoid_transformation(samples_neg)

    transformation = np.concatenate([transformation_pos, transformation_neg])
    pos_mean = np.mean(transformation_pos)
    pos_std = np.std(transformation_pos)
    neg_mean = np.mean(transformation_neg)
    neg_std = np.std(transformation_neg)
    
    # Create professional histogram plot with seaborn styling
    fig, ax = plt.subplots(figsize=fig_size, dpi=300)
    
    # Use seaborn's color palette for professional appearance
    # colors = sns.color_palette("husl", n_colors=3)
    colors = sns.color_palette("colorblind", n_colors=3)    
    histogram_color = colors[0]
    mean_color = colors[1]
    std_color = colors[2]
    
    # Add standard deviation shaded area first
    stddev_label = f'Std Dev = {pos_std:.3f}'
    stddev_label = f'Std Dev'
    ax.axvspan(pos_mean - pos_std, pos_mean + pos_std,
            color=std_color, alpha=0.2,
            label=stddev_label)
    # ax.axvline(pos_mean - pos_std, color=std_color, linestyle=':', linewidth=1.5)
    # ax.axvline(pos_mean + pos_std, color=std_color, linestyle=':', linewidth=1.5)
    
    ax.axvspan(neg_mean - neg_std, neg_mean + neg_std,
            color=std_color, alpha=0.2, label=None)
    # ax.axvline(neg_mean - neg_std, color=std_color, linestyle=':', linewidth=1.5)
    # ax.axvline(neg_mean + neg_std, color=std_color, linestyle=':', linewidth=1.5)

    # Create histogram with professional styling
    n, bins, patches = ax.hist(transformation, bins=50, density=True,
                              alpha=0.8, color=histogram_color, edgecolor='white',
                              linewidth=0.8)
    
    # Add mean line with contrasting color
    
    mean_label = rf'Mean = ${pos_mean:.2f}$'
    if shift > 0:
        mean_label = rf'Mean = ${neg_mean:.2f}/{pos_mean:.2f}$'

    ax.axvline(pos_mean, color=mean_color, linestyle='--', linewidth=2.5,
               label=mean_label, alpha=0.9)
    ax.axvline(neg_mean, color=mean_color, linestyle='--', linewidth=2.5,
               label=None, alpha=0.9)
    
    # Professional formatting
    ax.set_ylabel(r'Density', fontsize=13, fontweight='bold')
    def percentage_formatter(x, pos):
        return rf'{x:.1f}\%'
    ax.yaxis.set_major_formatter(FuncFormatter(percentage_formatter))
    
    # Set axis limits for better presentation
    ax.set_xlim(-0.05, 1.05)
    
    # Customize legend with seaborn styling
    legend_loc = 'center right'
    if shift > 0:
        legend_loc='upper center'
    legend = ax.legend(loc=legend_loc, frameon=True, fancybox=True, 
                      shadow=True, fontsize=12)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)
    
    # Grid styling (seaborn already provides nice grids)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Spine styling
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color('black')
    
    # Tick formatting
    ax.tick_params(axis='both', which='major', labelsize=11, width=1.2)
    
    # Tight layout for publication
    plt.tight_layout()
    
    # Save the figure
    full_path = os.path.join(DIR_ANALYSIS, f"{plot_name}.png")
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    if SAVE_PGF:
        plt.savefig(f'{DIR_ANALYSIS}/{plot_name}.pgf', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    if SAVE_PDF:
        plt.savefig(f'{DIR_ANALYSIS}/{plot_name}.pdf', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    if SAVE_PNG:
        plt.savefig(f'{DIR_ANALYSIS}/{plot_name}.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    plt.close()

    print(f"Saved plot to: {full_path}")


if __name__ == "__main__":
    args = parse_args()
    fname = 'distribution_histograms' + "/" + args.object
    # Set random seed for reproducibility
    np.random.seed(0)
    if args.object == 'op-grad':        
        # Compute and plot for different sigma values
        sigma_values = [1.0, 2.0, 4.0, 8.0]
        y_values = [0.00,0.25,0.50,0.75,1.00]
        for y, sigma in itertools.product(y_values, sigma_values):
            plot_name = fname + f'/y={y}/sigma={sigma}'
            compute_transformation_histogram(args.N, sigma, y=y, plot_name=plot_name)            
    elif args.object == 'iwp-sin':
        sigma_values = [0.125,0.25,0.5,1.0]
        shifts = [0.0,1.2,1.5]
        for shift, sigma in itertools.product(shifts, sigma_values):
            plot_name = fname + f'/shift={shift}/sigma={sigma}'
            sin_histogram(args.N, sigma, shift, plot_name=plot_name, act='sin')
    elif args.object == 'iwp-sigmoid':
        sigma_values = [0.5,1.0]
        shifts = [3.0]
        for shift, sigma in itertools.product(shifts, sigma_values):
            plot_name = fname + f'/shift={shift}/sigma={sigma}'
            sin_histogram(args.N, sigma, shift, plot_name=plot_name, act='sigmoid')