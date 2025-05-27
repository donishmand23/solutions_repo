#!/usr/bin/env python3
"""
Central Limit Theorem Simulation Script
This script generates visualizations demonstrating the Central Limit Theorem
through simulations of different probability distributions.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy import stats
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec

# Set the style for the plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk")

# Create figures directory if it doesn't exist
figures_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'figures')
os.makedirs(figures_dir, exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
population_size = 10000
sample_sizes = [5, 10, 30, 50, 100]
num_samples = 1000

# Define distributions
def generate_uniform_population(size):
    """Generate data from a uniform distribution."""
    return np.random.uniform(0, 10, size)

def generate_exponential_population(size):
    """Generate data from an exponential distribution."""
    return np.random.exponential(scale=2.0, size=size)

def generate_binomial_population(size):
    """Generate data from a binomial distribution."""
    return np.random.binomial(n=20, p=0.3, size=size)

def generate_population_samples(population, sample_sizes, num_samples):
    """Generate samples of different sizes from the population."""
    results = {}
    for sample_size in sample_sizes:
        sample_means = []
        for _ in range(num_samples):
            sample = np.random.choice(population, size=sample_size, replace=True)
            sample_means.append(np.mean(sample))
        results[sample_size] = np.array(sample_means)
    return results

def plot_population_histogram(population, dist_name):
    """Plot histogram of the population distribution."""
    plt.figure(figsize=(10, 6))
    sns.histplot(population, kde=True, stat="density")
    
    # Add distribution parameters
    mean = np.mean(population)
    std = np.std(population)
    x = np.linspace(min(population), max(population), 1000)
    
    # Add title and axis labels
    plt.title(f"Population Distribution: {dist_name}")
    plt.xlabel("Value")
    plt.ylabel("Density")
    
    # Add mean and standard deviation information
    plt.axvline(mean, color='red', linestyle='dashed', linewidth=2)
    plt.text(mean + 0.1, plt.gca().get_ylim()[1] * 0.9, 
             f'Mean: {mean:.2f}\nStd Dev: {std:.2f}', 
             color='red')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, f'{dist_name.lower()}_population.png'), dpi=300)
    plt.close()

def plot_sampling_distributions(samples_dict, dist_name):
    """Plot histograms of sampling distributions for different sample sizes."""
    n_sizes = len(samples_dict)
    fig, axes = plt.subplots(1, n_sizes, figsize=(5*n_sizes, 6))
    
    # Calculate the population parameters
    all_samples = np.concatenate(list(samples_dict.values()))
    overall_min = np.min(all_samples)
    overall_max = np.max(all_samples)
    
    for i, (sample_size, sample_means) in enumerate(sorted(samples_dict.items())):
        ax = axes[i]
        
        # Histogram with KDE
        sns.histplot(sample_means, kde=True, stat="density", ax=ax)
        
        # Calculate mean and standard deviation
        mean = np.mean(sample_means)
        std = np.std(sample_means)
        
        # Plot the normal distribution curve for comparison
        x = np.linspace(overall_min, overall_max, 1000)
        normal_dist = stats.norm.pdf(x, mean, std)
        ax.plot(x, normal_dist, 'r-', linewidth=2)
        
        # Add title and labels
        ax.set_title(f"Sample Size = {sample_size}")
        ax.set_xlabel("Sample Mean")
        ax.set_ylabel("Density")
        
        # Set consistent x-axis limits
        ax.set_xlim(overall_min, overall_max)
        
        # Add normal QQ plot as an inset
        inset_ax = ax.inset_axes([0.6, 0.1, 0.35, 0.35])
        stats.probplot(sample_means, dist="norm", plot=inset_ax)
        inset_ax.set_title("Normal Q-Q Plot", fontsize=8)
        inset_ax.tick_params(labelsize=6)
    
    plt.suptitle(f"Sampling Distributions of the Mean: {dist_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, f'{dist_name.lower()}_sampling_distributions.png'), dpi=300)
    plt.close()

def create_static_comparison(samples_dict, dist_name, original_population):
    """Create a static comparison instead of animation to avoid potential compatibility issues."""
    sample_size = 30  # Use a moderate sample size
    sample_means = samples_dict[sample_size]
    
    fig = plt.figure(figsize=(15, 8))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])
    
    # Original distribution
    ax1 = plt.subplot(gs[0, :])
    sns.histplot(original_population, kde=True, stat="density", ax=ax1)
    ax1.set_title(f"Original Population Distribution: {dist_name}")
    ax1.set_xlabel("Value")
    ax1.set_ylabel("Density")
    
    population_mean = np.mean(original_population)
    population_std = np.std(original_population)
    ax1.axvline(population_mean, color='red', linestyle='dashed', linewidth=2)
    ax1.text(population_mean + 0.1, ax1.get_ylim()[1] * 0.9, 
             f'Population Mean: {population_mean:.2f}', color='red')
    
    # Sample demonstration
    ax2 = plt.subplot(gs[1, 0])
    random_indices = np.random.choice(len(original_population), size=sample_size, replace=True)
    random_sample = original_population[random_indices]
    sample_mean = np.mean(random_sample)
    
    ax2.scatter(np.arange(len(random_sample)), random_sample, color='blue', alpha=0.5)
    ax2.axhline(sample_mean, color='red', linestyle='dashed', linewidth=2)
    ax2.set_title(f"Random Sample (n={sample_size})")
    ax2.set_xlabel("Index")
    ax2.set_ylabel("Value")
    ax2.text(0, sample_mean + 0.1, f'Sample Mean: {sample_mean:.2f}', color='red')
    
    # Sampling distribution
    ax3 = plt.subplot(gs[1, 1])
    sns.histplot(sample_means, kde=True, stat="density", ax=ax3)
    
    # Add theoretical normal curve
    x_min = np.min(sample_means) - 0.5
    x_max = np.max(sample_means) + 0.5
    x = np.linspace(x_min, x_max, 1000)
    theoretical_mean = population_mean
    theoretical_std = population_std / np.sqrt(sample_size)
    theoretical_y = stats.norm.pdf(x, theoretical_mean, theoretical_std)
    ax3.plot(x, theoretical_y, 'r-', linewidth=2, 
             label=f'Normal: N({theoretical_mean:.2f}, {theoretical_std:.2f}²)')
    
    ax3.set_title("Sampling Distribution of Mean")
    ax3.set_xlabel("Sample Mean")
    ax3.set_ylabel("Density")
    ax3.legend()
    
    # Add stats
    mean_sample_means = np.mean(sample_means)
    std_sample_means = np.std(sample_means)
    stats_text = (f"Samples: {len(sample_means)}\n"
                 f"Mean: {mean_sample_means:.2f}\n"
                 f"Std Dev: {std_sample_means:.2f}\n"
                 f"Theoretical StdErr: {theoretical_std:.2f}")
    
    ax3.text(0.95, 0.95, stats_text, transform=ax3.transAxes,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, f'{dist_name.lower()}_sampling_process.png'), dpi=300)
    plt.close()

def create_animation(samples_dict, dist_name, original_population):
    """Create an animation showing the progression of the sampling distribution."""
    max_samples = 1000
    steps = 20  # Number of frames in the animation
    
    fig = plt.figure(figsize=(15, 8))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])
    
    ax1 = plt.subplot(gs[0, :])  # Original distribution
    ax2 = plt.subplot(gs[1, 0])  # Sampling process
    ax3 = plt.subplot(gs[1, 1])  # Evolving histogram
    
    # Plot the original population distribution
    sns.histplot(original_population, kde=True, stat="density", ax=ax1)
    ax1.set_title(f"Original Population Distribution: {dist_name}")
    ax1.set_xlabel("Value")
    ax1.set_ylabel("Density")
    
    population_mean = np.mean(original_population)
    population_std = np.std(original_population)
    ax1.axvline(population_mean, color='red', linestyle='dashed', linewidth=2)
    ax1.text(population_mean + 0.1, ax1.get_ylim()[1] * 0.9, 
             f'Population Mean: {population_mean:.2f}', color='red')
    
    # Initialize the animation function for different sample sizes
    sample_size = 30  # Use a moderate sample size for the animation
    sample_means = samples_dict[sample_size]
    
    # Determine x-axis limits for the histogram
    x_min = np.min(sample_means) - 0.5
    x_max = np.max(sample_means) + 0.5
    
    # Create empty plots that will be updated
    sample_scatter = ax2.scatter([], [], color='blue', alpha=0.5)
    sample_mean_line = ax2.axvline(0, color='red', linestyle='dashed', linewidth=2, label='Sample Mean')
    ax2.set_title(f"Random Sample (n={sample_size})")
    ax2.set_xlabel("Index")
    ax2.set_ylabel("Value")
    ax2.set_ylim(min(original_population), max(original_population))
    
    # Initialize histogram
    hist_bins = np.linspace(x_min, x_max, 30)
    n, bins, patches = ax3.hist([], bins=hist_bins, density=True, alpha=0.7)
    
    # Add theoretical normal curve
    x = np.linspace(x_min, x_max, 1000)
    theoretical_mean = population_mean
    theoretical_std = population_std / np.sqrt(sample_size)
    theoretical_line, = ax3.plot([], [], 'r-', linewidth=2, 
                                label=f'Normal: N({theoretical_mean:.2f}, {theoretical_std:.2f}²)')
    
    ax3.set_title("Sampling Distribution of Mean")
    ax3.set_xlabel("Sample Mean")
    ax3.set_ylabel("Density")
    ax3.set_xlim(x_min, x_max)
    ax3.legend()
    
    # Stats text for the evolving distribution
    stats_text = ax3.text(0.95, 0.95, "", transform=ax3.transAxes, 
                          verticalalignment='top', horizontalalignment='right',
                          bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))
    
    def init():
        sample_scatter.set_offsets(np.empty((0, 2)))
        sample_mean_line.set_xdata([0, 0])
        for patch in patches:
            patch.set_height(0)
        theoretical_line.set_data([], [])
        stats_text.set_text("")
        return list(patches) + [sample_scatter, sample_mean_line, theoretical_line, stats_text]
    
    def update(frame):
        # Generate a new random sample
        indices = np.random.choice(len(original_population), size=sample_size, replace=True)
        sample = original_population[indices]
        sample_mean = np.mean(sample)
        
        # Update sample plot
        xy = np.column_stack([np.arange(len(sample)), sample])
        sample_scatter.set_offsets(xy)
        sample_mean_line.set_xdata([0, len(sample)])
        sample_mean_line.set_ydata([sample_mean, sample_mean])
        
        # Calculate number of samples for this frame
        num_current_samples = int((frame + 1) * max_samples / steps)
        current_means = sample_means[:num_current_samples]
        
        # Update histogram
        n, bins = np.histogram(current_means, bins=hist_bins, density=True)
        for i, patch in enumerate(patches):
            patch.set_height(n[i])
        
        # Update theoretical normal curve
        theoretical_y = stats.norm.pdf(x, theoretical_mean, theoretical_std)
        theoretical_line.set_data(x, theoretical_y)
        
        # Update stats text
        current_mean = np.mean(current_means)
        current_std = np.std(current_means)
        stats_text.set_text(f"Samples: {num_current_samples}\n"
                           f"Mean: {current_mean:.2f}\n"
                           f"Std Dev: {current_std:.2f}")
        
        return list(patches) + [sample_scatter, sample_mean_line, theoretical_line, stats_text]
    
    ani = FuncAnimation(fig, update, frames=steps, init_func=init, blit=True, interval=500)
    plt.tight_layout()
    
    # Save the animation
    ani.save(os.path.join(figures_dir, f'{dist_name.lower()}_sampling_animation.gif'), 
             writer='pillow', fps=2, dpi=150)
    plt.close()

def plot_convergence_rates(distributions, dist_names):
    """Plot how quickly different distributions converge to normality based on sample size."""
    sample_sizes = [2, 5, 10, 20, 30, 50, 100, 200]
    num_samples = 1000
    
    # Measure of non-normality: Kolmogorov-Smirnov test statistic
    results = {name: [] for name in dist_names}
    
    for i, (dist, name) in enumerate(zip(distributions, dist_names)):
        for size in sample_sizes:
            ks_stats = []
            for _ in range(30):  # Run multiple trials for stability
                sample_means = []
                for _ in range(num_samples):
                    sample = np.random.choice(dist, size=size, replace=True)
                    sample_means.append(np.mean(sample))
                
                # Calculate KS test statistic (higher = further from normal)
                ks_stat, _ = stats.kstest(sample_means, 'norm', 
                                        args=(np.mean(sample_means), np.std(sample_means)))
                ks_stats.append(ks_stat)
            
            results[name].append(np.mean(ks_stats))
    
    # Plot the results
    plt.figure(figsize=(12, 7))
    
    for name, values in results.items():
        plt.plot(sample_sizes, values, marker='o', label=name)
    
    plt.axhline(y=0.05, color='r', linestyle='--', alpha=0.7, 
                label='Typical threshold for normality')
    
    plt.xscale('log')
    plt.grid(True, which="both", ls="--", alpha=0.7)
    plt.xlabel('Sample Size (log scale)')
    plt.ylabel('Distance from Normal Distribution (KS statistic)')
    plt.title('Convergence to Normality by Distribution Type')
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(os.path.join(figures_dir, 'convergence_comparison.png'), dpi=300)
    plt.close()

def plot_variance_impact(dist_name, population):
    """Plot the impact of population variance on the sampling distribution."""
    sample_size = 30
    num_samples = 1000
    
    # Create different versions of the population with different variances
    population_mean = np.mean(population)
    centered_pop = population - population_mean
    
    # Create populations with different variances
    variance_factors = [0.5, 1.0, 2.0]
    populations = []
    
    for factor in variance_factors:
        # Scale the centered population to achieve the desired variance
        scaled_pop = centered_pop * np.sqrt(factor) + population_mean
        populations.append(scaled_pop)
    
    # Generate samples and calculate sample means
    sample_means_list = []
    for pop in populations:
        sample_means = []
        for _ in range(num_samples):
            sample = np.random.choice(pop, size=sample_size, replace=True)
            sample_means.append(np.mean(sample))
        sample_means_list.append(np.array(sample_means))
    
    # Plot the results
    plt.figure(figsize=(12, 7))
    
    for i, (factor, means) in enumerate(zip(variance_factors, sample_means_list)):
        pop_std = np.std(populations[i])
        sample_std = np.std(means)
        
        sns.kdeplot(means, label=f'Variance Factor: {factor} (σ = {pop_std:.2f}, Sample σ = {sample_std:.2f})')
    
    plt.xlabel('Sample Mean')
    plt.ylabel('Density')
    plt.title(f'Effect of Population Variance on Sampling Distribution (n={sample_size})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(os.path.join(figures_dir, f'{dist_name.lower()}_variance_impact.png'), dpi=300)
    plt.close()

def main():
    # Generate populations
    uniform_pop = generate_uniform_population(population_size)
    exponential_pop = generate_exponential_population(population_size)
    binomial_pop = generate_binomial_population(population_size)
    
    distributions = [uniform_pop, exponential_pop, binomial_pop]
    dist_names = ["Uniform", "Exponential", "Binomial"]
    
    # Plot population histograms
    for dist, name in zip(distributions, dist_names):
        plot_population_histogram(dist, name)
    
    # Generate samples and plot sampling distributions
    for dist, name in zip(distributions, dist_names):
        samples = generate_population_samples(dist, sample_sizes, num_samples)
        plot_sampling_distributions(samples, name)
        # Create static visualization instead of animation to avoid compatibility issues
        create_static_comparison(samples, name, dist)
        try:
            # Attempt to create animation but handle gracefully if it fails
            create_animation(samples, name, dist)
        except Exception as e:
            print(f"Animation generation for {name} failed: {e}\nFalling back to static visualizations.")
        plot_variance_impact(name, dist)
    
    # Plot convergence rates
    plot_convergence_rates(distributions, dist_names)
    
    print("All figures have been generated and saved to the 'figures' directory.")

if __name__ == "__main__":
    main()
