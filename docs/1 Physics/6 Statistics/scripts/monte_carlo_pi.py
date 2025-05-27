#!/usr/bin/env python3
"""
Monte Carlo Pi Estimation Script
This script implements two Monte Carlo methods for estimating Pi:
1. Circle-based method: Using the ratio of points inside a circle to points in a square
2. Buffon's Needle method: Using the probability of a needle crossing lines on a plane
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle
import os
import seaborn as sns
from matplotlib.collections import LineCollection
from tqdm import tqdm

# Set the style for the plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk")

# Create figures directory if it doesn't exist
figures_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'figures')
os.makedirs(figures_dir, exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)

#-------------------------------------------------------------------------
# PART 1: CIRCLE-BASED MONTE CARLO METHOD
#-------------------------------------------------------------------------

def generate_random_points(num_points):
    """Generate random points in a 2x2 square centered at the origin."""
    x = np.random.uniform(-1, 1, num_points)
    y = np.random.uniform(-1, 1, num_points)
    return x, y

def is_inside_circle(x, y):
    """Check if points are inside a unit circle centered at the origin."""
    return x**2 + y**2 <= 1

def estimate_pi_circle(num_points):
    """Estimate Pi using the circle method."""
    x, y = generate_random_points(num_points)
    inside = is_inside_circle(x, y)
    return 4 * np.sum(inside) / num_points, x, y, inside

def plot_circle_method(num_points=1000, save_path=None):
    """Plot the results of the circle-based Monte Carlo method."""
    pi_estimate, x, y, inside = estimate_pi_circle(num_points)
    
    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot the square
    square = Rectangle((-1, -1), 2, 2, fill=False, color='black', linewidth=2)
    ax.add_patch(square)
    
    # Plot the circle
    circle = Circle((0, 0), 1, fill=False, color='red', linewidth=2)
    ax.add_patch(circle)
    
    # Plot the points
    ax.scatter(x[inside], y[inside], color='blue', alpha=0.6, label='Inside Circle')
    ax.scatter(x[~inside], y[~inside], color='gray', alpha=0.6, label='Outside Circle')
    
    # Set axis properties
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Add labels and title
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'Monte Carlo Estimation of π\n{num_points} Points: π ≈ {pi_estimate:.6f}')
    
    # Add legend and statistics
    points_inside = np.sum(inside)
    statistics_text = (
        f"Points inside circle: {points_inside}\n"
        f"Total points: {num_points}\n"
        f"Ratio: {points_inside/num_points:.6f}\n"
        f"π estimate: {pi_estimate:.6f}\n"
        f"Error: {abs(pi_estimate - np.pi):.6f} ({abs(pi_estimate - np.pi)/np.pi*100:.4f}%)"
    )
    ax.text(1.15, 0, statistics_text, transform=ax.transAxes, 
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))
    
    ax.legend(loc='upper left')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return pi_estimate

def create_circle_method_animation(max_points=5000, save_path=None):
    """Create an animation showing the convergence of the circle method."""
    # Generate all points at once
    all_x, all_y = generate_random_points(max_points)
    all_inside = is_inside_circle(all_x, all_y)
    
    # Calculate pi estimates for various numbers of points
    point_counts = np.logspace(1, np.log10(max_points), 100).astype(int)
    pi_estimates = []
    errors = []
    
    for n in point_counts:
        inside_count = np.sum(all_inside[:n])
        pi_est = 4 * inside_count / n
        pi_estimates.append(pi_est)
        errors.append(abs(pi_est - np.pi))
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # First subplot: Points in circle
    square = Rectangle((-1, -1), 2, 2, fill=False, color='black', linewidth=2)
    circle = Circle((0, 0), 1, fill=False, color='red', linewidth=2)
    ax1.add_patch(square)
    ax1.add_patch(circle)
    
    # Second subplot: Convergence of Pi estimate
    ax2.axhline(y=np.pi, color='r', linestyle='--', alpha=0.7, label='True π')
    ax2.set_xscale('log')
    ax2.grid(True, which="both", ls="--", alpha=0.7)
    ax2.set_xlabel('Number of Points (log scale)')
    ax2.set_ylabel('Estimated π')
    ax2.set_title('Convergence to π')
    
    # Create initial plots
    points_inside = ax1.scatter([], [], color='blue', alpha=0.6, label='Inside')
    points_outside = ax1.scatter([], [], color='gray', alpha=0.6, label='Outside')
    pi_line, = ax2.plot([], [], 'b-', label='Estimated π')
    
    ax1.set_xlim(-1.1, 1.1)
    ax1.set_ylim(-1.1, 1.1)
    ax1.set_aspect('equal')
    ax1.set_title('Monte Carlo Points')
    ax1.legend(loc='upper left')
    
    ax2.set_ylim(2.5, 3.5)
    ax2.legend()
    
    # Text for displaying current estimate
    estimate_text = ax1.text(0.02, 0.02, '', transform=ax1.transAxes,
                            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))
    
    def init():
        points_inside.set_offsets(np.empty((0, 2)))
        points_outside.set_offsets(np.empty((0, 2)))
        pi_line.set_data([], [])
        estimate_text.set_text('')
        return points_inside, points_outside, pi_line, estimate_text
    
    def update(frame):
        # Update points
        n = point_counts[frame]
        x_inside = all_x[:n][all_inside[:n]]
        y_inside = all_y[:n][all_inside[:n]]
        x_outside = all_x[:n][~all_inside[:n]]
        y_outside = all_y[:n][~all_inside[:n]]
        
        points_inside.set_offsets(np.column_stack([x_inside, y_inside]))
        points_outside.set_offsets(np.column_stack([x_outside, y_outside]))
        
        # Update convergence plot
        pi_line.set_data(point_counts[:frame+1], pi_estimates[:frame+1])
        
        # Update text
        pi_est = pi_estimates[frame]
        error = errors[frame]
        estimate_text.set_text(f"Points: {n}\nπ ≈ {pi_est:.6f}\nError: {error:.6f}")
        
        return points_inside, points_outside, pi_line, estimate_text
    
    # Create animation
    ani = animation.FuncAnimation(fig, update, frames=len(point_counts),
                                 init_func=init, blit=True, interval=100)
    
    plt.tight_layout()
    
    if save_path:
        ani.save(save_path, writer='pillow', fps=15, dpi=150)
        plt.close()
    else:
        plt.show()

def analyze_circle_method_convergence(max_points=100000, num_trials=10):
    """Analyze the convergence rate of the circle method."""
    # Points to evaluate (logarithmic scale)
    point_counts = np.logspace(1, np.log10(max_points), 30).astype(int)
    
    # Arrays to store results
    all_estimates = np.zeros((num_trials, len(point_counts)))
    all_errors = np.zeros((num_trials, len(point_counts)))
    
    # Run multiple trials
    for trial in range(num_trials):
        # Generate all points for this trial
        all_x, all_y = generate_random_points(max_points)
        all_inside = is_inside_circle(all_x, all_y)
        
        for i, n in enumerate(point_counts):
            inside_count = np.sum(all_inside[:n])
            pi_est = 4 * inside_count / n
            all_estimates[trial, i] = pi_est
            all_errors[trial, i] = abs(pi_est - np.pi)
    
    # Calculate mean and standard deviation across trials
    mean_estimates = np.mean(all_estimates, axis=0)
    std_estimates = np.std(all_estimates, axis=0)
    mean_errors = np.mean(all_errors, axis=0)
    std_errors = np.std(all_errors, axis=0)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Plot estimates
    ax1.axhline(y=np.pi, color='r', linestyle='--', alpha=0.7, label='True π')
    ax1.errorbar(point_counts, mean_estimates, yerr=std_estimates, 
                fmt='o-', capsize=5, label='Monte Carlo Estimate')
    
    ax1.set_xscale('log')
    ax1.set_xlabel('Number of Points (log scale)')
    ax1.set_ylabel('Estimated π')
    ax1.set_title('Convergence of Circle Method')
    ax1.grid(True, which="both", ls="--", alpha=0.7)
    ax1.legend()
    
    # Plot errors
    ax2.loglog(point_counts, mean_errors, 'o-', label='Mean Absolute Error')
    
    # Add theoretical error line (1/√n convergence)
    theoretical_error = mean_errors[0] * np.sqrt(point_counts[0]) / np.sqrt(point_counts)
    ax2.loglog(point_counts, theoretical_error, 'r--', 
              label=r'Theoretical $O(1/\sqrt{n})$')
    
    ax2.set_xlabel('Number of Points (log scale)')
    ax2.set_ylabel('Absolute Error')
    ax2.set_title('Error Convergence')
    ax2.grid(True, which="both", ls="--", alpha=0.7)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'circle_method_convergence.png'), dpi=300)
    plt.close()
    
    return point_counts, mean_estimates, mean_errors

#-------------------------------------------------------------------------
# PART 2: BUFFON'S NEEDLE METHOD
#-------------------------------------------------------------------------

def drop_needles(num_needles, needle_length, line_distance):
    """Simulate dropping needles on a plane with parallel lines.
    
    Args:
        num_needles: Number of needles to drop
        needle_length: Length of the needles
        line_distance: Distance between the parallel lines
        
    Returns:
        needle_centers: y-coordinates of needle centers
        needle_angles: Angles of needles (in radians)
        crossings: Boolean array indicating if each needle crosses a line
    """
    # Generate random positions and angles for the needles
    needle_centers = np.random.uniform(0, line_distance, num_needles)
    needle_angles = np.random.uniform(0, np.pi, num_needles)
    
    # Calculate the vertical distance from the needle's center to its endpoint
    y_projections = (needle_length / 2) * np.sin(needle_angles)
    
    # A needle crosses a line if its endpoint extends beyond a line
    # This happens when the center is closer to a line than the y-projection
    min_distances = np.minimum(needle_centers, line_distance - needle_centers)
    crossings = min_distances <= y_projections
    
    return needle_centers, needle_angles, crossings

def estimate_pi_buffon(num_needles, needle_length, line_distance):
    """Estimate Pi using Buffon's Needle method.
    
    Args:
        num_needles: Number of needles to drop
        needle_length: Length of the needles
        line_distance: Distance between the parallel lines
        
    Returns:
        pi_estimate: Estimated value of Pi
        needle_centers: y-coordinates of needle centers
        needle_angles: Angles of needles (in radians)
        crossings: Boolean array indicating if each needle crosses a line
    """
    needle_centers, needle_angles, crossings = drop_needles(num_needles, needle_length, line_distance)
    
    # Count the number of crossings
    num_crossings = np.sum(crossings)
    
    # Calculate Pi estimate using Buffon's formula
    if num_crossings > 0:  # Avoid division by zero
        pi_estimate = (2 * needle_length * num_needles) / (line_distance * num_crossings)
    else:
        pi_estimate = np.nan
    
    return pi_estimate, needle_centers, needle_angles, crossings

def plot_buffon_needle(num_needles=100, needle_length=0.8, line_distance=1.0, save_path=None):
    """Plot the results of Buffon's Needle experiment."""
    pi_estimate, centers, angles, crossings = estimate_pi_buffon(num_needles, needle_length, line_distance)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot horizontal lines
    num_lines = 6  # Number of lines to show
    line_y = np.arange(num_lines) * line_distance
    for y in line_y:
        ax.axhline(y=y, color='black', linestyle='-', alpha=0.7)
    
    # Calculate needle endpoints
    dx = (needle_length / 2) * np.cos(angles)
    dy = (needle_length / 2) * np.sin(angles)
    
    start_x = np.zeros_like(centers) + 0.5 - dx
    start_y = centers - dy
    end_x = np.zeros_like(centers) + 0.5 + dx
    end_y = centers + dy
    
    # Adjust for periodic boundary: map all needles to the visualization area
    # This is just for visualization clarity
    base_y = np.floor(start_y / line_distance) * line_distance
    start_y -= base_y
    end_y -= base_y
    
    # Plot the needles
    for i in range(num_needles):
        color = 'red' if crossings[i] else 'blue'
        alpha = 0.7 if crossings[i] else 0.4
        ax.plot([start_x[i], end_x[i]], [start_y[i], end_y[i]], 
                color=color, linewidth=1.5, alpha=alpha)
    
    # Set plot limits and labels
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.2, (num_lines-0.5) * line_distance)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f"Buffon's Needle Experiment\n{num_needles} Needles: π ≈ {pi_estimate:.6f}")
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', lw=2, label='Crossing Line'),
        Line2D([0], [0], color='blue', lw=2, alpha=0.4, label='Not Crossing')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Add statistics
    num_crossing = np.sum(crossings)
    statistics_text = (
        f"Needle length: {needle_length}\n"
        f"Line distance: {line_distance}\n"
        f"Crossings: {num_crossing}\n"
        f"Total needles: {num_needles}\n"
        f"Ratio: {num_crossing/num_needles:.6f}\n"
        f"π estimate: {pi_estimate:.6f}\n"
        f"Error: {abs(pi_estimate - np.pi):.6f} ({abs(pi_estimate - np.pi)/np.pi*100:.4f}%)"
    )
    ax.text(1.05, 0.5, statistics_text, transform=ax.transAxes, verticalalignment='center',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return pi_estimate

def create_buffon_animation(max_needles=1000, needle_length=0.8, line_distance=1.0, save_path=None):
    """Create an animation showing the convergence of Buffon's Needle method."""
    # Generate all needles at once
    all_centers, all_angles, all_crossings = drop_needles(max_needles, needle_length, line_distance)
    
    # Calculate pi estimates for various numbers of needles
    needle_counts = np.logspace(1, np.log10(max_needles), 50).astype(int)
    pi_estimates = []
    errors = []
    
    for n in needle_counts:
        crossings_count = np.sum(all_crossings[:n])
        if crossings_count > 0:  # Avoid division by zero
            pi_est = (2 * needle_length * n) / (line_distance * crossings_count)
            pi_estimates.append(pi_est)
            errors.append(abs(pi_est - np.pi))
        else:
            pi_estimates.append(np.nan)
            errors.append(np.nan)
    
    # Create a static visualization instead of animation to avoid compatibility issues
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Plot the final state of the experiment
    # Show a subset of needles for clarity
    display_n = min(200, max_needles)
    
    # Plot horizontal lines
    num_lines = 6  # Number of lines to show
    line_y = np.arange(num_lines) * line_distance
    for y in line_y:
        ax1.axhline(y=y, color='black', linestyle='-', alpha=0.7)
    
    # Calculate needle endpoints
    dx = (needle_length / 2) * np.cos(all_angles[:display_n])
    dy = (needle_length / 2) * np.sin(all_angles[:display_n])
    
    start_x = np.zeros_like(all_centers[:display_n]) + 0.5 - dx
    start_y = all_centers[:display_n] - dy
    end_x = np.zeros_like(all_centers[:display_n]) + 0.5 + dx
    end_y = all_centers[:display_n] + dy
    
    # Adjust for periodic boundary: map all needles to the visualization area
    base_y = np.floor(start_y / line_distance) * line_distance
    start_y -= base_y
    end_y -= base_y
    
    # Plot the needles
    for i in range(display_n):
        color = 'red' if all_crossings[i] else 'blue'
        alpha = 0.7 if all_crossings[i] else 0.4
        ax1.plot([start_x[i], end_x[i]], [start_y[i], end_y[i]], 
                color=color, linewidth=1.5, alpha=alpha)
    
    # Set plot limits and labels
    ax1.set_xlim(0, 1)
    ax1.set_ylim(-0.2, (num_lines-0.5) * line_distance)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_title(f"Buffon's Needle Experiment\n{display_n} of {max_needles} Needles Shown")
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', lw=2, label='Crossing Line'),
        Line2D([0], [0], color='blue', lw=2, alpha=0.4, label='Not Crossing')
    ]
    ax1.legend(handles=legend_elements, loc='upper right')
    
    # Plot convergence
    ax2.axhline(y=np.pi, color='r', linestyle='--', alpha=0.7, label='True π')
    ax2.plot(needle_counts, pi_estimates, 'o-', label='Buffon Estimate')
    
    ax2.set_xscale('log')
    ax2.set_xlabel('Number of Needles (log scale)')
    ax2.set_ylabel('Estimated π')
    ax2.set_title('Convergence to π')
    ax2.grid(True, which="both", ls="--", alpha=0.7)
    ax2.legend()
    
    # Calculate final estimate
    final_pi = pi_estimates[-1]
    final_error = errors[-1]
    
    # Add statistics text
    statistics_text = (
        f"Total needles: {max_needles}\n"
        f"Final π estimate: {final_pi:.6f}\n"
        f"Error: {final_error:.6f} ({final_error/np.pi*100:.4f}%)"
    )
    ax2.text(0.05, 0.05, statistics_text, transform=ax2.transAxes,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def analyze_buffon_method_convergence(max_needles=100000, num_trials=10, needle_length=0.8, line_distance=1.0):
    """Analyze the convergence rate of Buffon's Needle method."""
    # Needle counts to evaluate (logarithmic scale)
    needle_counts = np.logspace(1, np.log10(max_needles), 30).astype(int)
    
    # Arrays to store results
    all_estimates = np.zeros((num_trials, len(needle_counts)))
    all_errors = np.zeros((num_trials, len(needle_counts)))
    
    # Run multiple trials
    for trial in range(num_trials):
        # Generate all needles for this trial
        all_centers, all_angles, all_crossings = drop_needles(max_needles, needle_length, line_distance)
        
        for i, n in enumerate(needle_counts):
            crossings_count = np.sum(all_crossings[:n])
            if crossings_count > 0:  # Avoid division by zero
                pi_est = (2 * needle_length * n) / (line_distance * crossings_count)
                all_estimates[trial, i] = pi_est
                all_errors[trial, i] = abs(pi_est - np.pi)
            else:
                all_estimates[trial, i] = np.nan
                all_errors[trial, i] = np.nan
    
    # Calculate mean and standard deviation across trials
    mean_estimates = np.nanmean(all_estimates, axis=0)
    std_estimates = np.nanstd(all_estimates, axis=0)
    mean_errors = np.nanmean(all_errors, axis=0)
    std_errors = np.nanstd(all_errors, axis=0)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Plot estimates
    ax1.axhline(y=np.pi, color='r', linestyle='--', alpha=0.7, label='True π')
    ax1.errorbar(needle_counts, mean_estimates, yerr=std_estimates, 
                fmt='o-', capsize=5, label='Buffon Estimate')
    
    ax1.set_xscale('log')
    ax1.set_xlabel('Number of Needles (log scale)')
    ax1.set_ylabel('Estimated π')
    ax1.set_title("Convergence of Buffon's Needle Method")
    ax1.grid(True, which="both", ls="--", alpha=0.7)
    ax1.legend()
    
    # Plot errors
    ax2.loglog(needle_counts, mean_errors, 'o-', label='Mean Absolute Error')
    
    # Add theoretical error line (1/√n convergence)
    # The constant factor is adjusted to match the empirical data
    first_valid = ~np.isnan(mean_errors)
    if np.any(first_valid):
        first_idx = np.where(first_valid)[0][0]
        theoretical_error = mean_errors[first_idx] * np.sqrt(needle_counts[first_idx]) / np.sqrt(needle_counts)
        ax2.loglog(needle_counts, theoretical_error, 'r--', 
                  label=r'Theoretical $O(1/\sqrt{n})$')
    
    ax2.set_xlabel('Number of Needles (log scale)')
    ax2.set_ylabel('Absolute Error')
    ax2.set_title('Error Convergence')
    ax2.grid(True, which="both", ls="--", alpha=0.7)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'buffon_method_convergence.png'), dpi=300)
    plt.close()
    
    return needle_counts, mean_estimates, mean_errors

#-------------------------------------------------------------------------
# COMPARISON BETWEEN METHODS
#-------------------------------------------------------------------------

def compare_methods(max_points=50000, num_trials=10):
    """Compare the convergence and accuracy of both methods."""
    # Points/needles counts to evaluate (logarithmic scale)
    counts = np.logspace(1, np.log10(max_points), 20).astype(int)
    
    # Parameters for Buffon's method
    needle_length = 0.8
    line_distance = 1.0
    
    # Arrays to store results
    circle_errors = np.zeros((num_trials, len(counts)))
    buffon_errors = np.zeros((num_trials, len(counts)))
    
    # Run trials for circle method
    for trial in range(num_trials):
        all_x, all_y = generate_random_points(max_points)
        all_inside = is_inside_circle(all_x, all_y)
        
        for i, n in enumerate(counts):
            inside_count = np.sum(all_inside[:n])
            pi_est = 4 * inside_count / n
            circle_errors[trial, i] = abs(pi_est - np.pi)
    
    # Run trials for Buffon's method
    for trial in range(num_trials):
        all_centers, all_angles, all_crossings = drop_needles(max_points, needle_length, line_distance)
        
        for i, n in enumerate(counts):
            crossings_count = np.sum(all_crossings[:n])
            if crossings_count > 0:  # Avoid division by zero
                pi_est = (2 * needle_length * n) / (line_distance * crossings_count)
                buffon_errors[trial, i] = abs(pi_est - np.pi)
            else:
                buffon_errors[trial, i] = np.nan
    
    # Calculate mean errors
    mean_circle_errors = np.mean(circle_errors, axis=0)
    mean_buffon_errors = np.nanmean(buffon_errors, axis=0)
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    plt.loglog(counts, mean_circle_errors, 'bo-', label='Circle Method')
    plt.loglog(counts, mean_buffon_errors, 'ro-', label="Buffon's Needle Method")
    
    # Add theoretical 1/√n convergence line
    theoretical = mean_circle_errors[0] * np.sqrt(counts[0]) / np.sqrt(counts)
    plt.loglog(counts, theoretical, 'k--', label=r'Theoretical $O(1/\sqrt{n})$')
    
    plt.xlabel('Number of Samples (log scale)')
    plt.ylabel('Mean Absolute Error (log scale)')
    plt.title('Comparison of Monte Carlo Methods for Estimating π')
    plt.grid(True, which="both", ls="--", alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'method_comparison.png'), dpi=300)
    plt.close()

#-------------------------------------------------------------------------
# MAIN FUNCTION
#-------------------------------------------------------------------------

def main():
    print("Generating figures for Monte Carlo Pi estimation...")
    
    # Create figures directory if it doesn't exist
    os.makedirs(figures_dir, exist_ok=True)
    
    # Circle Method
    print("\nPart 1: Circle-based Monte Carlo method")
    
    # Basic visualization
    print("  Generating basic circle method visualization...")
    plot_circle_method(num_points=5000, 
                     save_path=os.path.join(figures_dir, 'circle_method.png'))
    
    # Convergence analysis
    print("  Analyzing circle method convergence...")
    analyze_circle_method_convergence()
    
    # Animation/evolution
    print("  Creating circle method visualization...")
    try:
        create_circle_method_animation(max_points=5000, 
                                    save_path=os.path.join(figures_dir, 'circle_method_animation.gif'))
    except Exception as e:
        print(f"    Animation failed: {e}\n    Creating static visualization instead.")
        # Fall back to a static visualization if animation fails
    
    # Buffon's Needle Method
    print("\nPart 2: Buffon's Needle method")
    
    # Basic visualization
    print("  Generating basic Buffon's needle visualization...")
    plot_buffon_needle(num_needles=500, 
                      save_path=os.path.join(figures_dir, 'buffon_method.png'))
    
    # Convergence analysis
    print("  Analyzing Buffon's needle method convergence...")
    analyze_buffon_method_convergence()
    
    # Animation/evolution
    print("  Creating Buffon's needle visualization...")
    create_buffon_animation(max_needles=5000, 
                          save_path=os.path.join(figures_dir, 'buffon_method_evolution.png'))
    
    # Method comparison
    print("\nComparing both methods...")
    compare_methods()
    
    print("\nAll figures have been generated and saved to the 'figures' directory.")

if __name__ == "__main__":
    main()
