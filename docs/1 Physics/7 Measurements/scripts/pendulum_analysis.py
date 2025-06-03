#!/usr/bin/env python3
"""
Pendulum Analysis Script
This script generates visualizations for the measurement of Earth's gravitational 
acceleration using a simple pendulum experiment.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Rectangle, Circle
from matplotlib.animation import FuncAnimation
import pandas as pd
import seaborn as sns
from scipy.optimize import curve_fit

# Set the style for the plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk")

# Create figures directory if it doesn't exist
figures_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'figures')
os.makedirs(figures_dir, exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)

# Constants
TRUE_G = 9.81  # m/s²

def pendulum_period(L, g):
    """Calculate the period of a simple pendulum.
    
    Args:
        L: Length of the pendulum in meters
        g: Gravitational acceleration in m/s²
        
    Returns:
        Period in seconds
    """
    return 2 * np.pi * np.sqrt(L / g)

def calculate_g(T, L):
    """Calculate gravitational acceleration from pendulum period.
    
    Args:
        T: Period of the pendulum in seconds
        L: Length of the pendulum in meters
        
    Returns:
        Gravitational acceleration in m/s²
    """
    return 4 * np.pi**2 * L / T**2

def calculate_g_uncertainty(g, delta_L, L, delta_T, T):
    """Calculate uncertainty in g using error propagation.
    
    Args:
        g: Calculated gravitational acceleration
        delta_L: Uncertainty in length measurement
        L: Length of the pendulum
        delta_T: Uncertainty in period measurement
        T: Period of the pendulum
        
    Returns:
        Uncertainty in g
    """
    # Using the error propagation formula: 
    # Δg = g * sqrt((ΔL/L)² + (2*ΔT/T)²)
    return g * np.sqrt((delta_L/L)**2 + (2*delta_T/T)**2)

def generate_sample_data():
    """Generate realistic sample data for a pendulum experiment.
    
    Returns:
        Dictionary containing experimental data
    """
    # Experimental parameters
    L = 1.0  # meter
    delta_L = 0.005  # 5 mm uncertainty in length
    
    # True period for this pendulum length (with small random offset to simulate real-world conditions)
    true_period = pendulum_period(L, TRUE_G) * (1 + np.random.normal(0, 0.005))
    
    # Simulate 10 measurements of 10 oscillations each with timing errors
    n_measurements = 10
    
    # Timing uncertainty (human reaction time ~ 0.1-0.2s)
    reaction_time_error = 0.15  # seconds
    
    # Simulate measurements with random errors
    T10_measurements = []
    for _ in range(n_measurements):
        # Add random error to represent timing uncertainty
        error = np.random.normal(0, reaction_time_error)
        # Time for 10 oscillations (with error)
        T10 = 10 * true_period + error
        T10_measurements.append(T10)
    
    # Calculate statistics
    T10_mean = np.mean(T10_measurements)
    T10_std = np.std(T10_measurements, ddof=1)  # Sample standard deviation
    T10_uncertainty = T10_std / np.sqrt(n_measurements)  # Standard error of the mean
    
    # Calculate the period and its uncertainty
    T = T10_mean / 10
    delta_T = T10_uncertainty / 10
    
    # Calculate g and its uncertainty
    g_measured = calculate_g(T, L)
    delta_g = calculate_g_uncertainty(g_measured, delta_L, L, delta_T, T)
    
    # Return all the data
    return {
        'L': L,
        'delta_L': delta_L,
        'T10_measurements': T10_measurements,
        'T10_mean': T10_mean,
        'T10_std': T10_std,
        'T10_uncertainty': T10_uncertainty,
        'T': T,
        'delta_T': delta_T,
        'g': g_measured,
        'delta_g': delta_g
    }

def plot_pendulum_setup():
    """Create a visual illustration of the pendulum setup."""
    fig, ax = plt.subplots(figsize=(10, 12))
    
    # Draw the support
    support_width = 3
    support_height = 0.5
    support = Rectangle((-support_width/2, 0), support_width, support_height, 
                       color='saddlebrown', alpha=0.7)
    ax.add_patch(support)
    
    # Pendulum parameters
    L = 5  # pendulum length in arbitrary units
    theta_max = np.pi/12  # maximum angle (15 degrees)
    
    # Draw the string
    theta = theta_max * np.sin(np.pi/4)  # angle at which to draw the pendulum
    x_end = L * np.sin(theta)
    y_end = -L * np.cos(theta)
    ax.plot([0, x_end], [support_height, y_end], 'k-', linewidth=2)
    
    # Draw the bob
    bob_radius = 0.4
    bob = Circle((x_end, y_end), bob_radius, color='blue', alpha=0.7)
    ax.add_patch(bob)
    
    # Add annotations
    ax.annotate('Support', xy=(-0.5, support_height + 0.2), fontsize=12)
    ax.annotate('String', xy=(x_end/2 + 0.5, -L/2), fontsize=12)
    ax.annotate('Bob', xy=(x_end + bob_radius + 0.2, y_end), fontsize=12)
    ax.annotate(f'L = {L} units', xy=(-3, -L/2), fontsize=12)
    ax.annotate(r'$\theta < 15°$', xy=(1.5, -1), fontsize=12)
    
    # Add a dashed line to show the rest position
    ax.plot([0, 0], [support_height, -(L+bob_radius)], 'k--', alpha=0.5)
    
    # Add arrows to show oscillation
    arrow_y = -L * 0.8
    ax.annotate('', xy=(theta_max*L*0.8, arrow_y), xytext=(-theta_max*L*0.8, arrow_y),
                arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    
    # Set axis properties
    ax.set_xlim(-3, 3)
    ax.set_ylim(-(L+2), 1)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Add title
    ax.set_title('Simple Pendulum Setup', fontsize=16)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'pendulum_setup.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_pendulum_animation():
    """Create an animation of a simple pendulum motion."""
    # Create a static pendulum visualization instead of an animation
    # to avoid potential compatibility issues
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Draw the support
    support_width = 3
    support_height = 0.5
    support = Rectangle((-support_width/2, 0), support_width, support_height, 
                       color='saddlebrown', alpha=0.7)
    ax.add_patch(support)
    
    # Pendulum parameters
    L = 5  # pendulum length in arbitrary units
    theta_max = np.pi/12  # maximum angle (15 degrees)
    
    # Positions for multiple pendulum instances
    num_positions = 5
    thetas = np.linspace(theta_max, -theta_max, num_positions)
    
    # Draw pendulum at different positions
    for i, theta in enumerate(thetas):
        x_end = L * np.sin(theta)
        y_end = -L * np.cos(theta)
        
        # Draw string with decreasing opacity
        alpha = 0.3 + 0.7 * (1 - i/(num_positions-1))
        ax.plot([0, x_end], [support_height, y_end], 'k-', linewidth=2, alpha=alpha)
        
        # Draw bob
        bob_radius = 0.4
        if i == 0 or i == num_positions-1:
            bob_color = 'red'
            bob_alpha = 0.7
        elif i == num_positions//2:
            bob_color = 'blue'
            bob_alpha = 0.9
        else:
            bob_color = 'gray'
            bob_alpha = 0.5
        
        bob = Circle((x_end, y_end), bob_radius, color=bob_color, alpha=bob_alpha)
        ax.add_patch(bob)
    
    # Add a dashed line to show the rest position
    ax.plot([0, 0], [support_height, -(L+bob_radius)], 'k--', alpha=0.5)
    
    # Add arrows to show oscillation
    arrow_y = -L * 0.8
    ax.annotate('', xy=(theta_max*L*0.8, arrow_y), xytext=(-theta_max*L*0.8, arrow_y),
                arrowprops=dict(arrowstyle='<->', color='red', lw=2))
    
    # Add an annotation for the period
    ax.text(0, -L-1, "One complete oscillation = Period T", 
           ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
    
    # Set axis properties
    ax.set_xlim(-3, 3)
    ax.set_ylim(-(L+2), 1)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Add title
    ax.set_title('Simple Pendulum Motion', fontsize=16)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'pendulum_motion.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_measurement_data(data):
    """Plot the experimental measurements and results."""
    # Create a DataFrame for the T10 measurements
    measurements_df = pd.DataFrame({
        'Measurement': range(1, len(data['T10_measurements'])+1),
        'T10 (s)': data['T10_measurements']
    })
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Plot the T10 measurements
    sns.barplot(x='Measurement', y='T10 (s)', data=measurements_df, ax=ax1, color='skyblue')
    
    # Add horizontal line for the mean
    ax1.axhline(y=data['T10_mean'], color='red', linestyle='--', 
               label=f'Mean: {data["T10_mean"]:.3f} s')
    
    # Add standard deviation range
    ax1.axhspan(data['T10_mean'] - data['T10_std'], 
              data['T10_mean'] + data['T10_std'], 
              alpha=0.2, color='red', 
              label=f'Std Dev: {data["T10_std"]:.3f} s')
    
    # Set plot properties
    ax1.set_xlabel('Measurement Number')
    ax1.set_ylabel('Time for 10 Oscillations (s)')
    ax1.set_title('T10 Measurements')
    ax1.legend()
    
    # Create data for histogram
    g_values = []
    for T10 in data['T10_measurements']:
        # Calculate g for each individual measurement
        T = T10 / 10
        g = calculate_g(T, data['L'])
        g_values.append(g)
    
    # Plot histogram of g values
    sns.histplot(g_values, kde=True, ax=ax2, color='skyblue')
    
    # Add vertical line for the mean g
    ax2.axvline(x=data['g'], color='red', linestyle='--', 
               label=f'Mean g: {data["g"]:.3f} m/s²')
    
    # Add vertical line for true g
    ax2.axvline(x=TRUE_G, color='green', linestyle='--', 
               label=f'True g: {TRUE_G:.3f} m/s²')
    
    # Add uncertainty range
    ax2.axvspan(data['g'] - data['delta_g'], 
              data['g'] + data['delta_g'], 
              alpha=0.2, color='red', 
              label=f'Uncertainty: {data["delta_g"]:.3f} m/s²')
    
    # Set plot properties
    ax2.set_xlabel('Gravitational Acceleration (m/s²)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of g Measurements')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'pendulum_measurements.png'), dpi=300)
    plt.close()

def plot_uncertainty_analysis(data):
    """Plot the uncertainty analysis for the pendulum experiment."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Create data for the relative contribution plot
    error_sources = ['Length', 'Period']
    rel_L_error = (data['delta_L'] / data['L'])**2
    rel_T_error = (2 * data['delta_T'] / data['T'])**2
    total_error = rel_L_error + rel_T_error
    
    rel_contributions = [rel_L_error / total_error * 100, rel_T_error / total_error * 100]
    
    # Plot the relative contributions
    bars = ax1.bar(error_sources, rel_contributions, color=['skyblue', 'lightgreen'])
    
    # Add value labels on the bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom')
    
    # Set plot properties
    ax1.set_ylim(0, 110)  # Set y-axis limit to accommodate the text
    ax1.set_ylabel('Contribution to Uncertainty (%)')
    ax1.set_title('Relative Contributions to g Uncertainty')
    
    # Plot how g uncertainty changes with different parameters
    
    # Generate ranges for L and T uncertainties
    delta_L_factors = np.linspace(0.5, 2, 10)
    delta_T_factors = np.linspace(0.5, 2, 10)
    
    # Calculate g uncertainties for different length uncertainties
    delta_g_L = []
    for factor in delta_L_factors:
        new_delta_L = data['delta_L'] * factor
        delta_g_L.append(calculate_g_uncertainty(
            data['g'], new_delta_L, data['L'], data['delta_T'], data['T']))
    
    # Calculate g uncertainties for different period uncertainties
    delta_g_T = []
    for factor in delta_T_factors:
        new_delta_T = data['delta_T'] * factor
        delta_g_T.append(calculate_g_uncertainty(
            data['g'], data['delta_L'], data['L'], new_delta_T, data['T']))
    
    # Plot how uncertainty changes
    ax2.plot(delta_L_factors, delta_g_L, 'o-', label='Length Uncertainty')
    ax2.plot(delta_T_factors, delta_g_T, 's-', label='Period Uncertainty')
    
    # Set plot properties
    ax2.set_xlabel('Relative Change in Uncertainty')
    ax2.set_ylabel('g Uncertainty (m/s²)')
    ax2.set_title('Sensitivity to Measurement Uncertainties')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'uncertainty_analysis.png'), dpi=300)
    plt.close()

def plot_length_vs_period():
    """Plot the relationship between pendulum length and period."""
    # Generate a range of lengths
    lengths = np.linspace(0.1, 2.0, 100)
    
    # Calculate periods for these lengths
    periods = [pendulum_period(L, TRUE_G) for L in lengths]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the relationship
    ax.plot(lengths, periods, 'b-', linewidth=2)
    
    # Add some sample points with error bars
    sample_lengths = np.array([0.25, 0.5, 1.0, 1.5])
    sample_periods = pendulum_period(sample_lengths, TRUE_G)
    
    # Add small random errors to make it look like experimental data
    length_errors = 0.01 * np.ones_like(sample_lengths)
    period_errors = 0.02 * np.ones_like(sample_periods)
    
    ax.errorbar(sample_lengths, sample_periods, 
               xerr=length_errors, yerr=period_errors,
               fmt='ro', capsize=5, markersize=8, 
               label='Sample Measurements')
    
    # Add a curve fit to the "experimental" data
    def fit_function(x, g):
        return 2 * np.pi * np.sqrt(x / g)
    
    popt, pcov = curve_fit(fit_function, sample_lengths, sample_periods)
    g_fit = popt[0]
    g_err = np.sqrt(pcov[0,0])
    
    # Plot the best fit curve
    ax.plot(lengths, fit_function(lengths, g_fit), 'g--', linewidth=2,
           label=f'Best Fit: g = {g_fit:.3f} ± {g_err:.3f} m/s²')
    
    # Add the theoretical curve
    ax.plot(lengths, pendulum_period(lengths, TRUE_G), 'r:', linewidth=2,
           label=f'Theoretical: g = {TRUE_G:.3f} m/s²')
    
    # Set plot properties
    ax.set_xlabel('Pendulum Length (m)')
    ax.set_ylabel('Period (s)')
    ax.set_title('Pendulum Period vs. Length')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add the formula
    ax.text(0.05, 0.9, r'$T = 2\pi\sqrt{\frac{L}{g}}$', 
           transform=ax.transAxes, fontsize=16, 
           bbox=dict(facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'length_vs_period.png'), dpi=300)
    plt.close()

def main():
    """Generate all figures for the pendulum experiment."""
    print("Generating figures for pendulum experiment...")
    
    # Generate experimental data
    data = generate_sample_data()
    
    # Create all figures
    plot_pendulum_setup()
    create_pendulum_animation()
    plot_measurement_data(data)
    plot_uncertainty_analysis(data)
    plot_length_vs_period()
    
    print("All figures have been generated and saved to the 'figures' directory.")
    
    # Print the data table for inclusion in the markdown file
    print("\nData Table for Markdown File:")
    print("| Parameter | Value | Uncertainty |")
    print("|-----------|-------|------------|")
    print(f"| Length (L) | {data['L']:.3f} m | {data['delta_L']:.5f} m |")
    print(f"| Period (T) | {data['T']:.5f} s | {data['delta_T']:.5f} s |")
    print(f"| Gravitational Acceleration (g) | {data['g']:.5f} m/s² | {data['delta_g']:.5f} m/s² |")
    
    print("\nIndividual T10 Measurements:")
    print("| Measurement | Time for 10 Oscillations (s) |")
    print("|-------------|----------------------------|")
    for i, T10 in enumerate(data['T10_measurements']):
        print(f"| {i+1} | {T10:.3f} |")
    
    print(f"\nMean T10: {data['T10_mean']:.5f} s")
    print(f"Standard Deviation of T10: {data['T10_std']:.5f} s")
    print(f"Uncertainty in T10: {data['T10_uncertainty']:.5f} s")

if __name__ == "__main__":
    main()
