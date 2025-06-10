#!/usr/bin/env python3
"""
Real-Life Pendulum Analysis Script
This script generates visualizations for the measurement of Earth's gravitational 
acceleration using a simple pendulum made from everyday objects (USB charger, necklace, keys).
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch
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
    relative_uncertainty = np.sqrt((delta_L / L)**2 + (2 * delta_T / T)**2)
    return g * relative_uncertainty

def generate_realistic_data():
    """Generate realistic sample data for real-life pendulum experiments.
    
    Returns:
        Dictionary containing experimental data for different pendulum types
    """
    
    # Define three different pendulum setups
    pendulums = {
        'USB Charger': {
            'length': 0.85,  # 85 cm USB cable
            'length_uncertainty': 0.001,  # 1 mm ruler uncertainty
            'description': 'USB charger cable with charger head as bob'
        },
        'Necklace': {
            'length': 0.45,  # 45 cm necklace
            'length_uncertainty': 0.001,  # 1 mm ruler uncertainty
            'description': 'Chain necklace with pendant'
        },
        'Keys on String': {
            'length': 0.65,  # 65 cm string with keys
            'length_uncertainty': 0.001,  # 1 mm ruler uncertainty
            'description': 'Keys attached to string/lanyard'
        }
    }
    
    data = {}
    
    for name, setup in pendulums.items():
        L = setup['length']
        delta_L = setup['length_uncertainty']
        
        # Calculate theoretical period
        theoretical_T = pendulum_period(L, TRUE_G)
        
        # Generate 10 realistic measurements with some variation
        # Simulate measurement errors and human timing variations
        measurement_error = np.random.normal(0, 0.05, 10)  # ±50ms timing variation
        times_10_oscillations = (theoretical_T * 10) + measurement_error
        
        # Calculate individual periods
        periods = times_10_oscillations / 10
        
        # Calculate statistics
        mean_period = np.mean(periods)
        std_period = np.std(periods, ddof=1)  # Sample standard deviation
        std_error = std_period / np.sqrt(10)  # Standard error of the mean
        
        # Calculate g and its uncertainty
        g_calculated = calculate_g(mean_period, L)
        g_uncertainty = calculate_g_uncertainty(g_calculated, delta_L, L, std_error, mean_period)
        
        data[name] = {
            'length': L,
            'length_uncertainty': delta_L,
            'times_10_oscillations': times_10_oscillations,
            'periods': periods,
            'mean_period': mean_period,
            'std_period': std_period,
            'std_error': std_error,
            'g_calculated': g_calculated,
            'g_uncertainty': g_uncertainty,
            'description': setup['description']
        }
    
    return data

def plot_real_pendulum_setup():
    """Create a visual illustration of real-life pendulum setups."""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 8))
    fig.suptitle('Real-Life Pendulum Setups Using Everyday Objects', fontsize=16, fontweight='bold')
    
    setups = [
        ('USB Charger Pendulum', 'USB cable + charger head'),
        ('Necklace Pendulum', 'Chain necklace + pendant'),
        ('Keys Pendulum', 'String/lanyard + keys')
    ]
    
    for i, (title, description) in enumerate(setups):
        ax = axes[i]
        ax.set_xlim(-2, 2)
        ax.set_ylim(-3, 1)
        ax.set_aspect('equal')
        
        # Draw support point
        support = Rectangle((-0.2, 0.8), 0.4, 0.2, fill=True, color='brown', alpha=0.8)
        ax.add_patch(support)
        ax.plot(0, 0.9, 'ko', markersize=8)  # Attachment point
        
        if i == 0:  # USB Charger
            # Draw USB cable (wavy line)
            y_cable = np.linspace(0.9, -1.5, 50)
            x_cable = 0.1 * np.sin(10 * y_cable) * np.exp((y_cable - 0.9) / 2)
            ax.plot(x_cable, y_cable, 'k-', linewidth=3, alpha=0.7)
            
            # Draw USB charger head
            charger = FancyBboxPatch((-0.3, -1.8), 0.6, 0.4, 
                                   boxstyle="round,pad=0.05", 
                                   facecolor='white', edgecolor='black', linewidth=2)
            ax.add_patch(charger)
            ax.text(0, -1.6, 'USB', ha='center', va='center', fontweight='bold', fontsize=8)
            
        elif i == 1:  # Necklace
            # Draw chain (series of small circles)
            chain_y = np.linspace(0.9, -1.2, 20)
            for y in chain_y:
                ax.plot(0, y, 'o', color='gold', markersize=3, alpha=0.8)
            
            # Draw pendant
            pendant = Circle((0, -1.5), 0.15, fill=True, color='gold', alpha=0.8)
            ax.add_patch(pendant)
            ax.plot(0, -1.5, 'o', color='darkred', markersize=8)  # Gem
            
        else:  # Keys
            # Draw string/lanyard
            ax.plot([0, 0], [0.9, -1.3], 'k-', linewidth=2)
            
            # Draw keys
            key_positions = [(-0.2, -1.5), (0, -1.6), (0.2, -1.4)]
            for x, y in key_positions:
                # Key body
                key_body = Rectangle((x-0.1, y-0.05), 0.2, 0.1, 
                                   fill=True, color='silver', alpha=0.8)
                ax.add_patch(key_body)
                # Key teeth
                ax.plot([x+0.1, x+0.15], [y, y], 'k-', linewidth=2)
                ax.plot([x+0.1, x+0.15], [y-0.03, y-0.03], 'k-', linewidth=2)
        
        # Add measurement arrow and label
        if i == 0:
            ax.annotate('', xy=(1.2, 0.9), xytext=(1.2, -1.6),
                       arrowprops=dict(arrowstyle='<->', color='red', lw=2))
            ax.text(1.4, -0.35, 'L = 0.85 m', rotation=90, va='center', 
                   fontsize=10, color='red', fontweight='bold')
        elif i == 1:
            ax.annotate('', xy=(1.2, 0.9), xytext=(1.2, -1.5),
                       arrowprops=dict(arrowstyle='<->', color='red', lw=2))
            ax.text(1.4, -0.3, 'L = 0.45 m', rotation=90, va='center', 
                   fontsize=10, color='red', fontweight='bold')
        else:
            ax.annotate('', xy=(1.2, 0.9), xytext=(1.2, -1.5),
                       arrowprops=dict(arrowstyle='<->', color='red', lw=2))
            ax.text(1.4, -0.3, 'L = 0.65 m', rotation=90, va='center', 
                   fontsize=10, color='red', fontweight='bold')
        
        ax.set_title(title, fontweight='bold', fontsize=12)
        ax.text(0, -2.5, description, ha='center', va='center', 
               fontsize=10, style='italic', wrap=True)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'pendulum_setup.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def create_measurement_protocol_figure():
    """Create a figure showing the measurement protocol."""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Real-Life Pendulum Measurement Protocol', fontsize=16, fontweight='bold')
    
    # Step 1: Length measurement
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 6)
    
    # Draw ruler
    ruler = Rectangle((1, 2), 8, 0.5, fill=True, color='yellow', alpha=0.7)
    ax1.add_patch(ruler)
    
    # Draw ruler markings
    for i in range(9):
        ax1.plot([1 + i, 1 + i], [2, 2.5], 'k-', linewidth=1)
        if i % 2 == 0:
            ax1.text(1 + i, 1.7, f'{i*10}', ha='center', fontsize=8)
    
    # Draw pendulum length
    ax1.plot([2, 2], [2.5, 5], 'r-', linewidth=3, alpha=0.8)
    ax1.plot(2, 5, 'ko', markersize=8)  # Support point
    ax1.plot(2, 2.5, 'bs', markersize=10)  # Bob
    
    ax1.text(5, 4, 'Measure length L\nwith ruler\n(±1 mm precision)', 
             ha='center', va='center', fontsize=10, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue'))
    
    ax1.set_title('Step 1: Length Measurement', fontweight='bold')
    ax1.axis('off')
    
    # Step 2: Timing setup
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 6)
    
    # Draw smartphone
    phone = Rectangle((4, 2), 2, 3, fill=True, color='black', alpha=0.8)
    ax2.add_patch(phone)
    screen = Rectangle((4.2, 2.5), 1.6, 2, fill=True, color='lightblue', alpha=0.9)
    ax2.add_patch(screen)
    
    # Stopwatch display
    ax2.text(5, 3.5, '00:20.14', ha='center', va='center', 
             fontsize=14, fontweight='bold', color='red')
    ax2.text(5, 3, 'STOPWATCH', ha='center', va='center', 
             fontsize=8, fontweight='bold')
    
    ax2.text(5, 1, 'Use smartphone\nstopwatch for timing\n10 oscillations', 
             ha='center', va='center', fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen'))
    
    ax2.set_title('Step 2: Timing Setup', fontweight='bold')
    ax2.axis('off')
    
    # Step 3: Oscillation measurement
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 6)
    
    # Draw pendulum positions
    positions = [(3, 4), (5, 3), (7, 4)]  # Left, center, right
    colors = ['red', 'blue', 'red']
    labels = ['Start', 'Bottom', 'End']
    
    for i, ((x, y), color, label) in enumerate(zip(positions, colors, labels)):
        ax3.plot([5, x], [5, y], '--', color=color, alpha=0.6, linewidth=2)
        ax3.plot(x, y, 'o', color=color, markersize=8)
        ax3.text(x, y-0.5, label, ha='center', fontsize=8, color=color, fontweight='bold')
    
    ax3.plot(5, 5, 'ko', markersize=8)  # Support point
    
    # Draw arc showing oscillation
    theta = np.linspace(-0.5, 0.5, 50)
    arc_x = 5 + 2 * np.sin(theta)
    arc_y = 5 - 2 * np.cos(theta)
    ax3.plot(arc_x, arc_y, 'g-', linewidth=2, alpha=0.7)
    
    ax3.text(5, 1.5, 'Measure time for\n10 complete oscillations\n(< 15° amplitude)', 
             ha='center', va='center', fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow'))
    
    ax3.set_title('Step 3: Oscillation Measurement', fontweight='bold')
    ax3.axis('off')
    
    # Step 4: Data collection
    ax4.set_xlim(0, 10)
    ax4.set_ylim(0, 6)
    
    # Draw data table
    table_data = [
        ['Trial', 'Time (s)'],
        ['1', '20.14'],
        ['2', '20.13'],
        ['3', '20.28'],
        ['...', '...'],
        ['10', '20.07']
    ]
    
    for i, row in enumerate(table_data):
        for j, cell in enumerate(row):
            rect = Rectangle((2 + j*2, 4.5 - i*0.5), 2, 0.5, 
                           fill=True, color='white', edgecolor='black')
            ax4.add_patch(rect)
            ax4.text(3 + j*2, 4.75 - i*0.5, cell, ha='center', va='center', 
                    fontsize=9, fontweight='bold' if i == 0 else 'normal')
    
    ax4.text(5, 1, 'Repeat measurement\n10 times for\nstatistical analysis', 
             ha='center', va='center', fontsize=10,
             bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral'))
    
    ax4.set_title('Step 4: Data Collection', fontweight='bold')
    ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'measurement_protocol.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def plot_measurement_data(data):
    """Plot the experimental measurements and results for all pendulum types."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Real-Life Pendulum Measurement Results', fontsize=16, fontweight='bold')
    
    pendulum_names = list(data.keys())
    colors = ['blue', 'green', 'red']
    
    for i, (name, color) in enumerate(zip(pendulum_names, colors)):
        pendulum_data = data[name]
        
        # Top row: Time measurements
        ax_top = axes[0, i]
        measurements = range(1, 11)
        times = pendulum_data['times_10_oscillations']
        mean_time = np.mean(times)
        
        ax_top.scatter(measurements, times, color=color, s=60, alpha=0.7, edgecolors='black')
        ax_top.axhline(y=mean_time, color=color, linestyle='--', linewidth=2, alpha=0.8)
        ax_top.fill_between([0.5, 10.5], mean_time - pendulum_data['std_period']*10, 
                           mean_time + pendulum_data['std_period']*10, 
                           alpha=0.2, color=color)
        
        ax_top.set_xlabel('Measurement Number')
        ax_top.set_ylabel('Time for 10 Oscillations (s)')
        ax_top.set_title(f'{name}\nTime Measurements')
        ax_top.grid(True, alpha=0.3)
        ax_top.set_xlim(0.5, 10.5)
        
        # Add statistics text
        stats_text = f'Mean: {mean_time:.3f} s\nStd: {pendulum_data["std_period"]*10:.3f} s'
        ax_top.text(0.02, 0.98, stats_text, transform=ax_top.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Bottom row: Calculated g values
        ax_bottom = axes[1, i]
        
        # Calculate g for each individual measurement
        individual_g = [calculate_g(t/10, pendulum_data['length']) for t in times]
        
        ax_bottom.scatter(measurements, individual_g, color=color, s=60, alpha=0.7, edgecolors='black')
        ax_bottom.axhline(y=pendulum_data['g_calculated'], color=color, linestyle='--', linewidth=2, alpha=0.8)
        ax_bottom.axhline(y=TRUE_G, color='black', linestyle='-', linewidth=2, alpha=0.8, label='True g = 9.81 m/s²')
        
        # Error bars for uncertainty
        ax_bottom.fill_between([0.5, 10.5], 
                              pendulum_data['g_calculated'] - pendulum_data['g_uncertainty'],
                              pendulum_data['g_calculated'] + pendulum_data['g_uncertainty'],
                              alpha=0.3, color=color)
        
        ax_bottom.set_xlabel('Measurement Number')
        ax_bottom.set_ylabel('Calculated g (m/s²)')
        ax_bottom.set_title(f'Calculated g Values\nL = {pendulum_data["length"]:.3f} m')
        ax_bottom.grid(True, alpha=0.3)
        ax_bottom.set_xlim(0.5, 10.5)
        ax_bottom.set_ylim(9.5, 10.2)
        
        if i == 0:
            ax_bottom.legend()
        
        # Add results text
        results_text = f'g = {pendulum_data["g_calculated"]:.3f} ± {pendulum_data["g_uncertainty"]:.3f} m/s²'
        ax_bottom.text(0.02, 0.02, results_text, transform=ax_bottom.transAxes, 
                      verticalalignment='bottom', 
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'pendulum_measurements.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def plot_uncertainty_analysis(data):
    """Plot the uncertainty analysis for the pendulum experiment."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    pendulum_names = list(data.keys())
    colors = ['blue', 'green', 'red']
    
    # Plot 1: Relative contributions to uncertainty for each pendulum
    x_pos = np.arange(len(pendulum_names))
    width = 0.35
    
    length_contributions = []
    period_contributions = []
    
    for name in pendulum_names:
        pendulum_data = data[name]
        rel_L_error = (pendulum_data['length_uncertainty'] / pendulum_data['length'])**2
        rel_T_error = (2 * pendulum_data['std_error'] / pendulum_data['mean_period'])**2
        total_error = rel_L_error + rel_T_error
        
        length_contributions.append(rel_L_error / total_error * 100)
        period_contributions.append(rel_T_error / total_error * 100)
    
    bars1 = ax1.bar(x_pos - width/2, length_contributions, width, label='Length Uncertainty', color='skyblue', alpha=0.7)
    bars2 = ax1.bar(x_pos + width/2, period_contributions, width, label='Period Uncertainty', color='lightgreen', alpha=0.7)
    
    # Add value labels on the bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=8)
    
    ax1.set_xlabel('Pendulum Type')
    ax1.set_ylabel('Contribution to Uncertainty (%)')
    ax1.set_title('Relative Contributions to g Uncertainty')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(pendulum_names, rotation=45)
    ax1.legend()
    ax1.set_ylim(0, 110)
    
    # Plot 2: Sensitivity analysis - how g uncertainty changes with measurement precision
    delta_factors = np.linspace(0.5, 3, 20)
    
    for i, (name, color) in enumerate(zip(pendulum_names, colors)):
        pendulum_data = data[name]
        
        # Calculate g uncertainties for different timing precisions
        delta_g_values = []
        for factor in delta_factors:
            new_delta_T = pendulum_data['std_error'] * factor
            delta_g = calculate_g_uncertainty(
                pendulum_data['g_calculated'], 
                pendulum_data['length_uncertainty'], 
                pendulum_data['length'], 
                new_delta_T, 
                pendulum_data['mean_period']
            )
            delta_g_values.append(delta_g)
        
        ax2.plot(delta_factors, delta_g_values, 'o-', label=name, color=color, linewidth=2, markersize=4)
    
    ax2.set_xlabel('Timing Precision Factor')
    ax2.set_ylabel('g Uncertainty (m/s²)')
    ax2.set_title('Sensitivity to Timing Precision')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'uncertainty_analysis.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def plot_length_vs_period():
    """Plot the relationship between pendulum length and period."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Generate theoretical curve
    lengths = np.linspace(0.2, 1.2, 100)
    theoretical_periods = [pendulum_period(L, TRUE_G) for L in lengths]
    
    # Plot theoretical relationship
    ax1.plot(lengths, theoretical_periods, 'k-', linewidth=2, label='Theoretical: $T = 2\pi\sqrt{L/g}$')
    
    # Add experimental data points from our three pendulums
    data = generate_realistic_data()
    pendulum_names = list(data.keys())
    colors = ['blue', 'green', 'red']
    
    for name, color in zip(pendulum_names, colors):
        pendulum_data = data[name]
        L = pendulum_data['length']
        T = pendulum_data['mean_period']
        T_err = pendulum_data['std_error']
        
        ax1.errorbar(L, T, yerr=T_err, fmt='o', color=color, markersize=8, 
                    capsize=5, label=f'{name} (L={L:.2f}m)', alpha=0.8)
    
    ax1.set_xlabel('Pendulum Length (m)')
    ax1.set_ylabel('Period (s)')
    ax1.set_title('Pendulum Length vs Period')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Linearized relationship (T² vs L)
    theoretical_T_squared = np.array(theoretical_periods)**2
    
    ax2.plot(lengths, theoretical_T_squared, 'k-', linewidth=2, 
            label='Theoretical: $T^2 = 4\pi^2 L/g$')
    
    for name, color in zip(pendulum_names, colors):
        pendulum_data = data[name]
        L = pendulum_data['length']
        T = pendulum_data['mean_period']
        T_err = pendulum_data['std_error']
        
        T_squared = T**2
        T_squared_err = 2 * T * T_err  # Error propagation for T²
        
        ax2.errorbar(L, T_squared, yerr=T_squared_err, fmt='s', color=color, 
                    markersize=8, capsize=5, label=f'{name}', alpha=0.8)
    
    # Add slope annotation
    slope = 4 * np.pi**2 / TRUE_G
    ax2.text(0.7, 3, f'Theoretical slope = $4\pi^2/g$ = {slope:.2f} s²/m', 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax2.set_xlabel('Pendulum Length (m)')
    ax2.set_ylabel('Period² (s²)')
    ax2.set_title('Linearized Relationship: Period² vs Length')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'length_vs_period.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def create_pendulum_motion_figure():
    """Create a figure showing pendulum motion and oscillation."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle('Pendulum Motion and Oscillation Analysis', fontsize=16, fontweight='bold')
    
    # Left plot: Pendulum motion diagram
    ax1.set_xlim(-3, 3)
    ax1.set_ylim(-4, 1)
    ax1.set_aspect('equal')
    
    # Draw support
    support = Rectangle((-0.3, 0.7), 0.6, 0.3, fill=True, color='brown', alpha=0.8)
    ax1.add_patch(support)
    ax1.plot(0, 0.85, 'ko', markersize=8)  # Attachment point
    
    # Draw pendulum at different positions
    L = 3
    angles = [-np.pi/8, 0, np.pi/8]  # Left, center, right positions
    colors = ['red', 'blue', 'red']
    labels = ['Maximum\nDisplacement', 'Equilibrium\nPosition', 'Maximum\nDisplacement']
    
    for i, (theta, color, label) in enumerate(zip(angles, colors, labels)):
        x = L * np.sin(theta)
        y = 0.85 - L * np.cos(theta)
        
        # Draw string
        ax1.plot([0, x], [0.85, y], '-', color=color, linewidth=2, alpha=0.7)
        
        # Draw bob
        ax1.plot(x, y, 'o', color=color, markersize=12, alpha=0.8)
        
        # Add labels
        if i != 1:  # Don't label the center position to avoid clutter
            ax1.text(x, y-0.5, label, ha='center', va='top', fontsize=9, 
                    color=color, fontweight='bold')
    
    # Draw arc showing oscillation path
    theta_arc = np.linspace(-np.pi/8, np.pi/8, 50)
    x_arc = L * np.sin(theta_arc)
    y_arc = 0.85 - L * np.cos(theta_arc)
    ax1.plot(x_arc, y_arc, 'g--', linewidth=2, alpha=0.8, label='Oscillation Path')
    
    # Add angle annotation
    ax1.annotate('', xy=(0.5, 0.85), xytext=(0, 0.85),
                arrowprops=dict(arrowstyle='->', color='black', lw=1))
    ax1.text(0.25, 1.1, r'$\theta < 15°$', fontsize=12, ha='center')
    
    # Add length annotation
    ax1.annotate('', xy=(0, 0.85), xytext=(0, -2.15),
                arrowprops=dict(arrowstyle='<->', color='purple', lw=2))
    ax1.text(-0.5, -0.65, 'L', fontsize=14, color='purple', fontweight='bold')
    
    ax1.set_title('Pendulum Motion Diagram')
    ax1.axis('off')
    
    # Right plot: Period vs amplitude (showing small angle approximation)
    ax2.set_xlim(0, 90)
    ax2.set_ylim(0.95, 1.15)
    
    # Generate data for different amplitudes
    amplitudes = np.linspace(1, 85, 50)  # angles in degrees
    L_example = 1.0  # 1 meter pendulum
    
    # Exact period calculation (elliptic integral approximation)
    exact_periods = []
    for amp in amplitudes:
        theta_rad = np.radians(amp)
        # First-order correction for large angles
        correction = 1 + (1/16) * theta_rad**2 + (11/3072) * theta_rad**4
        exact_period = pendulum_period(L_example, TRUE_G) * correction
        exact_periods.append(exact_period)
    
    # Small angle approximation (constant period)
    small_angle_period = pendulum_period(L_example, TRUE_G)
    small_angle_periods = [small_angle_period] * len(amplitudes)
    
    # Normalize to show relative difference
    exact_normalized = np.array(exact_periods) / small_angle_period
    small_angle_normalized = np.array(small_angle_periods) / small_angle_period
    
    ax2.plot(amplitudes, exact_normalized, 'r-', linewidth=2, label='Exact (with corrections)')
    ax2.plot(amplitudes, small_angle_normalized, 'b--', linewidth=2, label='Small angle approximation')
    
    # Highlight the valid range (< 15°)
    ax2.axvspan(0, 15, alpha=0.2, color='green', label='Valid range (< 15°)')
    
    ax2.set_xlabel('Amplitude (degrees)')
    ax2.set_ylabel('Normalized Period (T/T₀)')
    ax2.set_title('Period vs Amplitude\n(Showing Small Angle Approximation Validity)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'pendulum_motion.png'), 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def main():
    """Generate all figures for the pendulum experiment."""
    print("Generating figures for real-life pendulum experiment...")
    
    # Generate experimental data
    data = generate_realistic_data()
    
    # Create all figures
    plot_real_pendulum_setup()
    create_measurement_protocol_figure()
    create_pendulum_motion_figure()
    plot_measurement_data(data)
    plot_uncertainty_analysis(data)
    plot_length_vs_period()
    
    print("All figures generated successfully!")
    print("Figures saved in:", figures_dir)
    
    print("\nData Table for Markdown File:")
    print("| Parameter | Value | Uncertainty |")
    print("|-----------|-------|------------|")
    for name, pendulum_data in data.items():
        print(f"| {name} Length (L) | {pendulum_data['length']:.3f} m | {pendulum_data['length_uncertainty']:.5f} m |")
        print(f"| {name} Period (T) | {pendulum_data['mean_period']:.5f} s | {pendulum_data['std_error']:.5f} s |")
        print(f"| {name} Gravitational Acceleration (g) | {pendulum_data['g_calculated']:.5f} m/s² | {pendulum_data['g_uncertainty']:.5f} m/s² |")
    
    print("\nIndividual T10 Measurements:")
    for name, pendulum_data in data.items():
        print(f"\n{name} Measurements:")
        print("| Measurement | Time for 10 Oscillations (s) |")
        print("|-------------|----------------------------|")
        for i, T10 in enumerate(pendulum_data['times_10_oscillations']):
            print(f"| {i+1} | {T10:.3f} |")
    
    for name, pendulum_data in data.items():
        print(f"\n{name} Mean T10: {np.mean(pendulum_data['times_10_oscillations']):.5f} s")
        print(f"{name} Standard Deviation of T10: {np.std(pendulum_data['times_10_oscillations'], ddof=1):.5f} s")
        print(f"{name} Uncertainty in T10: {np.std(pendulum_data['times_10_oscillations'], ddof=1) / np.sqrt(10):.5f} s")

if __name__ == "__main__":
    main()
