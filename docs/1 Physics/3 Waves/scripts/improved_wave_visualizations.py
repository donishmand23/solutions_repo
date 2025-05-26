#!/usr/bin/env python3
"""
Improved Wave Interference Visualizations

This script generates enhanced wave interference patterns with proper scaling,
consistent units, and animated visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import os

# Create figures directory if it doesn't exist
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
figures_dir = os.path.join(parent_dir, 'figures')
if not os.path.exists(figures_dir):
    os.makedirs(figures_dir)

# Wave parameters - with consistent units
A = 1.0         # Amplitude (arbitrary units)
lambda_val = 5.0  # Wavelength (m)
k = 2 * np.pi / lambda_val  # Wave number (rad/m)
f = 1.0         # Frequency (Hz)
omega = 2 * np.pi * f  # Angular frequency (rad/s)

def calculate_displacement(x, y, sources, t=0, phases=None):
    """
    Calculate the total displacement at point (x,y) due to all sources at time t.
    
    Parameters:
    - x, y: Grid coordinates (m)
    - sources: List of (x,y) coordinates of wave sources (m)
    - t: Time (s)
    - phases: List of initial phases for each source (rad)
    
    Returns:
    - total: Total displacement at each point (normalized to [-4,4] range)
    """
    total = np.zeros_like(x)
    
    if phases is None:
        phases = [0] * len(sources)
    
    for (x0, y0), phi in zip(sources, phases):
        r = np.sqrt((x - x0)**2 + (y - y0)**2)
        # Avoid division by zero at source positions
        r = np.maximum(r, 1e-10)
        # Wave equation with amplitude decay proportional to 1/√r
        displacement = A * np.cos(k*r - omega*t + phi) / np.sqrt(r)
        total += displacement
    
    # Normalize displacement values to exact -4 to 4 range
    # Instead of using theoretical maximum, find actual min/max
    max_val = np.max(np.abs(total))
    if max_val > 0:  # Avoid division by zero
        total = 4.0 * (total / max_val)
        
    # Double-check to ensure we're exactly in the -4 to 4 range
    total = np.clip(total, -4.0, 4.0)
    
    return total

def generate_polygon_vertices(n, radius):
    """Generate vertices of a regular polygon with n sides and given radius (m)."""
    vertices = []
    for i in range(n):
        angle = 2 * np.pi * i / n
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        vertices.append((x, y))
    return vertices

def create_phase_difference_comparison(size=20, resolution=200):
    """Create a comparison of interference patterns with different phase relationships."""
    # Create a figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    # Source positions (two sources) - separated by 10m
    source1 = (5, 0)
    source2 = (-5, 0)
    sources = [source1, source2]
    
    # Different phase differences
    phase_diffs = [0, np.pi/2, np.pi, 3*np.pi/2]
    phase_labels = ["0° (In phase)", "90°", "180° (Out of phase)", "270°"]
    
    # Create a grid of points
    x = np.linspace(-size, size, resolution)
    y = np.linspace(-size, size, resolution)
    X, Y = np.meshgrid(x, y)
    
    # Plot for each phase difference
    for i, (phase_diff, label) in enumerate(zip(phase_diffs, phase_labels)):
        phases = [0, phase_diff]
        Z = calculate_displacement(X, Y, sources, phases=phases)
        
        # Create contour plot with fixed scale limits
        contour = axes[i].contourf(X, Y, Z, 50, cmap='coolwarm', vmin=-4, vmax=4)
        
        # Plot source positions
        for x0, y0 in sources:
            axes[i].plot(x0, y0, 'ko', markersize=8)
        
        # Set plot properties
        axes[i].set_title(f'Phase Difference: {label}', fontsize=14)
        axes[i].set_xlabel('X Position (m)', fontsize=12)
        axes[i].set_ylabel('Y Position (m)', fontsize=12)
        axes[i].set_aspect('equal')
        axes[i].grid(True, linestyle='--', alpha=0.7)
        
        # Add distance scale for reference
        axes[i].text(15, -15, f'λ = {lambda_val} m', fontsize=12, 
                    bbox=dict(facecolor='white', alpha=0.7))
    
    # Add a colorbar to the figure
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(contour, cax=cbar_ax)
    cbar.set_label('Displacement (arb. units)', fontsize=12)
    
    plt.suptitle('Effect of Phase Difference on Two-Source Interference Patterns', fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    
    # Save the figure
    plt.savefig(os.path.join(figures_dir, 'phase_difference_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_phase_difference_animation(size=20, resolution=150, frames=48):
    """Create an animation showing how the interference pattern changes with phase difference."""
    # Source positions (two sources)
    source1 = (5, 0)
    source2 = (-5, 0)
    sources = [source1, source2]
    
    # Create a grid of points
    x = np.linspace(-size, size, resolution)
    y = np.linspace(-size, size, resolution)
    X, Y = np.meshgrid(x, y)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Initial plot
    phases = [0, 0]
    Z = calculate_displacement(X, Y, sources, phases=phases)
    contour = ax.contourf(X, Y, Z, 50, cmap='coolwarm', vmin=-4, vmax=4)
    
    # Plot source positions
    for x0, y0 in sources:
        ax.plot(x0, y0, 'ko', markersize=8)
    
    # Set plot properties
    ax.set_title('Phase Difference: 0°', fontsize=14)
    ax.set_xlabel('X Position (m)', fontsize=12)
    ax.set_ylabel('Y Position (m)', fontsize=12)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add wavelength scale indicator
    scale_text = ax.text(15, -15, f'λ = {lambda_val} m', fontsize=12,
                       bbox=dict(facecolor='white', alpha=0.7))
    
    # Add colorbar
    cbar = fig.colorbar(contour)
    cbar.set_label('Displacement (arb. units)', fontsize=12)
    
    # Add phase difference indicator
    phase_text = ax.text(-15, 15, 'Phase Diff: 0°', fontsize=14,
                       bbox=dict(facecolor='white', alpha=0.7))
    
    # Update function for animation
    def update(frame):
        # Calculate phase difference for this frame (0 to 2π)
        phase_diff = 2 * np.pi * frame / frames
        phases = [0, phase_diff]
        
        # Update title
        phase_degrees = int(phase_diff * 180 / np.pi) % 360
        phase_text.set_text(f'Phase Diff: {phase_degrees}°')
        
        # Calculate new displacement
        Z = calculate_displacement(X, Y, sources, phases=phases)
        
        # Clear previous contour and create new one
        for coll in ax.collections:
            coll.remove()
        contour = ax.contourf(X, Y, Z, 50, cmap='coolwarm', vmin=-4, vmax=4)
        
        # Return updated artists
        return [contour, phase_text]
    
    # Create animation
    anim = FuncAnimation(fig, update, frames=frames, interval=50, blit=False)
    
    # Save as GIF
    writer = PillowWriter(fps=10)
    anim.save(os.path.join(figures_dir, 'phase_difference_animation.gif'), writer=writer)
    print("Phase difference animation saved!")
    plt.close()

def create_wave_propagation_animation(n_sides=4, size=20, resolution=150, frames=40):
    """Create an animation showing wave propagation from multiple sources."""
    # Generate source positions
    radius = 10  # Radius in meters
    sources = generate_polygon_vertices(n_sides, radius)
    
    # Create a grid of points
    x = np.linspace(-size, size, resolution)
    y = np.linspace(-size, size, resolution)
    X, Y = np.meshgrid(x, y)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Initial plot
    Z = calculate_displacement(X, Y, sources, t=0)
    contour = ax.contourf(X, Y, Z, 50, cmap='viridis', vmin=-4, vmax=4)
    
    # Plot source positions
    for x0, y0 in sources:
        ax.plot(x0, y0, 'wo', markersize=8, markeredgecolor='black')
    
    # Set plot properties
    ax.set_title(f'Wave Propagation from {n_sides} Sources', fontsize=14)
    ax.set_xlabel('X Position (m)', fontsize=12)
    ax.set_ylabel('Y Position (m)', fontsize=12)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add wavelength scale indicator
    scale_text = ax.text(0.05, 0.05, f'λ = {lambda_val} m', fontsize=12,
                       transform=ax.transAxes,
                       bbox=dict(facecolor='white', alpha=0.7))
    
    # Add colorbar
    cbar = fig.colorbar(contour)
    cbar.set_label('Displacement (arb. units)', fontsize=12)
    
    # Add time indicator
    time_text = ax.text(0.05, 0.95, 'Time: 0.00 s', fontsize=14, transform=ax.transAxes,
                       bbox=dict(facecolor='white', alpha=0.7))
    
    # Update function for animation
    def update(frame):
        # Calculate time for this frame (0 to 2 seconds)
        t = 2 * frame / frames
        
        # Update time text
        time_text.set_text(f'Time: {t:.2f} s')
        
        # Calculate new displacement
        Z = calculate_displacement(X, Y, sources, t=t)
        
        # Clear previous contour and create new one
        for coll in ax.collections:
            coll.remove()
        contour = ax.contourf(X, Y, Z, 50, cmap='viridis', vmin=-4, vmax=4)
        
        # Plot source positions again
        source_points = []
        for x0, y0 in sources:
            point, = ax.plot(x0, y0, 'wo', markersize=8, markeredgecolor='black')
            source_points.append(point)
        
        # Return the updated artists
        return [contour, time_text] + source_points
    
    # Create animation
    anim = FuncAnimation(fig, update, frames=frames, interval=50, blit=False)
    
    # Save as GIF
    writer = PillowWriter(fps=10)
    anim.save(os.path.join(figures_dir, f'{n_sides}_sources_wave_propagation.gif'), writer=writer)
    print(f"Wave propagation animation with {n_sides} sources saved!")
    plt.close()

def plot_interference_pattern(n_sides, size=20, resolution=300):
    """Plot the interference pattern for a regular polygon with n_sides."""
    # Generate source positions (vertices of the polygon)
    radius = 10  # Radius in meters
    sources = generate_polygon_vertices(n_sides, radius)
    
    # Create a grid of points
    x = np.linspace(-size, size, resolution)
    y = np.linspace(-size, size, resolution)
    X, Y = np.meshgrid(x, y)
    
    # Calculate displacement at each point
    Z = calculate_displacement(X, Y, sources)
    
    # Plot the interference pattern
    plt.figure(figsize=(10, 8))
    contour = plt.contourf(X, Y, Z, 50, cmap='viridis', vmin=-4, vmax=4)
    cbar = plt.colorbar(contour)
    cbar.set_label('Displacement (arb. units, -4 to +4)', fontsize=12)
    
    # Plot source positions
    for x0, y0 in sources:
        plt.plot(x0, y0, 'wo', markersize=8, markeredgecolor='black')
    
    # Set plot properties
    plt.title(f'Interference Pattern for {n_sides} Sources in a Regular Polygon', fontsize=14)
    plt.xlabel('X Position (m)', fontsize=12)
    plt.ylabel('Y Position (m)', fontsize=12)
    plt.axis('equal')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add wavelength scale indicator
    plt.text(0.05, 0.05, f'λ = {lambda_val} m', fontsize=12, transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.7))
    
    # Save the figure
    plt.savefig(os.path.join(figures_dir, f'{n_sides}_sided_polygon_interference.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_3d_surface_plot(n_sides, size=20, resolution=150):
    """Create a 3D surface plot of the interference pattern."""
    # Generate source positions
    radius = 10  # meters
    sources = generate_polygon_vertices(n_sides, radius)
    
    # Create a grid of points
    x = np.linspace(-size, size, resolution)
    y = np.linspace(-size, size, resolution)
    X, Y = np.meshgrid(x, y)
    
    # Calculate displacement at each point
    Z = calculate_displacement(X, Y, sources)
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the surface with consistent z limits
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', linewidth=0, antialiased=True)
    
    # Set z-axis limits for consistent scaling
    ax.set_zlim(-4, 4)
    
    # Plot source positions
    for x0, y0 in sources:
        ax.scatter(x0, y0, 0, color='white', s=50, marker='o', edgecolor='black')
    
    # Set plot properties
    ax.set_title(f'3D Visualization of Interference Pattern ({n_sides} Sources)', fontsize=16)
    ax.set_xlabel('X Position (m)', fontsize=12)
    ax.set_ylabel('Y Position (m)', fontsize=12)
    ax.set_zlabel('Displacement (arb. units)', fontsize=12)
    
    # Add a color bar
    cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    cbar.set_label('Displacement (arb. units)', fontsize=12)
    
    # Add wavelength scale indicator as text annotation
    ax.text2D(0.05, 0.05, f'λ = {lambda_val} m', fontsize=12, transform=ax.transAxes,
             bbox=dict(facecolor='white', alpha=0.7))
    
    # Save the figure
    plt.savefig(os.path.join(figures_dir, f'{n_sides}_sided_3d_interference.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_cross_section_animation(n_sides=4, size=20, resolution=150, frames=40):
    """Create an animation showing a cross-section of the wave pattern over time."""
    # Generate source positions
    radius = 10  # meters
    sources = generate_polygon_vertices(n_sides, radius)
    
    # Create figure with 2 subplots (2D contour and 1D cross-section)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), gridspec_kw={'height_ratios': [2, 1]})
    
    # Create a grid of points for 2D plot
    x = np.linspace(-size, size, resolution)
    y = np.linspace(-size, size, resolution)
    X, Y = np.meshgrid(x, y)
    
    # Initial plots
    Z = calculate_displacement(X, Y, sources, t=0)
    contour = ax1.contourf(X, Y, Z, 50, cmap='viridis', vmin=-4, vmax=4)
    
    # Plot source positions
    for x0, y0 in sources:
        ax1.plot(x0, y0, 'wo', markersize=8, markeredgecolor='black')
    
    # Draw the line for cross-section
    ax1.axhline(y=0, color='r', linestyle='-', linewidth=2, alpha=0.7)
    
    # Set properties for 2D plot
    ax1.set_title(f'Wave Interference Pattern ({n_sides} Sources)', fontsize=14)
    ax1.set_xlabel('X Position (m)', fontsize=12)
    ax1.set_ylabel('Y Position (m)', fontsize=12)
    ax1.set_aspect('equal')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Create 1D cross-section plot (y = 0)
    cross_section = Z[resolution//2, :]  # Middle row
    line, = ax2.plot(x, cross_section, 'b-', linewidth=2)
    
    # Set properties for 1D plot
    ax2.set_title('Cross-section along y = 0', fontsize=14)
    ax2.set_xlabel('X Position (m)', fontsize=12)
    ax2.set_ylabel('Displacement', fontsize=12)
    ax2.set_ylim(-4, 4)  # Match the normalized scale range
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Add colorbar
    cbar = fig.colorbar(contour, ax=ax1)
    cbar.set_label('Displacement (arb. units)', fontsize=12)
    
    # Add time indicator
    time_text = ax1.text(0.05, 0.95, 'Time: 0.00 s', fontsize=14, transform=ax1.transAxes,
                       bbox=dict(facecolor='white', alpha=0.7))
    
    # Update function for animation
    def update(frame):
        # Calculate time for this frame (0 to 2 seconds)
        t = 2 * frame / frames
        
        # Update time text
        time_text.set_text(f'Time: {t:.2f} s')
        
        # Calculate new displacement
        Z = calculate_displacement(X, Y, sources, t=t)
        
        # Update 2D contour plot
        for coll in ax1.collections:
            coll.remove()
        contour = ax1.contourf(X, Y, Z, 50, cmap='viridis', vmin=-4, vmax=4)
        
        # Plot source positions again
        source_points = []
        for x0, y0 in sources:
            point, = ax1.plot(x0, y0, 'wo', markersize=8, markeredgecolor='black')
            source_points.append(point)
        
        # Update cross-section line
        cross_section = Z[resolution//2, :]  # Middle row
        line.set_ydata(cross_section)
        
        # Draw the line for cross-section
        cross_line = ax1.axhline(y=0, color='r', linestyle='-', linewidth=2, alpha=0.7)
        
        return [contour, line, time_text, cross_line] + source_points
    
    # Create animation
    anim = FuncAnimation(fig, update, frames=frames, interval=50, blit=False)
    
    # Save as GIF
    writer = PillowWriter(fps=10)
    anim.save(os.path.join(figures_dir, 'cross_section_animation.gif'), writer=writer)
    print("Cross-section animation saved!")
    plt.close()

def main():
    print("Generating improved wave interference visualizations...")
    
    # Create basic interference patterns with proper scales
    print("Creating interference patterns...")
    plot_interference_pattern(3)  # Triangle
    plot_interference_pattern(4)  # Square
    plot_interference_pattern(6)  # Hexagon
    
    # Create 3D surface plots
    print("Creating 3D surface plots...")
    create_3d_surface_plot(3)  # Triangle
    create_3d_surface_plot(4)  # Square
    
    # Create phase difference comparison with proper scales
    print("Creating phase difference comparison...")
    create_phase_difference_comparison()
    
    # Create animations
    print("Creating animations...")
    create_phase_difference_animation()
    create_wave_propagation_animation(3)  # Triangle
    create_wave_propagation_animation(4)  # Square
    create_cross_section_animation()
    
    print("All visualizations generated successfully!")

if __name__ == "__main__":
    main()
