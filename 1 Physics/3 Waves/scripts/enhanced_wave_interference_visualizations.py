import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import os

# Create figures directory if it doesn't exist
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
figures_dir = os.path.join(parent_dir, 'figures')
if not os.path.exists(figures_dir):
    os.makedirs(figures_dir)

# Wave parameters
A = 1.0         # Amplitude
lambda_val = 5.0  # Wavelength
k = 2 * np.pi / lambda_val  # Wave number
f = 1.0         # Frequency
omega = 2 * np.pi * f  # Angular frequency

def calculate_displacement(x, y, sources, t=0, phases=None):
    """
    Calculate the total displacement at point (x,y) due to all sources at time t.
    
    Parameters:
    - x, y: Grid coordinates
    - sources: List of (x,y) coordinates of wave sources
    - t: Time
    - phases: List of initial phases for each source (default: all 0)
    
    Returns:
    - total: Total displacement at each point
    """
    total = np.zeros_like(x)
    
    if phases is None:
        phases = [0] * len(sources)
    
    for (x0, y0), phi in zip(sources, phases):
        r = np.sqrt((x - x0)**2 + (y - y0)**2)
        # Avoid division by zero at source positions
        r = np.maximum(r, 1e-10)
        # Wave equation with amplitude decay
        displacement = A * np.cos(k*r - omega*t + phi) / np.sqrt(r)
        total += displacement
    
    return total

def generate_polygon_vertices(n, radius):
    """Generate vertices of a regular polygon with n sides and given radius."""
    vertices = []
    for i in range(n):
        angle = 2 * np.pi * i / n
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        vertices.append((x, y))
    return vertices

def create_3d_surface_plot(n_sides, size=20, resolution=200):
    """Create a 3D surface plot of the interference pattern."""
    # Generate source positions
    radius = 10
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
    
    # Plot the surface
    surf = ax.plot_surface(X, Y, Z, cmap='coolwarm', linewidth=0, antialiased=True)
    
    # Plot source positions
    for x0, y0 in sources:
        ax.scatter(x0, y0, 0, color='black', s=50, marker='o')
    
    # Set plot properties
    ax.set_title(f'3D Visualization of Interference Pattern ({n_sides} Sources)', fontsize=16)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Displacement')
    
    # Add a color bar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Displacement')
    
    # Save the figure
    plt.savefig(os.path.join(figures_dir, f'{n_sides}_sided_3d_interference.png'), dpi=300)
    plt.close()

def create_phase_difference_comparison(size=20, resolution=200):
    """Create a comparison of interference patterns with different phase relationships."""
    # Create a figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    # Source positions (two sources)
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
        
        # Create contour plot
        contour = axes[i].contourf(X, Y, Z, 50, cmap='coolwarm')
        
        # Plot source positions
        for x0, y0 in sources:
            axes[i].plot(x0, y0, 'ko', markersize=8)
        
        # Set plot properties
        axes[i].set_title(f'Phase Difference: {label}')
        axes[i].set_xlabel('X Position')
        axes[i].set_ylabel('Y Position')
        axes[i].set_aspect('equal')
        axes[i].grid(True, linestyle='--', alpha=0.7)
    
    # Add a colorbar to the figure
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(contour, cax=cbar_ax, label='Displacement')
    
    plt.suptitle('Effect of Phase Difference on Two-Source Interference Patterns', fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    
    # Save the figure
    plt.savefig(os.path.join(figures_dir, 'phase_difference_comparison.png'), dpi=300)
    plt.close()

def create_wavelength_comparison(n_sides=4, size=20, resolution=200):
    """Create a comparison of interference patterns with different wavelengths."""
    # Create a figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    # Generate source positions
    radius = 10
    sources = generate_polygon_vertices(n_sides, radius)
    
    # Different wavelengths
    wavelengths = [2.5, 5.0, 7.5, 10.0]
    
    # Create a grid of points
    x = np.linspace(-size, size, resolution)
    y = np.linspace(-size, size, resolution)
    X, Y = np.meshgrid(x, y)
    
    # Plot for each wavelength
    for i, lambda_val in enumerate(wavelengths):
        # Recalculate wave number for this wavelength
        k_val = 2 * np.pi / lambda_val
        
        # Calculate displacement
        Z = np.zeros_like(X)
        for x0, y0 in sources:
            r = np.sqrt((X - x0)**2 + (Y - y0)**2)
            r = np.maximum(r, 1e-10)
            Z += A * np.cos(k_val * r) / np.sqrt(r)
        
        # Create contour plot
        contour = axes[i].contourf(X, Y, Z, 50, cmap='coolwarm')
        
        # Plot source positions
        for x0, y0 in sources:
            axes[i].plot(x0, y0, 'ko', markersize=8)
        
        # Set plot properties
        axes[i].set_title(f'Wavelength: {lambda_val} units')
        axes[i].set_xlabel('X Position')
        axes[i].set_ylabel('Y Position')
        axes[i].set_aspect('equal')
        axes[i].grid(True, linestyle='--', alpha=0.7)
    
    # Add a colorbar to the figure
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(contour, cax=cbar_ax, label='Displacement')
    
    plt.suptitle('Effect of Wavelength on Interference Patterns', fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    
    # Save the figure
    plt.savefig(os.path.join(figures_dir, 'wavelength_comparison.png'), dpi=300)
    plt.close()

def create_cross_section_plot(n_sides=4, size=20, resolution=500):
    """Create a plot showing cross-sections of the interference pattern."""
    # Generate source positions
    radius = 10
    sources = generate_polygon_vertices(n_sides, radius)
    
    # Create a grid of points
    x = np.linspace(-size, size, resolution)
    y = np.linspace(-size, size, resolution)
    X, Y = np.meshgrid(x, y)
    
    # Calculate displacement at each point
    Z = calculate_displacement(X, Y, sources)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot the 2D interference pattern
    contour = ax1.contourf(X, Y, Z, 50, cmap='coolwarm')
    fig.colorbar(contour, ax=ax1, label='Displacement')
    
    # Plot source positions
    for x0, y0 in sources:
        ax1.plot(x0, y0, 'ko', markersize=8)
    
    # Add cross-section lines
    ax1.axhline(y=0, color='black', linestyle='--')
    ax1.axvline(x=0, color='black', linestyle='--')
    
    # Set plot properties for the 2D plot
    ax1.set_title(f'Interference Pattern for {n_sides} Sources with Cross-Section Lines')
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    ax1.set_aspect('equal')
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Plot cross-sections
    # Horizontal cross-section (y = 0)
    ax2.plot(x, Z[resolution//2, :], 'b-', label='Horizontal (y = 0)')
    
    # Vertical cross-section (x = 0)
    ax2.plot(y, Z[:, resolution//2], 'r-', label='Vertical (x = 0)')
    
    # Set plot properties for the cross-section plot
    ax2.set_title('Cross-Sections of the Interference Pattern')
    ax2.set_xlabel('Position')
    ax2.set_ylabel('Displacement')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(os.path.join(figures_dir, f'{n_sides}_sided_cross_section.png'), dpi=300)
    plt.close()

def create_source_number_comparison(size=20, resolution=200):
    """Create a comparison of interference patterns with different numbers of sources."""
    # Create a figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Different numbers of sources
    n_sides_list = [2, 3, 4, 5, 6, 8]
    
    # Create a grid of points
    x = np.linspace(-size, size, resolution)
    y = np.linspace(-size, size, resolution)
    X, Y = np.meshgrid(x, y)
    
    # Plot for each number of sources
    for i, n_sides in enumerate(n_sides_list):
        # Generate source positions
        radius = 10
        sources = generate_polygon_vertices(n_sides, radius)
        
        # Calculate displacement
        Z = calculate_displacement(X, Y, sources)
        
        # Create contour plot
        contour = axes[i].contourf(X, Y, Z, 50, cmap='coolwarm')
        
        # Plot source positions
        for x0, y0 in sources:
            axes[i].plot(x0, y0, 'ko', markersize=8)
        
        # Set plot properties
        axes[i].set_title(f'{n_sides} Sources')
        axes[i].set_xlabel('X Position')
        axes[i].set_ylabel('Y Position')
        axes[i].set_aspect('equal')
        axes[i].grid(True, linestyle='--', alpha=0.7)
    
    # Add a colorbar to the figure
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(contour, cax=cbar_ax, label='Displacement')
    
    plt.suptitle('Effect of Source Number on Interference Patterns', fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    
    # Save the figure
    plt.savefig(os.path.join(figures_dir, 'source_number_comparison.png'), dpi=300)
    plt.close()

def main():
    print("Generating enhanced wave interference visualizations...")
    
    # Create 3D surface plots
    print("Creating 3D surface plots...")
    create_3d_surface_plot(3)  # Triangle
    create_3d_surface_plot(4)  # Square
    
    # Create phase difference comparison
    print("Creating phase difference comparison...")
    create_phase_difference_comparison()
    
    # Create wavelength comparison
    print("Creating wavelength comparison...")
    create_wavelength_comparison()
    
    # Create cross-section plots
    print("Creating cross-section plots...")
    create_cross_section_plot()
    
    # Create source number comparison
    print("Creating source number comparison...")
    create_source_number_comparison()
    
    print("All enhanced visualizations generated successfully!")

if __name__ == "__main__":
    main()
