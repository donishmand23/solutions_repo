import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import cm
import os

# Create figures directory if it doesn't exist
figures_dir = 'figures'
if not os.path.exists(figures_dir):
    os.makedirs(figures_dir)

# Wave parameters
A = 1.0         # Amplitude
lambda_val = 5.0  # Wavelength
k = 2 * np.pi / lambda_val  # Wave number
f = 1.0         # Frequency
omega = 2 * np.pi * f  # Angular frequency

def calculate_displacement(x, y, sources, t=0):
    """Calculate the total displacement at point (x,y) due to all sources at time t."""
    total = np.zeros_like(x)
    
    for x0, y0 in sources:
        r = np.sqrt((x - x0)**2 + (y - y0)**2)
        # Avoid division by zero at source positions
        r = np.maximum(r, 1e-10)
        # Wave equation with amplitude decay
        displacement = A * np.cos(k*r - omega*t) / np.sqrt(r)
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

def plot_interference_pattern(n_sides, size=20, resolution=500):
    """Plot the interference pattern for a regular polygon with n_sides."""
    # Generate source positions (vertices of the polygon)
    radius = 10  # Radius of the polygon
    sources = generate_polygon_vertices(n_sides, radius)
    
    # Create a grid of points
    x = np.linspace(-size, size, resolution)
    y = np.linspace(-size, size, resolution)
    X, Y = np.meshgrid(x, y)
    
    # Calculate displacement at each point
    Z = calculate_displacement(X, Y, sources)
    
    # Plot the interference pattern
    plt.figure(figsize=(10, 8))
    plt.contourf(X, Y, Z, 50, cmap='coolwarm')
    plt.colorbar(label='Displacement')
    
    # Plot source positions
    for x0, y0 in sources:
        plt.plot(x0, y0, 'ko', markersize=8)
    
    # Set plot properties
    plt.title(f'Interference Pattern for {n_sides} Sources in a Regular Polygon')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save the figure
    plt.savefig(os.path.join(figures_dir, f'{n_sides}_sided_polygon_interference.png'), dpi=300)
    plt.close()

def create_time_evolution_snapshots(n_sides, size=20, resolution=200, num_frames=4):
    """Create snapshots of the interference pattern at different time points."""
    # Generate source positions
    radius = 10
    sources = generate_polygon_vertices(n_sides, radius)
    
    # Create a grid of points
    x = np.linspace(-size, size, resolution)
    y = np.linspace(-size, size, resolution)
    X, Y = np.meshgrid(x, y)
    
    # Create a figure with subplots for different time points
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    # Plot snapshots at different time points
    for i, ax in enumerate(axes):
        t = i / num_frames  # Time varies from 0 to 0.75 of a period
        Z = calculate_displacement(X, Y, sources, t=t)
        
        # Create contour plot
        contour = ax.contourf(X, Y, Z, 50, cmap='coolwarm')
        
        # Plot source positions
        for x0, y0 in sources:
            ax.plot(x0, y0, 'ko', markersize=6)
        
        # Set plot properties
        ax.set_title(f'Time t = {t:.2f}T')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')
        ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add a colorbar to the figure
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(contour, cax=cbar_ax, label='Displacement')
    
    plt.suptitle(f'Time Evolution of Interference Pattern ({n_sides} Sources)', fontsize=16)
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    
    # Save the figure
    plt.savefig(os.path.join(figures_dir, f'{n_sides}_sided_interference_time_evolution.png'), dpi=300)
    plt.close()

def create_wave_interference_principles_figure():
    """Create a figure illustrating the principles of constructive and destructive interference."""
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # X values for the waves
    x = np.linspace(0, 20, 1000)
    
    # Wave 1 and Wave 2 (in phase for constructive interference)
    wave1 = np.sin(x)
    wave2 = np.sin(x)
    sum_constructive = wave1 + wave2
    
    # Wave 1 and Wave 2 (out of phase for destructive interference)
    wave3 = np.sin(x)
    wave4 = np.sin(x + np.pi)  # 180 degrees out of phase
    sum_destructive = wave3 + wave4
    
    # Plot constructive interference
    ax1.plot(x, wave1, 'b--', label='Wave 1', alpha=0.7)
    ax1.plot(x, wave2, 'g--', label='Wave 2', alpha=0.7)
    ax1.plot(x, sum_constructive, 'r-', label='Resultant Wave')
    ax1.set_title('Constructive Interference (Waves in Phase)')
    ax1.set_ylabel('Amplitude')
    ax1.set_xlim(0, 20)
    ax1.set_ylim(-2.5, 2.5)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()
    
    # Plot destructive interference
    ax2.plot(x, wave3, 'b--', label='Wave 1', alpha=0.7)
    ax2.plot(x, wave4, 'g--', label='Wave 2', alpha=0.7)
    ax2.plot(x, sum_destructive, 'r-', label='Resultant Wave')
    ax2.set_title('Destructive Interference (Waves 180° Out of Phase)')
    ax2.set_xlabel('Distance')
    ax2.set_ylabel('Amplitude')
    ax2.set_xlim(0, 20)
    ax2.set_ylim(-2.5, 2.5)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'wave_interference_principles.png'), dpi=300)
    plt.close()

def main():
    print("Generating wave interference figures...")
    
    # Create wave interference principles figure
    print("Creating wave interference principles figure...")
    create_wave_interference_principles_figure()
    
    # Generate interference patterns for different regular polygons
    print("Creating triangle interference pattern...")
    plot_interference_pattern(3)  # Triangle
    
    print("Creating square interference pattern...")
    plot_interference_pattern(4)  # Square
    
    print("Creating hexagon interference pattern...")
    plot_interference_pattern(6)  # Hexagon
    
    # Create time evolution snapshots for the square case
    print("Creating time evolution snapshots for square arrangement...")
    create_time_evolution_snapshots(4)
    
    print("All figures generated successfully!")

if __name__ == "__main__":
    main()
