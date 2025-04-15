#!/usr/bin/env python3
"""
Script to generate projectile motion figures for Problem 1.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
import os

# Create figures directory if it doesn't exist
figures_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'figures')
os.makedirs(figures_dir, exist_ok=True)

def plot_trajectories_part_a():
    """Generate the first figure showing trajectories at different velocities with same angle."""
    plt.figure(figsize=(10, 6))
    
    # Set up the plot
    plt.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
    
    # Constants
    g = 9.8  # m/s^2
    theta = np.radians(45)  # 45 degrees in radians
    
    # Initial velocities
    velocities = [30, 40, 50]  # m/s
    colors = ['#B22222', '#9370DB', '#2E8B57']  # Red, Purple, Green
    
    # Calculate ranges
    ranges = [v**2 * np.sin(2*theta) / g for v in velocities]
    
    # Plot trajectories
    x_max = max(ranges) * 1.05
    for i, v in enumerate(velocities):
        # Time of flight
        t_flight = 2 * v * np.sin(theta) / g
        
        # Time points
        t = np.linspace(0, t_flight, 1000)
        
        # Trajectory
        x = v * np.cos(theta) * t
        y = v * np.sin(theta) * t - 0.5 * g * t**2
        
        # Plot
        plt.plot(x, y, color=colors[i], linewidth=2)
        
        # Add range marker
        plt.annotate('', xy=(ranges[i], 0), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='-', color='black', linestyle='--'))
        
        # Add range text
        plt.text(ranges[i]/2, -10, f'R = {ranges[i]:.1f} m', ha='center')
    
    # Add velocity vectors at origin
    for i, v in enumerate(velocities):
        dx = v * np.cos(theta) * 0.8
        dy = v * np.sin(theta) * 0.8
        plt.arrow(0, 0, dx, dy, head_width=3, head_length=5, fc=colors[i], ec=colors[i], linewidth=2)
        plt.text(-20, i*10 + 5, f'{v} m/s', color=colors[i], fontsize=12)
    
    # Add angle label
    arc = Arc((0, 0), 30, 30, theta1=0, theta2=45, color='black', linewidth=1)
    plt.gca().add_patch(arc)
    plt.text(10, 5, '45°', fontsize=12)
    
    # Add coordinate system
    plt.arrow(x_max*0.85, 50, 20, 0, head_width=5, head_length=5, fc='black', ec='black')
    plt.arrow(x_max*0.85, 50, 0, 20, head_width=5, head_length=5, fc='black', ec='black')
    plt.text(x_max*0.85 + 25, 50, 'x', fontsize=12)
    plt.text(x_max*0.85, 75, 'y', fontsize=12)
    
    # Set plot limits and labels
    plt.xlim(-30, x_max)
    plt.ylim(-20, max([v**2 * np.sin(theta)**2 / (2*g) for v in velocities]) * 1.2)
    plt.title('Projectile Motion: Same Angle (45°), Different Initial Velocities')
    plt.grid(False)
    plt.axis('off')
    
    # Save figure
    plt.savefig(os.path.join(figures_dir, 'projectile_motion_part_a.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_trajectories_part_b():
    """Generate the second figure showing trajectories at different angles with same velocity."""
    plt.figure(figsize=(10, 6))
    
    # Set up the plot
    plt.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
    
    # Constants
    g = 9.8  # m/s^2
    v0 = 50  # m/s
    
    # Angles
    angles_deg = [15, 45, 75]  # degrees
    angles = [np.radians(angle) for angle in angles_deg]  # radians
    colors = ['#B22222', '#9370DB', '#2E8B57']  # Red, Purple, Green
    
    # Calculate ranges
    ranges = [v0**2 * np.sin(2*angle) / g for angle in angles]
    
    # Plot trajectories
    x_max = max(ranges) * 1.05
    for i, angle in enumerate(angles):
        # Time of flight
        t_flight = 2 * v0 * np.sin(angle) / g
        
        # Time points
        t = np.linspace(0, t_flight, 1000)
        
        # Trajectory
        x = v0 * np.cos(angle) * t
        y = v0 * np.sin(angle) * t - 0.5 * g * t**2
        
        # Plot
        plt.plot(x, y, color=colors[i], linewidth=2)
        
        # Add range marker for first two trajectories
        if i < 2:
            plt.annotate('', xy=(ranges[i], 0), xytext=(0, 0),
                        arrowprops=dict(arrowstyle='-', color='black', linestyle='--'))
            
            # Add range text
            plt.text(ranges[i]/2, -10, f'R = {ranges[i]:.1f} m', ha='center')
    
    # Add range marker for the last trajectory
    plt.text(ranges[2]/2, -10, f'R = {ranges[2]:.1f} m', ha='center')
    
    # Add velocity vectors at origin
    for i, angle in enumerate(angles):
        dx = v0 * np.cos(angle) * 0.5
        dy = v0 * np.sin(angle) * 0.5
        plt.arrow(0, 0, dx, dy, head_width=3, head_length=5, fc=colors[i], ec=colors[i], linewidth=2)
    
    # Add angle labels
    for i, angle_deg in enumerate(angles_deg):
        arc = Arc((0, 0), 30, 30, theta1=0, theta2=angle_deg, color='black', linewidth=1)
        plt.gca().add_patch(arc)
        plt.text(10, i*5 + 5, f'{angle_deg}°', fontsize=12)
    
    # Add initial velocity label
    plt.text(-30, 100, f'v₀ = {v0} m/s', fontsize=12)
    
    # Set plot limits and labels
    plt.xlim(-40, x_max)
    plt.ylim(-20, v0**2 * np.sin(angles[2])**2 / (2*g) * 1.2)
    plt.title('Projectile Motion: Same Initial Velocity (50 m/s), Different Angles')
    plt.grid(False)
    plt.axis('off')
    
    # Save figure
    plt.savefig(os.path.join(figures_dir, 'projectile_motion_part_b.png'), dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    plot_trajectories_part_a()
    plot_trajectories_part_b()
    print("Figures generated successfully in the 'figures' directory.")
