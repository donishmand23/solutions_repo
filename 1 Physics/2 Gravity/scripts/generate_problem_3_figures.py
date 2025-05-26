#!/usr/bin/env python3
"""
Problem 3: Trajectories of a Freely Released Payload Near Earth
Simulation and Visualization Script

This script generates all necessary figures for the problem solution.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle
import os

# Constants
G = 6.67430e-11  # Gravitational constant (m^3/kg/s^2)
M_EARTH = 5.972e24  # Earth's mass (kg)
R_EARTH = 6.371e6  # Earth's radius (m)
MU = G * M_EARTH  # Standard gravitational parameter

# Create directories if they don't exist
os.makedirs('../figures', exist_ok=True)

class PayloadTrajectory:
    """Class to simulate payload trajectories near Earth"""
    
    def __init__(self, initial_altitude, initial_velocity):
        """
        Initialize trajectory with given conditions
        
        Parameters:
        initial_altitude: height above Earth's surface (m)
        initial_velocity: initial tangential velocity (m/s)
        """
        self.h0 = initial_altitude
        self.v0 = initial_velocity
        self.r0 = R_EARTH + initial_altitude
        
        # Initial state vector [x, y, vx, vy]
        self.state0 = np.array([0, self.r0, self.v0, 0])
        
        # Storage for trajectory data
        self.time = []
        self.position = []
        self.velocity = []
        self.energy = []
        
    def derivatives(self, state, t):
        """Calculate derivatives for the equations of motion"""
        x, y, vx, vy = state
        r = np.sqrt(x**2 + y**2)
        
        # Gravitational acceleration
        ax = -MU * x / r**3
        ay = -MU * y / r**3
        
        return np.array([vx, vy, ax, ay])
    
    def rk4_step(self, state, t, dt):
        """4th order Runge-Kutta integration step"""
        k1 = self.derivatives(state, t)
        k2 = self.derivatives(state + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = self.derivatives(state + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = self.derivatives(state + dt * k3, t + dt)
        
        return state + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
    
    def simulate(self, t_max=20000, dt=1.0):
        """Simulate the trajectory"""
        self.time = [0]
        self.position = [self.state0[:2].copy()]
        self.velocity = [self.state0[2:].copy()]
        
        state = self.state0.copy()
        t = 0
        
        while t < t_max:
            # Adaptive timestep near Earth
            r = np.sqrt(state[0]**2 + state[1]**2)
            if r < 2 * R_EARTH:
                dt_use = 0.1
            else:
                dt_use = dt
            
            # Integration step
            state = self.rk4_step(state, t, dt_use)
            t += dt_use
            
            # Store data
            self.time.append(t)
            self.position.append(state[:2].copy())
            self.velocity.append(state[2:].copy())
            
            # Check for collision with Earth
            if r < R_EARTH:
                break
            
            # Check for escape (very far from Earth)
            if r > 10 * R_EARTH:
                break
        
        # Convert to arrays
        self.time = np.array(self.time)
        self.position = np.array(self.position)
        self.velocity = np.array(self.velocity)
        
        # Calculate energy
        self._calculate_energy()
    
    def _calculate_energy(self):
        """Calculate kinetic, potential, and total energy"""
        r = np.sqrt(self.position[:, 0]**2 + self.position[:, 1]**2)
        v = np.sqrt(self.velocity[:, 0]**2 + self.velocity[:, 1]**2)
        
        self.kinetic_energy = 0.5 * v**2
        self.potential_energy = -MU / r
        self.total_energy = self.kinetic_energy + self.potential_energy
    
    def get_orbital_parameters(self):
        """Calculate orbital parameters"""
        r = np.sqrt(self.position[:, 0]**2 + self.position[:, 1]**2)
        v = np.sqrt(self.velocity[:, 0]**2 + self.velocity[:, 1]**2)
        
        # Specific orbital energy
        epsilon = self.total_energy[0]
        
        # Semi-major axis (for elliptical orbits)
        if epsilon < 0:
            a = -MU / (2 * epsilon)
        else:
            a = np.inf
        
        # Eccentricity
        h = np.cross(self.position[0], self.velocity[0])  # Specific angular momentum
        if epsilon < 0:
            e = np.sqrt(1 + 2 * epsilon * h**2 / MU**2)
        else:
            e = np.sqrt(1 + 2 * epsilon * h**2 / MU**2)
        
        # Perigee and apogee
        if e < 1:
            r_perigee = a * (1 - e)
            r_apogee = a * (1 + e)
            h_perigee = r_perigee - R_EARTH
            h_apogee = r_apogee - R_EARTH
        else:
            r_perigee = np.min(r)
            r_apogee = np.inf
            h_perigee = r_perigee - R_EARTH
            h_apogee = np.inf
        
        # Orbital period (for elliptical orbits)
        if epsilon < 0 and a > 0:
            T = 2 * np.pi * np.sqrt(a**3 / MU)
        else:
            T = np.inf
        
        return {
            'semi_major_axis': a,
            'eccentricity': e,
            'perigee_altitude': h_perigee,
            'apogee_altitude': h_apogee,
            'period': T,
            'energy': epsilon
        }

def plot_trajectories():
    """Generate the main trajectory plot"""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Earth - make it more visible with better coloring
    earth = Circle((0, 0), R_EARTH/1e3, color='royalblue', alpha=0.8, label='Earth')
    ax.add_patch(earth)
    
    # Initial position - at 800 km above Earth, at the top of the plot
    h0 = 800e3  # 800 km
    
    # Initial state will be at the top of Earth (y-axis)
    initial_y = R_EARTH + h0  # Distance from Earth's center
    
    # Mark initial position
    ax.plot(0, initial_y/1e3, 'ro', markersize=8, label='Initial Position (800 km altitude)')
    
    # Velocity range from 5 km/s to 13 km/s with smaller increments for detail
    velocities = np.array([5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 7.7, 8.0, 9.0, 10.0, 11.2, 12.0, 13.0]) * 1e3
    
    # Color map - use a better color scheme to distinguish trajectories
    colors = plt.cm.plasma(np.linspace(0, 1, len(velocities)))
    
    # Dictionary to categorize trajectories
    trajectory_types = {
        'Impact': [],
        'Elliptical': [],
        'Circular': [],
        'Escape': []
    }
    
    for i, v0 in enumerate(velocities):
        # Initialize with initial position at the top of Earth
        traj = PayloadTrajectory(h0, v0)
        traj.simulate(t_max=30000)  # Increase simulation time to see complete trajectories
        
        # Convert to km
        x_km = traj.position[:, 0] / 1e3
        y_km = traj.position[:, 1] / 1e3
        
        # Get orbital parameters to categorize trajectory
        params = traj.get_orbital_parameters()
        e = params['eccentricity']
        
        # Categorize trajectory
        if params['perigee_altitude'] < 0:  # Impact with Earth
            category = 'Impact'
        elif e < 0.01:  # Nearly circular
            category = 'Circular'
        elif e < 1:  # Elliptical
            category = 'Elliptical'
        else:  # Hyperbolic (escape)
            category = 'Escape'
        
        # Plot trajectory
        line = ax.plot(x_km, y_km, color=colors[i], linewidth=2, 
                label=f'v₀ = {v0/1e3:.1f} km/s ({category})')
        
        # Store for later reference
        trajectory_types[category].append((v0/1e3, line[0]))
    
    # Formatting with better scaling to focus on near-Earth trajectories
    ax.set_xlim(-12000, 12000)
    ax.set_ylim(-12000, 12000)
    ax.set_xlabel('X (km)', fontsize=14)
    ax.set_ylabel('Y (km)', fontsize=14)
    ax.set_title('Payload Trajectories for Different Initial Velocities\n(Initial Position: 800 km above Earth)', fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Create a more compact and readable legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11)
    
    # Add key information to the plot
    ax.text(0.02, 0.98, 'Trajectory Types:\n'
            '• Impact: v < 7.0 km/s\n'
            '• Elliptical: 7.0-7.7 km/s & 7.7-11.2 km/s\n'
            '• Circular: ~7.7 km/s\n'
            '• Escape: > 11.2 km/s', 
            transform=ax.transAxes, fontsize=12,
            bbox=dict(facecolor='white', alpha=0.7), verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig('../figures/payload_trajectories.png', dpi=300, bbox_inches='tight')
    print("Trajectory plot complete.")
    plt.close()

def plot_orbital_parameters():
    """Plot orbital parameters vs initial velocity"""
    h0 = 800e3  # 800 km
    velocities = np.linspace(5.0e3, 13.0e3, 100)  # More data points for smoother curves
    
    # Storage for parameters
    perigee_altitudes = []
    apogee_altitudes = []
    periods = []
    eccentricities = []
    energies = []
    
    # Important velocity thresholds
    v_circ = np.sqrt(MU / (R_EARTH + h0))  # Circular orbit velocity
    v_esc = np.sqrt(2 * MU / (R_EARTH + h0))  # Escape velocity
    
    for v0 in velocities:
        traj = PayloadTrajectory(h0, v0)
        traj.simulate(t_max=30000)  # Consistent with other simulations
        params = traj.get_orbital_parameters()
        
        perigee_altitudes.append(params['perigee_altitude'] / 1e3)  # Convert to km
        apogee_altitudes.append(params['apogee_altitude'] / 1e3 if params['apogee_altitude'] < 1e10 else np.nan)
        periods.append(params['period'] / 3600 if params['period'] < 1e10 else np.nan)  # Convert to hours
        eccentricities.append(params['eccentricity'])
        energies.append(params['energy'] / 1e6)  # Convert to MJ/kg for better scale
    
    # Create subplots with better layout
    fig, axs = plt.subplots(2, 2, figsize=(14, 12))
    
    # Common x-axis range for all plots
    v_range = np.array(velocities) / 1e3  # km/s
    
    # Perigee altitude
    axs[0, 0].plot(v_range, perigee_altitudes, 'b-', linewidth=2)
    axs[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=1.5, label='Earth Surface')
    # Mark circular and escape velocities
    axs[0, 0].axvline(x=v_circ/1e3, color='g', linestyle='--', linewidth=1.5, label=f'Circular Velocity: {v_circ/1e3:.2f} km/s')
    axs[0, 0].axvline(x=v_esc/1e3, color='purple', linestyle='--', linewidth=1.5, label=f'Escape Velocity: {v_esc/1e3:.2f} km/s')
    axs[0, 0].set_xlabel('Initial Velocity (km/s)', fontsize=14)
    axs[0, 0].set_ylabel('Perigee Altitude (km)', fontsize=14)
    axs[0, 0].set_title('Perigee Altitude vs Initial Velocity', fontsize=16)
    axs[0, 0].grid(True, alpha=0.3)
    axs[0, 0].legend(fontsize=12)
    
    # Apogee altitude with better formatting for infinity values
    axs[0, 1].plot(v_range, apogee_altitudes, 'g-', linewidth=2)
    # Mark regions
    axs[0, 1].axvline(x=v_circ/1e3, color='g', linestyle='--', linewidth=1.5, label='Circular Velocity')
    axs[0, 1].axvline(x=v_esc/1e3, color='purple', linestyle='--', linewidth=1.5, label='Escape Velocity')
    axs[0, 1].set_xlabel('Initial Velocity (km/s)', fontsize=14)
    axs[0, 1].set_ylabel('Apogee Altitude (km)', fontsize=14)
    axs[0, 1].set_title('Apogee Altitude vs Initial Velocity', fontsize=16)
    axs[0, 1].grid(True, alpha=0.3)
    axs[0, 1].legend(fontsize=12)
    
    # Set reasonable y-axis limits for apogee plot
    axs[0, 1].set_ylim(-1000, 50000)  # Limit to show meaningful altitude range
    axs[0, 1].text(v_esc/1e3 + 0.2, 40000, 'Hyperbolic Escape\n(Infinite Apogee)', 
                  fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
    
    # Orbital period with better formatting
    axs[1, 0].plot(v_range, periods, 'r-', linewidth=2)
    axs[1, 0].axvline(x=v_circ/1e3, color='g', linestyle='--', linewidth=1.5, label='Circular Velocity')
    axs[1, 0].axvline(x=v_esc/1e3, color='purple', linestyle='--', linewidth=1.5, label='Escape Velocity')
    axs[1, 0].set_xlabel('Initial Velocity (km/s)', fontsize=14)
    axs[1, 0].set_ylabel('Orbital Period (hours)', fontsize=14)
    axs[1, 0].set_title('Orbital Period vs Initial Velocity', fontsize=16)
    axs[1, 0].grid(True, alpha=0.3)
    axs[1, 0].set_ylim(0, 10)  # Limit to show meaningful period range
    axs[1, 0].legend(fontsize=12)
    
    # Eccentricity
    axs[1, 1].plot(v_range, eccentricities, 'm-', linewidth=2)
    axs[1, 1].axhline(y=0, color='g', linestyle='--', linewidth=1.5, label='Circular (e=0)')
    axs[1, 1].axhline(y=1, color='r', linestyle='--', linewidth=1.5, label='Parabolic (e=1)')
    axs[1, 1].axvline(x=v_circ/1e3, color='g', linestyle='--', linewidth=1.5, label='Circular Velocity')
    axs[1, 1].axvline(x=v_esc/1e3, color='purple', linestyle='--', linewidth=1.5, label='Escape Velocity')
    axs[1, 1].set_xlabel('Initial Velocity (km/s)', fontsize=14)
    axs[1, 1].set_ylabel('Eccentricity', fontsize=14)
    axs[1, 1].set_title('Orbit Eccentricity vs Initial Velocity', fontsize=16)
    axs[1, 1].grid(True, alpha=0.3)
    axs[1, 1].set_ylim(-0.1, 2.0)  # Show meaningful eccentricity range
    axs[1, 1].legend(fontsize=12)
    
    # Add text annotations explaining regions
    fig.text(0.5, 0.01, 'Initial velocity ranges and trajectory types:\n'
             '• v < 7.0 km/s: Impact trajectories (perigee below Earth surface)\n'
             f'• 7.0 km/s < v < {v_circ/1e3:.1f} km/s: Elliptical orbits\n'
             f'• v ≈ {v_circ/1e3:.1f} km/s: Circular orbit\n'
             f'• {v_circ/1e3:.1f} km/s < v < {v_esc/1e3:.1f} km/s: Elliptical orbits\n'
             f'• v ≥ {v_esc/1e3:.1f} km/s: Escape trajectories (hyperbolic)',
             fontsize=14, horizontalalignment='center',
             bbox=dict(facecolor='lightgray', alpha=0.5))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Make room for the annotation at the bottom
    plt.savefig('../figures/orbital_parameters.png', dpi=300, bbox_inches='tight')
    print("Orbital parameters plot complete.")
    plt.close()

def plot_energy_conservation():
    """Plot energy conservation for different trajectories"""
    h0 = 800e3  # 800 km
    velocities = [6.0e3, 7.7e3, 11.2e3]  # Impact, circular, escape
    labels = ['Impact (6.0 km/s)', 'Near Circular (7.7 km/s)', 'Escape (11.2 km/s)']
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    for i, (v0, label) in enumerate(zip(velocities, labels)):
        traj = PayloadTrajectory(h0, v0)
        traj.simulate(t_max=5000)
        
        # Convert to MJ/kg
        ke = traj.kinetic_energy / 1e6
        pe = traj.potential_energy / 1e6
        te = traj.total_energy / 1e6
        
        axes[i].plot(traj.time, ke, 'r-', linewidth=2, label='Kinetic')
        axes[i].plot(traj.time, pe, 'b-', linewidth=2, label='Potential')
        axes[i].plot(traj.time, te, 'k--', linewidth=2, label='Total')
        
        axes[i].set_ylabel('Specific Energy (MJ/kg)')
        axes[i].set_title(f'Energy Conservation: {label}')
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()
        
        # Add text showing energy conservation
        energy_variation = (np.max(te) - np.min(te)) / np.abs(np.mean(te)) * 100
        axes[i].text(0.02, 0.95, f'Energy variation: {energy_variation:.2e}%', 
                    transform=axes[i].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    axes[-1].set_xlabel('Time (s)')
    
    plt.tight_layout()
    plt.savefig('../figures/energy_conservation.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_animation():
    """Create animated visualization of trajectories"""
    h0 = 800e3  # 800 km
    # Selected velocities to demonstrate different trajectory types
    velocities = [5.5e3, 7.0e3, 7.7e3, 9.0e3, 11.2e3]
    colors = ['red', 'orange', 'green', 'blue', 'purple']
    labels = ['5.5 km/s (Impact)', '7.0 km/s (Elliptical)', '7.7 km/s (Circular)', 
              '9.0 km/s (Elliptical)', '11.2 km/s (Escape)']
    
    # Simulate all trajectories
    trajectories = []
    for v0 in velocities:
        traj = PayloadTrajectory(h0, v0)
        traj.simulate(t_max=30000)  # Longer simulation time
        trajectories.append(traj)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Earth - make it more visible
    earth = Circle((0, 0), R_EARTH/1e3, color='royalblue', alpha=0.8)
    ax.add_patch(earth)
    
    # Initial position
    initial_y = R_EARTH + h0  # Distance from Earth's center
    initial_point, = ax.plot(0, initial_y/1e3, 'ro', markersize=8, label='Initial Position')
    
    # Initialize lines and points
    lines = []
    points = []
    for i in range(len(velocities)):
        line, = ax.plot([], [], color=colors[i], linewidth=2, alpha=0.8, label=labels[i])
        point, = ax.plot([], [], 'o', color=colors[i], markersize=8)
        lines.append(line)
        points.append(point)
    
    # Set limits
    ax.set_xlim(-12000, 12000)
    ax.set_ylim(-12000, 12000)
    ax.set_xlabel('X (km)', fontsize=14)
    ax.set_ylabel('Y (km)', fontsize=14)
    ax.set_title('Animated Payload Trajectories\n(Initial Position: 800 km above Earth)', fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Create a more compact and readable legend
    ax.legend(loc='upper right', fontsize=11)
    
    # Time text
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                       fontsize=12, bbox=dict(facecolor='white', alpha=0.7),
                       verticalalignment='top')
    
    def init():
        for line in lines:
            line.set_data([], [])
        for point in points:
            point.set_data([], [])
        time_text.set_text('')
        return lines + points + [time_text]
    
    def animate(frame):
        t = frame * 100  # Time step (increased for faster animation)
        
        for i, traj in enumerate(trajectories):
            # Find index for current time
            idx = np.searchsorted(traj.time, t)
            if idx >= len(traj.time):
                idx = len(traj.time) - 1
            
            # Update line
            x_km = traj.position[:idx+1, 0] / 1e3
            y_km = traj.position[:idx+1, 1] / 1e3
            lines[i].set_data(x_km, y_km)
            
            # Update point - ensure it's a sequence
            if idx > 0:
                points[i].set_data([x_km[-1]], [y_km[-1]])  # Use single-element arrays
            else:
                points[i].set_data([], [])
        
        # Calculate elapsed time in appropriate units
        if t < 60:
            time_str = f'Time: {t:.0f} s'
        elif t < 3600:
            time_str = f'Time: {t/60:.1f} min'
        else:
            time_str = f'Time: {t/3600:.1f} hours'
            
        time_text.set_text(time_str)
        return lines + points + [time_text, initial_point]
    
    # Create animation with more frames for smoother animation
    anim = FuncAnimation(fig, animate, init_func=init, frames=300, 
                        interval=40, blit=True, repeat=True)
    
    # Save animation with better quality
    writer = PillowWriter(fps=25)
    anim.save('../figures/animated_trajectory.gif', writer=writer)
    print("Animation complete.")
    plt.close()

def main():
    """Main function to generate all figures"""
    print("Generating trajectory plot...")
    plot_trajectories()
    print("Trajectory plot complete.")
    
    print("Generating orbital parameters plot...")
    plot_orbital_parameters()
    print("Orbital parameters plot complete.")
    
    print("Generating energy conservation plot...")
    plot_energy_conservation()
    print("Energy conservation plot complete.")
    
    print("Creating animated visualization...")
    create_animation()
    print("Animation complete.")
    
    print("\nAll figures have been generated successfully!")
    print("Check the '../figures' directory for the output files.")

if __name__ == "__main__":
    main()