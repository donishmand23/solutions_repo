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
    
    # Earth
    earth = Circle((0, 0), R_EARTH/1e6, color='blue', alpha=0.7, label='Earth')
    ax.add_patch(earth)
    
    # Initial position
    h0 = 800e3  # 800 km
    
    # Velocity range
    velocities = np.array([5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0]) * 1e3
    
    # Color map
    colors = plt.cm.viridis(np.linspace(0, 1, len(velocities)))
    
    for i, v0 in enumerate(velocities):
        traj = PayloadTrajectory(h0, v0)
        traj.simulate()
        
        # Convert to km
        x_km = traj.position[:, 0] / 1e3
        y_km = traj.position[:, 1] / 1e3
        
        # Plot trajectory
        ax.plot(x_km, y_km, color=colors[i], linewidth=2, 
                label=f'v₀ = {v0/1e3:.1f} km/s')
    
    # Formatting
    ax.set_xlim(-15000, 15000)
    ax.set_ylim(-15000, 15000)
    ax.set_xlabel('X (km)', fontsize=12)
    ax.set_ylabel('Y (km)', fontsize=12)
    ax.set_title('Payload Trajectories for Different Initial Velocities', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig('../figures/payload_trajectories.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_orbital_parameters():
    """Plot orbital parameters vs initial velocity"""
    h0 = 800e3  # 800 km
    velocities = np.linspace(5.0e3, 13.0e3, 50)
    
    # Storage for parameters
    perigee_altitudes = []
    apogee_altitudes = []
    periods = []
    eccentricities = []
    
    for v0 in velocities:
        traj = PayloadTrajectory(h0, v0)
        traj.simulate()
        params = traj.get_orbital_parameters()
        
        perigee_altitudes.append(params['perigee_altitude'] / 1e3)  # Convert to km
        apogee_altitudes.append(params['apogee_altitude'] / 1e3 if params['apogee_altitude'] < 1e10 else np.nan)
        periods.append(params['period'] / 3600 if params['period'] < 1e10 else np.nan)  # Convert to hours
        eccentricities.append(params['eccentricity'])
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Perigee altitude
    axes[0, 0].plot(velocities/1e3, perigee_altitudes, 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Initial Velocity (km/s)')
    axes[0, 0].set_ylabel('Perigee Altitude (km)')
    axes[0, 0].set_title('Perigee Altitude vs Initial Velocity')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Earth Surface')
    axes[0, 0].legend()
    
    # Apogee altitude
    axes[0, 1].plot(velocities/1e3, apogee_altitudes, 'g-', linewidth=2)
    axes[0, 1].set_xlabel('Initial Velocity (km/s)')
    axes[0, 1].set_ylabel('Apogee Altitude (km)')
    axes[0, 1].set_title('Apogee Altitude vs Initial Velocity')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(0, 20000)
    
    # Orbital period
    axes[1, 0].plot(velocities/1e3, periods, 'r-', linewidth=2)
    axes[1, 0].set_xlabel('Initial Velocity (km/s)')
    axes[1, 0].set_ylabel('Orbital Period (hours)')
    axes[1, 0].set_title('Orbital Period vs Initial Velocity')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(0, 10)
    
    # Eccentricity
    axes[1, 1].plot(velocities/1e3, eccentricities, 'm-', linewidth=2)
    axes[1, 1].set_xlabel('Initial Velocity (km/s)')
    axes[1, 1].set_ylabel('Eccentricity')
    axes[1, 1].set_title('Eccentricity vs Initial Velocity')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axhline(y=1, color='k', linestyle='--', alpha=0.5, label='e = 1 (Parabolic)')
    axes[1, 1].legend()
    
    # Add vertical lines for important velocities
    v_circular = np.sqrt(MU / (R_EARTH + h0)) / 1e3
    v_escape = np.sqrt(2 * MU / (R_EARTH + h0)) / 1e3
    
    for ax in axes.flat:
        ax.axvline(x=v_circular, color='orange', linestyle=':', alpha=0.7, label=f'Circular: {v_circular:.2f} km/s')
        ax.axvline(x=v_escape, color='red', linestyle=':', alpha=0.7, label=f'Escape: {v_escape:.2f} km/s')
    
    plt.tight_layout()
    plt.savefig('../figures/orbital_parameters.png', dpi=300, bbox_inches='tight')
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
    velocities = [5.5e3, 7.0e3, 7.7e3, 9.0e3, 11.5e3]
    colors = ['red', 'orange', 'green', 'blue', 'purple']
    labels = ['5.5 km/s (Impact)', '7.0 km/s (Elliptical)', '7.7 km/s (Circular)', 
              '9.0 km/s (Elliptical)', '11.5 km/s (Escape)']
    
    # Simulate all trajectories
    trajectories = []
    for v0 in velocities:
        traj = PayloadTrajectory(h0, v0)
        traj.simulate(t_max=10000)
        trajectories.append(traj)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Earth
    earth = Circle((0, 0), R_EARTH/1e6, color='blue', alpha=0.7)
    ax.add_patch(earth)
    
    # Initialize lines and points
    lines = []
    points = []
    for i in range(len(velocities)):
        line, = ax.plot([], [], color=colors[i], linewidth=2, alpha=0.7, label=labels[i])
        point, = ax.plot([], [], 'o', color=colors[i], markersize=8)
        lines.append(line)
        points.append(point)
    
    # Set limits
    ax.set_xlim(-15000, 15000)
    ax.set_ylim(-15000, 15000)
    ax.set_xlabel('X (km)', fontsize=12)
    ax.set_ylabel('Y (km)', fontsize=12)
    ax.set_title('Animated Payload Trajectories', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.legend(loc='upper right')
    
    # Time text
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, verticalalignment='top')
    
    def init():
        for line in lines:
            line.set_data([], [])
        for point in points:
            point.set_data([], [])
        time_text.set_text('')
        return lines + points + [time_text]
    
    def animate(frame):
        t = frame * 50  # Time step
        
        for i, traj in enumerate(trajectories):
            # Find index for current time
            idx = np.searchsorted(traj.time, t)
            if idx >= len(traj.time):
                idx = len(traj.time) - 1
            
            # Update line
            x_km = traj.position[:idx, 0] / 1e3
            y_km = traj.position[:idx, 1] / 1e3
            lines[i].set_data(x_km, y_km)
            
            # Update point
            if idx > 0:
                points[i].set_data([x_km[-1]], [y_km[-1]])  # Use single-element arrays
            else:
                points[i].set_data([], [])
        
        time_text.set_text(f'Time: {t:.0f} s')
        return lines + points + [time_text]
    
    # Create animation
    anim = FuncAnimation(fig, animate, init_func=init, frames=200, 
                        interval=50, blit=True, repeat=True)
    
    # Save animation
    writer = PillowWriter(fps=20)
    anim.save('../figures/animated_trajectory.gif', writer=writer)
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