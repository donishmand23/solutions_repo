#!/usr/bin/env python3
"""
Problem 1: Simulating the Effects of the Lorentz Force
Complete simulation and visualization script

This script generates all figures for the Lorentz force problem solution.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D
import os

# Create directories if they don't exist
os.makedirs('../figures', exist_ok=True)

# Physical constants
e = 1.602e-19  # Elementary charge (C)
m_e = 9.109e-31  # Electron mass (kg)
m_p = 1.673e-27  # Proton mass (kg)

class ChargedParticle:
    """Class representing a charged particle"""
    
    def __init__(self, charge, mass, position, velocity):
        """
        Initialize a charged particle
        
        Parameters:
        charge: particle charge (C)
        mass: particle mass (kg)
        position: initial position vector [x, y, z] (m)
        velocity: initial velocity vector [vx, vy, vz] (m/s)
        """
        self.q = charge
        self.m = mass
        self.r0 = np.array(position)
        self.v0 = np.array(velocity)
        
        # Storage for trajectory
        self.time = []
        self.position = []
        self.velocity = []
        self.energy = []
        
    def reset(self):
        """Reset particle to initial conditions"""
        self.time = []
        self.position = []
        self.velocity = []
        self.energy = []

class ElectromagneticField:
    """Class representing electromagnetic fields"""
    
    def __init__(self, E_field=None, B_field=None):
        """
        Initialize electromagnetic field
        
        Parameters:
        E_field: Electric field function E(r, t) or constant vector
        B_field: Magnetic field function B(r, t) or constant vector
        """
        self.E_field = E_field if E_field is not None else np.zeros(3)
        self.B_field = B_field if B_field is not None else np.zeros(3)
    
    def E(self, r, t):
        """Get electric field at position r and time t"""
        if callable(self.E_field):
            return self.E_field(r, t)
        else:
            return self.E_field
    
    def B(self, r, t):
        """Get magnetic field at position r and time t"""
        if callable(self.B_field):
            return self.B_field(r, t)
        else:
            return self.B_field

class LorentzForceSolver:
    """Solver for charged particle motion in EM fields"""
    
    def __init__(self, particle, field):
        """
        Initialize solver
        
        Parameters:
        particle: ChargedParticle object
        field: ElectromagneticField object
        """
        self.particle = particle
        self.field = field
        
    def lorentz_force(self, r, v, t):
        """Calculate Lorentz force on particle"""
        E = self.field.E(r, t)
        B = self.field.B(r, t)
        return self.particle.q * (E + np.cross(v, B))
    
    def derivatives(self, state, t):
        """Calculate derivatives for equations of motion"""
        r = state[:3]
        v = state[3:]
        
        F = self.lorentz_force(r, v, t)
        a = F / self.particle.m
        
        return np.concatenate([v, a])
    
    def rk4_step(self, state, t, dt):
        """Fourth-order Runge-Kutta integration step"""
        k1 = self.derivatives(state, t)
        k2 = self.derivatives(state + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = self.derivatives(state + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = self.derivatives(state + dt * k3, t + dt)
        
        return state + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
    
    def solve(self, t_max, dt=1e-9):
        """
        Solve particle motion
        
        Parameters:
        t_max: maximum simulation time (s)
        dt: time step (s)
        """
        # Reset particle
        self.particle.reset()
        
        # Initial state
        state = np.concatenate([self.particle.r0, self.particle.v0])
        
        # Time integration
        t = 0
        self.particle.time.append(t)
        self.particle.position.append(state[:3].copy())
        self.particle.velocity.append(state[3:].copy())
        
        # Initial energy
        v = state[3:]
        E_kinetic = 0.5 * self.particle.m * np.dot(v, v)
        self.particle.energy.append(E_kinetic)
        
        while t < t_max:
            # Adaptive timestep for accuracy
            v_mag = np.linalg.norm(state[3:])
            B_mag = np.linalg.norm(self.field.B(state[:3], t))
            
            if B_mag > 0 and self.particle.q != 0:
                # Ensure we resolve cyclotron motion
                omega_c = abs(self.particle.q * B_mag / self.particle.m)
                dt_max = 0.1 * 2 * np.pi / omega_c
                dt_use = min(dt, dt_max)
            else:
                dt_use = dt
            
            # Integration step
            state = self.rk4_step(state, t, dt_use)
            t += dt_use
            
            # Store data
            self.particle.time.append(t)
            self.particle.position.append(state[:3].copy())
            self.particle.velocity.append(state[3:].copy())
            
            # Calculate energy
            v = state[3:]
            E_kinetic = 0.5 * self.particle.m * np.dot(v, v)
            self.particle.energy.append(E_kinetic)
        
        # Convert to arrays
        self.particle.time = np.array(self.particle.time)
        self.particle.position = np.array(self.particle.position)
        self.particle.velocity = np.array(self.particle.velocity)
        self.particle.energy = np.array(self.particle.energy)

# Visualization functions

def plot_uniform_magnetic_field():
    """Plot trajectories in uniform magnetic field"""
    fig = plt.figure(figsize=(12, 10))
    
    # Create 2x2 subplot layout
    ax1 = fig.add_subplot(221, projection='3d')
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)
    
    # Uniform magnetic field in z-direction
    B0 = 0.01  # Tesla
    field = ElectromagneticField(B_field=np.array([0, 0, B0]))
    
    # Different initial conditions
    velocities = [
        [1e6, 0, 0],      # Pure x velocity
        [1e6, 0, 1e6],    # x and z velocity (helical)
        [0, 1e6, 0.5e6],  # y and z velocity (helical)
    ]
    
    colors = ['blue', 'red', 'green']
    labels = ['v₀ = (v,0,0)', 'v₀ = (v,0,v)', 'v₀ = (0,v,0.5v)']
    
    for i, v0 in enumerate(velocities):
        # Create electron
        electron = ChargedParticle(-e, m_e, [0, 0, 0], v0)
        solver = LorentzForceSolver(electron, field)
        
        # Solve for multiple cyclotron periods
        omega_c = abs(electron.q * B0 / electron.m)
        T_c = 2 * np.pi / omega_c
        solver.solve(t_max=5*T_c, dt=T_c/100)
        
        # Convert to micrometers for better visualization
        pos_um = electron.position * 1e6
        
        # 3D trajectory
        ax1.plot(pos_um[:, 0], pos_um[:, 1], pos_um[:, 2], 
                color=colors[i], label=labels[i], linewidth=2)
        
        # XY projection (circular motion)
        ax2.plot(pos_um[:, 0], pos_um[:, 1], color=colors[i], linewidth=2)
        
        # XZ projection
        ax3.plot(pos_um[:, 0], pos_um[:, 2], color=colors[i], linewidth=2)
        
        # Calculate and plot Larmor radius
        if i == 0:  # Only for circular motion
            r_L_theory = electron.m * np.linalg.norm(v0) / (abs(electron.q) * B0)
            r_L_measured = np.max(np.sqrt(pos_um[:, 0]**2 + pos_um[:, 1]**2))
            ax4.text(0.1, 0.9-i*0.1, f'Larmor radius (theory): {r_L_theory*1e6:.1f} μm',
                    transform=ax4.transAxes)
            ax4.text(0.1, 0.85-i*0.1, f'Larmor radius (measured): {r_L_measured:.1f} μm',
                    transform=ax4.transAxes)
    
    # Time evolution of energy
    ax4.plot(electron.time*1e9, electron.energy/e, 'k-', linewidth=2)
    ax4.set_xlabel('Time (ns)')
    ax4.set_ylabel('Kinetic Energy (eV)')
    ax4.set_title('Energy Conservation')
    ax4.grid(True, alpha=0.3)
    
    # Formatting
    ax1.set_xlabel('X (μm)')
    ax1.set_ylabel('Y (μm)')
    ax1.set_zlabel('Z (μm)')
    ax1.set_title('3D Trajectories in Uniform B-field')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('X (μm)')
    ax2.set_ylabel('Y (μm)')
    ax2.set_title('XY Projection (Circular Motion)')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    ax3.set_xlabel('X (μm)')
    ax3.set_ylabel('Z (μm)')
    ax3.set_title('XZ Projection (Helical Motion)')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../figures/uniform_magnetic_field.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_combined_fields():
    """Plot trajectories in combined E and B fields"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Fields: E in x-direction, B in z-direction
    E0 = 1000  # V/m
    B0 = 0.01  # Tesla
    field = ElectromagneticField(
        E_field=np.array([E0, 0, 0]),
        B_field=np.array([0, 0, B0])
    )
    
    # Calculate drift velocity
    v_drift = E0 / B0  # E×B drift in y-direction
    
    # Different particles
    particles = [
        ('Electron', -e, m_e, 'blue'),
        ('Proton', e, m_p, 'red')
    ]
    
    for name, charge, mass, color in particles:
        particle = ChargedParticle(charge, mass, [0, 0, 0], [0, 0, 0])
        solver = LorentzForceSolver(particle, field)
        
        # Solve for multiple cyclotron periods
        omega_c = abs(charge * B0 / mass)
        T_c = 2 * np.pi / omega_c
        solver.solve(t_max=10*T_c, dt=T_c/100)
        
        # Convert to appropriate units
        if name == 'Electron':
            pos = particle.position * 1e6  # micrometers
            unit = 'μm'
        else:
            pos = particle.position * 1e3  # millimeters
            unit = 'mm'
        
        # XY trajectory (cycloidal motion)
        axes[0, 0].plot(pos[:, 0], pos[:, 1], color=color, label=name, linewidth=2)
        
        # Y position vs time (drift motion)
        axes[0, 1].plot(particle.time*1e6, pos[:, 1], color=color, linewidth=2)
        
        # Velocity components
        axes[1, 0].plot(particle.time*1e6, particle.velocity[:, 0]/1e3, 
                       color=color, linestyle='-', alpha=0.7, label=f'{name} vₓ')
        axes[1, 0].plot(particle.time*1e6, particle.velocity[:, 1]/1e3, 
                       color=color, linestyle='--', alpha=0.7, label=f'{name} vᵧ')
        
        # Energy vs time
        axes[1, 1].plot(particle.time*1e6, particle.energy/e, color=color, linewidth=2)
    
    # Add drift velocity line
    axes[0, 1].axhline(y=0, color='k', linestyle=':', alpha=0.5)
    t_line = np.linspace(0, particle.time[-1]*1e6, 100)
    axes[0, 1].plot(t_line, v_drift*t_line*1e-6, 'k--', 
                   label=f'E×B drift: {v_drift/1e3:.1f} km/s')
    
    # Formatting
    axes[0, 0].set_xlabel(f'X ({unit})')
    axes[0, 0].set_ylabel(f'Y ({unit})')
    axes[0, 0].set_title('Cycloidal Trajectories (E⊥B)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_xlabel('Time (μs)')
    axes[0, 1].set_ylabel(f'Y Position ({unit})')
    axes[0, 1].set_title('Drift Motion')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_xlabel('Time (μs)')
    axes[1, 0].set_ylabel('Velocity (km/s)')
    axes[1, 0].set_title('Velocity Components')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_xlabel('Time (μs)')
    axes[1, 1].set_ylabel('Kinetic Energy (eV)')
    axes[1, 1].set_title('Energy Evolution')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../figures/combined_fields.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_crossed_fields():
    """Plot trajectories in crossed E and B fields"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Fields: E in x-direction, B in y-direction
    E0 = 1000  # V/m
    B0 = 0.01  # Tesla
    field = ElectromagneticField(
        E_field=np.array([E0, 0, 0]),
        B_field=np.array([0, B0, 0])
    )
    
    # E×B drift is in z-direction
    v_drift = E0 / B0
    
    # Different initial velocities
    initial_velocities = [
        [0, 0, 0],           # Start at rest
        [5e5, 0, 0],         # Initial x velocity
        [0, 0, 5e5],         # Initial z velocity
        [5e5, 0, 5e5],       # Both x and z
    ]
    
    colors = ['blue', 'red', 'green', 'purple']
    labels = ['v₀ = 0', 'v₀ = (v,0,0)', 'v₀ = (0,0,v)', 'v₀ = (v,0,v)']
    
    for i, (v0, color, label) in enumerate(zip(initial_velocities, colors, labels)):
        electron = ChargedParticle(-e, m_e, [0, 0, 0], v0)
        solver = LorentzForceSolver(electron, field)
        solver.solve(t_max=1e-7, dt=1e-10)
        
        # Convert to micrometers
        pos = electron.position * 1e6
        
        # 3D trajectory
        ax_3d = fig.add_subplot(2, 2, 1, projection='3d')
        ax_3d.plot(pos[:, 0], pos[:, 1], pos[:, 2], color=color, 
                  label=label, linewidth=2, alpha=0.7)
        
        # XZ projection (drift plane)
        axes[0, 1].plot(pos[:, 0], pos[:, 2], color=color, linewidth=2)
        
        # Z position vs time (drift motion)
        axes[1, 0].plot(electron.time*1e9, pos[:, 2], color=color, linewidth=2)
        
        # Velocity magnitude
        v_mag = np.linalg.norm(electron.velocity, axis=1)
        axes[1, 1].plot(electron.time*1e9, v_mag/1e3, color=color, linewidth=2)
    
    # Add theoretical drift line
    t_line = np.linspace(0, electron.time[-1]*1e9, 100)
    axes[1, 0].plot(t_line, v_drift*t_line*1e-3, 'k--', 
                   label=f'E×B drift: {v_drift/1e3:.0f} km/s', linewidth=2)
    
    # Formatting
    ax_3d.set_xlabel('X (μm)')
    ax_3d.set_ylabel('Y (μm)')
    ax_3d.set_zlabel('Z (μm)')
    ax_3d.set_title('3D Trajectories in Crossed Fields')
    ax_3d.legend()
    ax_3d.grid(True, alpha=0.3)
    
    axes[0, 1].set_xlabel('X (μm)')
    axes[0, 1].set_ylabel('Z (μm)')
    axes[0, 1].set_title('XZ Projection (Drift Plane)')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_xlabel('Time (ns)')
    axes[1, 0].set_ylabel('Z Position (μm)')
    axes[1, 0].set_title('Drift Motion in Z-direction')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_xlabel('Time (ns)')
    axes[1, 1].set_ylabel('Speed (km/s)')
    axes[1, 1].set_title('Particle Speed')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../figures/crossed_fields.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_parameter_study():
    """Study effects of various parameters on trajectories"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Base parameters
    B0 = 0.01  # Tesla
    v0 = 1e6   # m/s
    
    # Study 1: Effect of charge-to-mass ratio
    ax = axes[0, 0]
    field = ElectromagneticField(B_field=np.array([0, 0, B0]))
    
    particles = [
        ('Electron', -e, m_e, 'blue'),
        ('Proton', e, m_p, 'red'),
        ('Alpha', 2*e, 4*m_p, 'green'),
    ]
    
    for name, charge, mass, color in particles:
        particle = ChargedParticle(charge, mass, [0, 0, 0], [v0, 0, 0])
        solver = LorentzForceSolver(particle, field)
        
        omega_c = abs(charge * B0 / mass)
        T_c = 2 * np.pi / omega_c
        solver.solve(t_max=2*T_c, dt=T_c/100)
        
        pos = particle.position * 1e6
        ax.plot(pos[:, 0], pos[:, 1], color=color, 
               label=f'{name} (q/m={charge/mass:.1e} C/kg)', linewidth=2)
    
    ax.set_xlabel('X (μm)')
    ax.set_ylabel('Y (μm)')
    ax.set_title('Effect of q/m Ratio')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Study 2: Effect of initial velocity magnitude
    ax = axes[0, 1]
    velocities = [0.5e6, 1e6, 2e6]
    colors = plt.cm.viridis(np.linspace(0, 1, len(velocities)))
    
    for v, color in zip(velocities, colors):
        electron = ChargedParticle(-e, m_e, [0, 0, 0], [v, 0, 0])
        solver = LorentzForceSolver(electron, field)
        
        omega_c = abs(electron.q * B0 / electron.m)
        T_c = 2 * np.pi / omega_c
        solver.solve(t_max=T_c, dt=T_c/100)
        
        pos = electron.position * 1e6
        r_L = electron.m * v / (abs(electron.q) * B0) * 1e6
        ax.plot(pos[:, 0], pos[:, 1], color=color, 
               label=f'v₀={v/1e6:.1f} Mm/s, r_L={r_L:.1f} μm', linewidth=2)
    
    ax.set_xlabel('X (μm)')
    ax.set_ylabel('Y (μm)')
    ax.set_title('Effect of Initial Velocity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Study 3: Effect of magnetic field strength
    ax = axes[0, 2]
    B_values = [0.005, 0.01, 0.02]
    colors = plt.cm.plasma(np.linspace(0, 1, len(B_values)))
    
    for B, color in zip(B_values, colors):
        field = ElectromagneticField(B_field=np.array([0, 0, B]))
        electron = ChargedParticle(-e, m_e, [0, 0, 0], [v0, 0, 0])
        solver = LorentzForceSolver(electron, field)
        
        omega_c = abs(electron.q * B / electron.m)
        T_c = 2 * np.pi / omega_c
        solver.solve(t_max=T_c, dt=T_c/100)
        
        pos = electron.position * 1e6
        r_L = electron.m * v0 / (abs(electron.q) * B) * 1e6
        ax.plot(pos[:, 0], pos[:, 1], color=color, 
               label=f'B={B*1e3:.0f} mT, r_L={r_L:.1f} μm', linewidth=2)
    
    ax.set_xlabel('X (μm)')
    ax.set_ylabel('Y (μm)')
    ax.set_title('Effect of Magnetic Field Strength')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Study 4: Pitch angle effect (helical motion)
    ax = axes[1, 0]
    field = ElectromagneticField(B_field=np.array([0, 0, B0]))
    pitch_angles = [0, 30, 60, 90]
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(pitch_angles)))
    
    ax_3d = fig.add_subplot(2, 3, 4, projection='3d')
    
    for angle, color in zip(pitch_angles, colors):
        angle_rad = np.radians(angle)
        v_perp = v0 * np.sin(angle_rad)
        v_para = v0 * np.cos(angle_rad)
        
        electron = ChargedParticle(-e, m_e, [0, 0, 0], [v_perp, 0, v_para])
        solver = LorentzForceSolver(electron, field)
        
        omega_c = abs(electron.q * B0 / electron.m)
        T_c = 2 * np.pi / omega_c
        solver.solve(t_max=3*T_c, dt=T_c/100)
        
        pos = electron.position * 1e6
        ax_3d.plot(pos[:, 0], pos[:, 1], pos[:, 2], color=color, 
                  label=f'α={angle}°', linewidth=2, alpha=0.7)
    
    ax_3d.set_xlabel('X (μm)')
    ax_3d.set_ylabel('Y (μm)')
    ax_3d.set_zlabel('Z (μm)')
    ax_3d.set_title('Effect of Pitch Angle')
    ax_3d.legend()
    ax_3d.grid(True, alpha=0.3)
    
    # Study 5: Combined E and B field strengths
    ax = axes[1, 1]
    E_values = [0, 500, 1000, 2000]
    colors = plt.cm.copper(np.linspace(0, 1, len(E_values)))
    
    for E, color in zip(E_values, colors):
        field = ElectromagneticField(
            E_field=np.array([E, 0, 0]),
            B_field=np.array([0, 0, B0])
        )
        electron = ChargedParticle(-e, m_e, [0, 0, 0], [0, 0, 0])
        solver = LorentzForceSolver(electron, field)
        
        omega_c = abs(electron.q * B0 / electron.m)
        T_c = 2 * np.pi / omega_c
        solver.solve(t_max=5*T_c, dt=T_c/100)
        
        pos = electron.position * 1e6
        v_drift = E / B0 if E > 0 else 0
        ax.plot(pos[:, 0], pos[:, 1], color=color, 
               label=f'E={E} V/m, v_d={v_drift/1e3:.0f} km/s', linewidth=2)
    
    ax.set_xlabel('X (μm)')
    ax.set_ylabel('Y (μm)')
    ax.set_title('Effect of Electric Field (E⊥B)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Study 6: Time evolution comparison
    ax = axes[1, 2]
    field = ElectromagneticField(B_field=np.array([0, 0, B0]))
    
    for name, charge, mass, color in particles[:2]:  # Just electron and proton
        particle = ChargedParticle(charge, mass, [0, 0, 0], [v0, 0, 0])
        solver = LorentzForceSolver(particle, field)
        
        omega_c = abs(charge * B0 / mass)
        T_c = 2 * np.pi / omega_c
        solver.solve(t_max=5*T_c, dt=T_c/100)
        
        # Plot phase space (x vs vx)
        pos = particle.position * 1e6
        vel = particle.velocity / 1e3
        ax.plot(pos[:, 0], vel[:, 0], color=color, label=name, linewidth=2)
    
    ax.set_xlabel('X Position (μm)')
    ax.set_ylabel('X Velocity (km/s)')
    ax.set_title('Phase Space Portrait')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../figures/parameter_study.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_3d_trajectories():
    """Create comprehensive 3D visualization"""
    fig = plt.figure(figsize=(15, 10))
    
    # Create different field configurations
    configs = [
        ('Uniform B', np.array([0, 0, 0]), np.array([0, 0, 0.01])),
        ('E and B Parallel', np.array([1000, 0, 0]), np.array([0.01, 0, 0])),
        ('E and B Perpendicular', np.array([1000, 0, 0]), np.array([0, 0, 0.01])),
        ('E and B Crossed', np.array([1000, 0, 0]), np.array([0, 0.01, 0])),
    ]
    
    for i, (title, E, B) in enumerate(configs):
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        
        field = ElectromagneticField(E_field=E, B_field=B)
        
        # Multiple particles with different initial conditions
        initial_conditions = [
            ([0, 0, 0], [1e6, 0, 0], 'blue'),
            ([0, 0, 0], [0, 1e6, 0], 'red'),
            ([0, 0, 0], [0.7e6, 0.7e6, 0], 'green'),
        ]
        
        for r0, v0, color in initial_conditions:
            electron = ChargedParticle(-e, m_e, r0, v0)
            solver = LorentzForceSolver(electron, field)
            
            # Determine appropriate time scale
            B_mag = np.linalg.norm(B)
            if B_mag > 0:
                omega_c = abs(electron.q * B_mag / electron.m)
                T_c = 2 * np.pi / omega_c
                t_max = 5 * T_c
            else:
                t_max = 1e-7
            
            solver.solve(t_max=t_max, dt=t_max/1000)
            
            # Convert to micrometers
            pos = electron.position * 1e6
            
            # Plot trajectory
            ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], color=color, 
                   linewidth=2, alpha=0.7)
            
            # Mark starting point
            ax.scatter(pos[0, 0], pos[0, 1], pos[0, 2], color=color, s=50, marker='o')
        
        # Add field vectors
        ax.quiver(0, 0, 0, E[0]/1000, E[1]/1000, E[2]/1000, 
                 color='orange', arrow_length_ratio=0.2, linewidth=3, label='E field')
        ax.quiver(0, 0, 0, B[0]*100, B[1]*100, B[2]*100, 
                 color='purple', arrow_length_ratio=0.2, linewidth=3, label='B field')
        
        ax.set_xlabel('X (μm)')
        ax.set_ylabel('Y (μm)')
        ax.set_zlabel('Z (μm)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../figures/trajectory_3d.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_larmor_radius_analysis():
    """Analyze and verify Larmor radius formula"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Study 1: Larmor radius vs velocity
    ax = axes[0, 0]
    B0 = 0.01  # Tesla
    field = ElectromagneticField(B_field=np.array([0, 0, B0]))
    
    velocities = np.linspace(1e5, 2e6, 20)
    r_L_theory = []
    r_L_measured = []
    
    for v in velocities:
        electron = ChargedParticle(-e, m_e, [0, 0, 0], [v, 0, 0])
        solver = LorentzForceSolver(electron, field)
        
        omega_c = abs(electron.q * B0 / electron.m)
        T_c = 2 * np.pi / omega_c
        solver.solve(t_max=T_c, dt=T_c/100)
        
        # Theoretical Larmor radius
        r_L_th = electron.m * v / (abs(electron.q) * B0)
        r_L_theory.append(r_L_th * 1e6)
        
        # Measured Larmor radius
        pos = electron.position * 1e6
        r_measured = np.max(np.sqrt(pos[:, 0]**2 + pos[:, 1]**2))
        r_L_measured.append(r_measured)
    
    ax.plot(velocities/1e6, r_L_theory, 'b-', linewidth=2, label='Theory: r_L = mv/(qB)')
    ax.scatter(velocities/1e6, r_L_measured, color='red', s=30, alpha=0.7, label='Measured')
    ax.set_xlabel('Velocity (Mm/s)')
    ax.set_ylabel('Larmor Radius (μm)')
    ax.set_title('Larmor Radius vs Velocity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Study 2: Larmor radius vs B field
    ax = axes[0, 1]
    v0 = 1e6  # m/s
    B_values = np.linspace(0.005, 0.02, 20)
    r_L_theory = []
    r_L_measured = []
    
    for B in B_values:
        field = ElectromagneticField(B_field=np.array([0, 0, B]))
        electron = ChargedParticle(-e, m_e, [0, 0, 0], [v0, 0, 0])
        solver = LorentzForceSolver(electron, field)
        
        omega_c = abs(electron.q * B / electron.m)
        T_c = 2 * np.pi / omega_c
        solver.solve(t_max=T_c, dt=T_c/100)
        
        # Theoretical Larmor radius
        r_L_th = electron.m * v0 / (abs(electron.q) * B)
        r_L_theory.append(r_L_th * 1e6)
        
        # Measured Larmor radius
        pos = electron.position * 1e6
        r_measured = np.max(np.sqrt(pos[:, 0]**2 + pos[:, 1]**2))
        r_L_measured.append(r_measured)
    
    ax.plot(B_values*1000, r_L_theory, 'b-', linewidth=2, label='Theory')
    ax.scatter(B_values*1000, r_L_measured, color='red', s=30, alpha=0.7, label='Measured')
    ax.set_xlabel('Magnetic Field (mT)')
    ax.set_ylabel('Larmor Radius (μm)')
    ax.set_title('Larmor Radius vs B Field')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Study 3: Cyclotron frequency verification
    ax = axes[1, 0]
    B0 = 0.01  # Tesla
    field = ElectromagneticField(B_field=np.array([0, 0, B0]))
    
    particles = [
        ('Electron', -e, m_e),
        ('Proton', e, m_p),
        ('Deuteron', e, 2*m_p),
        ('Alpha', 2*e, 4*m_p),
    ]
    
    names = []
    omega_c_theory = []
    omega_c_measured = []
    
    for name, charge, mass in particles:
        particle = ChargedParticle(charge, mass, [0, 0, 0], [1e6, 0, 0])
        solver = LorentzForceSolver(particle, field)
        
        # Theoretical cyclotron frequency
        omega_c_th = abs(charge * B0 / mass)
        omega_c_theory.append(omega_c_th / (2*np.pi) / 1e6)  # MHz
        
        # Measure frequency from trajectory
        T_c = 2 * np.pi / omega_c_th
        solver.solve(t_max=10*T_c, dt=T_c/200)
        
        # Find period from x-position zero crossings
        x_pos = particle.position[:, 0]
        zero_crossings = np.where(np.diff(np.sign(x_pos)))[0]
        if len(zero_crossings) > 2:
            periods = np.diff(particle.time[zero_crossings[::2]])
            T_measured = np.mean(periods)
            omega_c_meas = 2 * np.pi / T_measured
            omega_c_measured.append(omega_c_meas / (2*np.pi) / 1e6)  # MHz
        else:
            omega_c_measured.append(0)
        
        names.append(name)
    
    x_pos = np.arange(len(names))
    width = 0.35
    
    ax.bar(x_pos - width/2, omega_c_theory, width, label='Theory', color='blue', alpha=0.7)
    ax.bar(x_pos + width/2, omega_c_measured, width, label='Measured', color='red', alpha=0.7)
    ax.set_xlabel('Particle Type')
    ax.set_ylabel('Cyclotron Frequency (MHz)')
    ax.set_title('Cyclotron Frequency Verification')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Study 4: Phase space conservation
    ax = axes[1, 1]
    field = ElectromagneticField(B_field=np.array([0, 0, B0]))
    electron = ChargedParticle(-e, m_e, [0, 0, 0], [1e6, 0, 0])
    solver = LorentzForceSolver(electron, field)
    
    omega_c = abs(electron.q * B0 / electron.m)
    T_c = 2 * np.pi / omega_c
    solver.solve(t_max=3*T_c, dt=T_c/100)
    
    # Plot phase space
    pos = electron.position * 1e6
    vel = electron.velocity / 1e3
    
    ax.plot(pos[:, 0], vel[:, 0], 'b-', linewidth=2, label='x vs vₓ')
    ax.plot(pos[:, 1], vel[:, 1], 'r-', linewidth=2, label='y vs vᵧ')
    ax.set_xlabel('Position (μm)')
    ax.set_ylabel('Velocity (km/s)')
    ax.set_title('Phase Space Portrait')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../figures/larmor_radius_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_drift_velocity():
    """Analyze drift velocity in various field configurations"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # E×B drift
    ax = axes[0, 0]
    E0 = 1000  # V/m
    B0 = 0.01  # Tesla
    
    # Different angles between E and B
    angles = np.linspace(0, 180, 10)
    v_drift_theory = []
    v_drift_measured = []
    
    for angle in angles:
        angle_rad = np.radians(angle)
        E = np.array([E0, 0, 0])
        B = np.array([B0*np.cos(angle_rad), B0*np.sin(angle_rad), 0])
        
        if np.linalg.norm(B) > 0:
            # Theoretical drift velocity
            v_d_theory = np.cross(E, B) / np.dot(B, B)
            v_drift_theory.append(np.linalg.norm(v_d_theory))
            
            # Simulate
            field = ElectromagneticField(E_field=E, B_field=B)
            electron = ChargedParticle(-e, m_e, [0, 0, 0], [0, 0, 0])
            solver = LorentzForceSolver(electron, field)
            
            solver.solve(t_max=1e-6, dt=1e-9)
            
            # Measure drift from trajectory
            if len(electron.position) > 100:
                # Linear fit to find drift velocity
                t_fit = electron.time[50:]
                pos_fit = electron.position[50:]
                
                # Fit each component
                drift_components = []
                for i in range(3):
                    if np.std(pos_fit[:, i]) > 1e-10:
                        p = np.polyfit(t_fit, pos_fit[:, i], 1)
                        drift_components.append(p[0])
                    else:
                        drift_components.append(0)
                
                v_drift_measured.append(np.linalg.norm(drift_components))
            else:
                v_drift_measured.append(0)
        else:
            v_drift_theory.append(0)
            v_drift_measured.append(0)
    
    ax.plot(angles, np.array(v_drift_theory)/1e3, 'b-', linewidth=2, label='Theory')
    ax.scatter(angles, np.array(v_drift_measured)/1e3, color='red', s=30, alpha=0.7, label='Measured')
    ax.set_xlabel('Angle between E and B (degrees)')
    ax.set_ylabel('Drift Velocity (km/s)')
    ax.set_title('E×B Drift vs Field Angle')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Drift in gradient B field
    ax = axes[0, 1]
    # Simplified gradient: B increases linearly in x direction
    def B_gradient(r, t):
        x, y, z = r
        B_z = B0 * (1 + x/0.01)  # 1% per cm gradient
        return np.array([0, 0, B_z])
    
    field = ElectromagneticField(B_field=B_gradient)
    
    # Different particle energies
    energies_eV = [100, 500, 1000, 2000]
    colors = plt.cm.viridis(np.linspace(0, 1, len(energies_eV)))
    
    for E_eV, color in zip(energies_eV, colors):
        # Convert energy to velocity
        E_J = E_eV * e
        v = np.sqrt(2 * E_J / m_e)
        
        electron = ChargedParticle(-e, m_e, [0, 0, 0], [v, 0, 0])
        solver = LorentzForceSolver(electron, field)
        
        omega_c = abs(electron.q * B0 / electron.m)
        T_c = 2 * np.pi / omega_c
        solver.solve(t_max=20*T_c, dt=T_c/100)
        
        pos = electron.position * 1e6
        ax.plot(pos[:, 0], pos[:, 1], color=color, linewidth=2, 
               label=f'{E_eV} eV', alpha=0.7)
    
    ax.set_xlabel('X (μm)')
    ax.set_ylabel('Y (μm)')
    ax.set_title('Gradient B Drift')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Curvature drift
    ax = axes[1, 0]
    # Implement simple curved field lines
    def B_curved(r, t):
        x, y, z = r
        # Simplified toroidal-like field
        R = 0.1  # 10 cm major radius
        r_cyl = np.sqrt(x**2 + y**2)
        if r_cyl > 0:
            B_phi = B0 * R / r_cyl
            B_x = -B_phi * y / r_cyl
            B_y = B_phi * x / r_cyl
            B_z = B0 * 0.1  # Small vertical component
            return np.array([B_x, B_y, B_z])
        else:
            return np.array([0, 0, B0])
    
    field = ElectromagneticField(B_field=B_curved)
    
    # Initial positions around the torus
    angles = [0, 90, 180, 270]
    colors = ['blue', 'red', 'green', 'purple']
    
    for angle, color in zip(angles, colors):
        angle_rad = np.radians(angle)
        r_start = 0.05  # 5 cm
        x0 = r_start * np.cos(angle_rad)
        y0 = r_start * np.sin(angle_rad)
        
        # Velocity tangent to field line
        v_mag = 1e6
        electron = ChargedParticle(-e, m_e, [x0, y0, 0], [0, 0, v_mag])
        solver = LorentzForceSolver(electron, field)
        solver.solve(t_max=1e-6, dt=1e-10)
        
        pos = electron.position * 1e3  # Convert to mm
        ax.plot(pos[:, 0], pos[:, 1], color=color, linewidth=2, 
               label=f'Start: {angle}°', alpha=0.7)
    
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_title('Curvature Drift in Curved B Field')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Combined drifts
    ax = axes[1, 1]
    # E×B + gradient B
    E = np.array([500, 0, 0])
    field = ElectromagneticField(E_field=E, B_field=B_gradient)
    
    electron = ChargedParticle(-e, m_e, [0, 0, 0], [1e6, 0, 0])
    solver = LorentzForceSolver(electron, field)
    
    omega_c = abs(electron.q * B0 / electron.m)
    T_c = 2 * np.pi / omega_c
    solver.solve(t_max=50*T_c, dt=T_c/100)
    
    pos = electron.position * 1e6
    
    # Plot trajectory
    ax.plot(pos[:, 0], pos[:, 1], 'b-', linewidth=2, alpha=0.7, label='Trajectory')
    
    # Calculate and plot average drift
    if len(pos) > 1000:
        # Average position over time
        window = len(pos) // 10
        avg_x = np.convolve(pos[:, 0], np.ones(window)/window, mode='valid')
        avg_y = np.convolve(pos[:, 1], np.ones(window)/window, mode='valid')
        time_avg = electron.time[:len(avg_x)]
        
        ax.plot(avg_x, avg_y, 'r--', linewidth=3, label='Average drift')
    
    ax.set_xlabel('X (μm)')
    ax.set_ylabel('Y (μm)')
    ax.set_title('Combined E×B and Gradient B Drift')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../figures/drift_velocity.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_animation():
    """Create animated visualization of particle motion"""
    # Set up the figure
    fig = plt.figure(figsize=(12, 8))
    
    # Create subplots
    ax1 = fig.add_subplot(221, projection='3d')
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)
    
    # Field configuration: E and B perpendicular
    E0 = 1000  # V/m
    B0 = 0.01  # Tesla
    field = ElectromagneticField(
        E_field=np.array([E0, 0, 0]),
        B_field=np.array([0, 0, B0])
    )
    
    # Create particles
    particles = [
        ChargedParticle(-e, m_e, [0, 0, 0], [0, 0, 0]),  # Electron
        ChargedParticle(e, m_p, [0, 0, 0], [0, 0, 0]),   # Proton
    ]
    
    solvers = [LorentzForceSolver(p, field) for p in particles]
    
    # Solve for all particles
    t_max = 1e-6
    for solver in solvers:
        solver.solve(t_max=t_max, dt=1e-10)
    
    # Initialize plot elements
    lines_3d = []
    points_3d = []
    lines_xy = []
    points_xy = []
    lines_energy = []
    lines_velocity = []
    
    colors = ['blue', 'red']
    labels = ['Electron', 'Proton']
    
    for i, (particle, color, label) in enumerate(zip(particles, colors, labels)):
        # 3D trajectory
        line_3d, = ax1.plot([], [], [], color=color, linewidth=2, label=label)
        point_3d, = ax1.plot([], [], [], 'o', color=color, markersize=8)
        lines_3d.append(line_3d)
        points_3d.append(point_3d)
        
        # XY projection
        line_xy, = ax2.plot([], [], color=color, linewidth=2)
        point_xy, = ax2.plot([], [], 'o', color=color, markersize=8)
        lines_xy.append(line_xy)
        points_xy.append(point_xy)
        
        # Energy
        line_energy, = ax3.plot([], [], color=color, linewidth=2)
        lines_energy.append(line_energy)
        
        # Velocity components
        line_vx, = ax4.plot([], [], color=color, linestyle='-', linewidth=2, 
                           label=f'{label} vₓ')
        line_vy, = ax4.plot([], [], color=color, linestyle='--', linewidth=2,
                           label=f'{label} vᵧ')
        lines_velocity.extend([line_vx, line_vy])
    
    # Set up axes
    ax1.set_xlabel('X (μm)')
    ax1.set_ylabel('Y (μm)')
    ax1.set_zlabel('Z (μm)')
    ax1.set_title('3D Trajectories')
    ax1.legend()
    
    # Set axis limits based on particle trajectories
    all_pos = np.concatenate([p.position for p in particles])
    pos_um = all_pos * 1e6
    margin = 0.1
    x_range = [pos_um[:, 0].min() - margin, pos_um[:, 0].max() + margin]
    y_range = [pos_um[:, 1].min() - margin, pos_um[:, 1].max() + margin]
    z_range = [pos_um[:, 2].min() - margin, pos_um[:, 2].max() + margin]
    
    ax1.set_xlim(x_range)
    ax1.set_ylim(y_range)
    ax1.set_zlim(z_range)
    
    ax2.set_xlim(x_range)
    ax2.set_ylim(y_range)
    ax2.set_xlabel('X (μm)')
    ax2.set_ylabel('Y (μm)')
    ax2.set_title('XY Projection')
    ax2.grid(True, alpha=0.3)
    
    ax3.set_xlim(0, t_max*1e6)
    ax3.set_ylim(0, max([p.energy.max()/e for p in particles]) * 1.1)
    ax3.set_xlabel('Time (μs)')
    ax3.set_ylabel('Kinetic Energy (eV)')
    ax3.set_title('Energy Evolution')
    ax3.grid(True, alpha=0.3)
    
    ax4.set_xlim(0, t_max*1e6)
    all_vel = np.concatenate([p.velocity for p in particles])
    v_max = np.abs(all_vel).max() / 1e3 * 1.1
    ax4.set_ylim(-v_max, v_max)
    ax4.set_xlabel('Time (μs)')
    ax4.set_ylabel('Velocity (km/s)')
    ax4.set_title('Velocity Components')
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
    
    # Time text
    time_text = fig.text(0.5, 0.02, '', ha='center', fontsize=12)
    
    def init():
        for line in lines_3d + lines_xy + lines_energy + lines_velocity:
            line.set_data([], [])
        for line in lines_3d:
            line.set_3d_properties([])
        for point in points_3d + points_xy:
            point.set_data([], [])
        for point in points_3d:
            point.set_3d_properties([])
        time_text.set_text('')
        return lines_3d + points_3d + lines_xy + points_xy + lines_energy + lines_velocity + [time_text]
    
    def animate(frame):
        # Time index
        t_current = frame * t_max / 200
        
        for i, particle in enumerate(particles):
            # Find index for current time
            idx = np.searchsorted(particle.time, t_current)
            if idx >= len(particle.time):
                idx = len(particle.time) - 1
            
            # Get position in micrometers
            pos_um = particle.position[:idx] * 1e6
            
            if idx > 0:
                # Update 3D trajectory
                lines_3d[i].set_data(pos_um[:, 0], pos_um[:, 1])
                lines_3d[i].set_3d_properties(pos_um[:, 2])
                points_3d[i].set_data([pos_um[-1, 0]], [pos_um[-1, 1]])
                points_3d[i].set_3d_properties([pos_um[-1, 2]])
                
                # Update XY projection
                lines_xy[i].set_data(pos_um[:, 0], pos_um[:, 1])
                points_xy[i].set_data([pos_um[-1, 0]], [pos_um[-1, 1]])
                
                # Update energy
                lines_energy[i].set_data(particle.time[:idx]*1e6, 
                                       particle.energy[:idx]/e)
                
                # Update velocity components
                lines_velocity[2*i].set_data(particle.time[:idx]*1e6, 
                                           particle.velocity[:idx, 0]/1e3)
                lines_velocity[2*i+1].set_data(particle.time[:idx]*1e6, 
                                             particle.velocity[:idx, 1]/1e3)
        
        time_text.set_text(f'Time: {t_current*1e6:.2f} μs')
        
        return lines_3d + points_3d + lines_xy + points_xy + lines_energy + lines_velocity + [time_text]
    
    # Create animation
    anim = FuncAnimation(fig, animate, init_func=init, frames=201, 
                        interval=50, blit=True, repeat=True)
    
    # Save animation
    plt.tight_layout()
    writer = PillowWriter(fps=20)
    anim.save('../figures/particle_animation.gif', writer=writer)
    plt.close()

def main():
    """Generate all figures for the Lorentz force simulation"""
    print("Generating uniform magnetic field plot...")
    plot_uniform_magnetic_field()
    print("Done.")
    
    print("Generating combined fields plot...")
    plot_combined_fields()
    print("Done.")
    
    print("Generating crossed fields plot...")
    plot_crossed_fields()
    print("Done.")
    
    print("Generating parameter study plot...")
    plot_parameter_study()
    print("Done.")
    
    print("Generating 3D trajectories plot...")
    plot_3d_trajectories()
    print("Done.")
    
    print("Generating Larmor radius analysis plot...")
    plot_larmor_radius_analysis()
    print("Done.")
    
    print("Generating drift velocity plot...")
    plot_drift_velocity()
    print("Done.")
    
    print("Creating animation...")
    create_animation()
    print("Done.")
    
    print("\nAll figures have been generated successfully!")
    print("Check the '../figures' directory for the output files:")
    print("  - uniform_magnetic_field.png")
    print("  - combined_fields.png")
    print("  - crossed_fields.png")
    print("  - parameter_study.png")
    print("  - trajectory_3d.png")
    print("  - larmor_radius_analysis.png")
    print("  - drift_velocity.png")
    print("  - particle_animation.gif")

if __name__ == "__main__":
    main()