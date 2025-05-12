import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.integrate import solve_ivp
import os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Constants
G = 6.67430e-11  # Gravitational constant (m^3 kg^-1 s^-2)
M_EARTH = 5.972e24  # Mass of Earth (kg)
R_EARTH = 6.371e6  # Radius of Earth (m)
GM = G * M_EARTH  # Standard gravitational parameter for Earth

# Create output directory if it doesn't exist
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
figures_dir = os.path.join(parent_dir, 'figures')

if not os.path.exists(figures_dir):
    os.makedirs(figures_dir)

def simulate_trajectory(r0, v0, tmax=100000, dt=1000):
    """
    Simulate the trajectory of a payload released near Earth.
    
    Parameters:
    - r0: Initial position vector [x, y, z] in Earth radii
    - v0: Initial velocity vector [vx, vy, vz] in units of sqrt(GM/R_EARTH)
    - tmax: Maximum simulation time
    - dt: Time step for output
    
    Returns:
    - t: Time points
    - positions: Position vectors at each time point
    """
    # Convert to standard units
    r0_m = np.array(r0) * R_EARTH
    v0_m = np.array(v0) * np.sqrt(GM / R_EARTH)
    
    # Initial state vector [x, y, z, vx, vy, vz]
    state0 = np.concatenate([r0_m, v0_m])
    
    # Define the derivatives function for the equations of motion
    def derivatives(t, state):
        r = state[:3]
        v = state[3:]
        r_mag = np.linalg.norm(r)
        
        # Check if payload has crashed into Earth
        if r_mag < R_EARTH:
            return np.zeros(6)
        
        # Gravitational acceleration
        a = -GM * r / r_mag**3
        
        return np.concatenate([v, a])
    
    # Solve the differential equations
    sol = solve_ivp(
        derivatives,
        [0, tmax],
        state0,
        method='RK45',
        t_eval=np.arange(0, tmax, dt),
        rtol=1e-8,
        atol=1e-8
    )
    
    # Convert positions back to Earth radii
    positions = sol.y[:3, :].T / R_EARTH
    
    return sol.t, positions

def calculate_specific_energy(r, v):
    """
    Calculate the specific energy of an orbit.
    
    Parameters:
    - r: Position vector in Earth radii
    - v: Velocity vector in units of sqrt(GM/R_EARTH)
    
    Returns:
    - e: Specific energy
    """
    r_m = np.array(r) * R_EARTH
    v_m = np.array(v) * np.sqrt(GM / R_EARTH)
    
    r_mag = np.linalg.norm(r_m)
    v_mag = np.linalg.norm(v_m)
    
    # Specific energy = kinetic energy - potential energy
    e = 0.5 * v_mag**2 - GM / r_mag
    
    return e

def create_release_parameter_effects():
    """Create a visualization showing the effects of different release parameters on trajectory outcomes."""
    fig = plt.figure(figsize=(15, 10))
    
    # Create a 2x2 grid of subplots
    ax1 = fig.add_subplot(221, projection='3d')
    ax2 = fig.add_subplot(222, projection='3d')
    ax3 = fig.add_subplot(223, projection='3d')
    ax4 = fig.add_subplot(224)
    
    axes = [ax1, ax2, ax3]
    
    # Draw Earth in all 3D subplots
    for ax in axes:
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = np.cos(u) * np.sin(v)
        y = np.sin(u) * np.sin(v)
        z = np.cos(v)
        ax.plot_surface(x, y, z, color='blue', alpha=0.3)
        
        # Set axis properties
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.set_zlim(-5, 5)
        ax.set_xlabel('X (Earth radii)')
        ax.set_ylabel('Y (Earth radii)')
        ax.set_zlabel('Z (Earth radii)')
    
    # 1. Effect of release velocity magnitude (ax1)
    # Base position: 2 Earth radii from center, on x-axis
    r0 = [2.0, 0.0, 0.0]
    
    # Different velocity magnitudes (as fraction of circular velocity)
    v_circ = np.sqrt(1/2)  # Circular velocity at r=2 (in normalized units)
    
    velocity_factors = [0.8, 1.0, 1.2, 1.4]
    velocity_labels = ['0.8 × v_circ (Elliptical)', 'v_circ (Circular)', '1.2 × v_circ (Elliptical)', '1.4 × v_circ (Hyperbolic)']
    colors = ['blue', 'green', 'orange', 'red']
    
    for i, factor in enumerate(velocity_factors):
        v0 = [0.0, v_circ * factor, 0.0]  # Tangential velocity
        
        # Calculate specific energy
        energy = calculate_specific_energy(r0, v0)
        energy_type = "Bound (E < 0)" if energy < 0 else "Escape (E > 0)"
        
        # Simulate trajectory
        _, positions = simulate_trajectory(r0, v0)
        
        # Plot trajectory
        ax1.plot(positions[:, 0], positions[:, 1], positions[:, 2], color=colors[i], linewidth=2, label=f"{velocity_labels[i]}\n{energy_type}")
        
        # Add marker for initial position
        ax1.scatter(r0[0], r0[1], r0[2], color=colors[i], s=50)
    
    ax1.set_title('Effect of Release Velocity Magnitude')
    ax1.legend(loc='upper right', fontsize=8)
    
    # 2. Effect of release angle (ax2)
    # Base position and velocity magnitude
    r0 = [2.0, 0.0, 0.0]
    v_mag = v_circ
    
    # Different release angles (in degrees from tangential)
    angles = [0, 30, 60, 90]
    angle_labels = ['0° (Tangential)', '30°', '60°', '90° (Radial)']
    
    for i, angle in enumerate(angles):
        # Convert angle to radians
        angle_rad = np.radians(angle)
        
        # Calculate velocity components
        vx = -v_mag * np.sin(angle_rad)
        vy = v_mag * np.cos(angle_rad)
        v0 = [vx, vy, 0.0]
        
        # Simulate trajectory
        _, positions = simulate_trajectory(r0, v0)
        
        # Plot trajectory
        ax2.plot(positions[:, 0], positions[:, 1], positions[:, 2], color=colors[i], linewidth=2, label=angle_labels[i])
        
        # Add marker for initial position
        ax2.scatter(r0[0], r0[1], r0[2], color=colors[i], s=50)
    
    ax2.set_title('Effect of Release Angle')
    ax2.legend(loc='upper right', fontsize=8)
    
    # 3. Effect of release altitude (ax3)
    # Different release altitudes (in Earth radii)
    altitudes = [1.5, 2.0, 3.0, 4.0]
    altitude_labels = ['1.5 R_Earth', '2.0 R_Earth', '3.0 R_Earth', '4.0 R_Earth']
    
    for i, altitude in enumerate(altitudes):
        r0 = [altitude, 0.0, 0.0]
        
        # Calculate circular velocity at this altitude
        v_circ_alt = np.sqrt(1/altitude)
        v0 = [0.0, v_circ_alt, 0.0]
        
        # Simulate trajectory
        _, positions = simulate_trajectory(r0, v0)
        
        # Plot trajectory
        ax3.plot(positions[:, 0], positions[:, 1], positions[:, 2], color=colors[i], linewidth=2, label=altitude_labels[i])
        
        # Add marker for initial position
        ax3.scatter(r0[0], r0[1], r0[2], color=colors[i], s=50)
    
    ax3.set_title('Effect of Release Altitude')
    ax3.legend(loc='upper right', fontsize=8)
    
    # 4. Phase space diagram (ax4)
    # Create a grid of initial velocities and distances
    r_values = np.linspace(1.1, 5.0, 20)
    v_values = np.linspace(0.1, 1.5, 20)
    R, V = np.meshgrid(r_values, v_values)
    
    # Calculate the specific energy for each point
    E = np.zeros_like(R)
    for i in range(len(r_values)):
        for j in range(len(v_values)):
            r = r_values[i]
            v = v_values[j]
            
            # Specific energy in normalized units
            E[j, i] = 0.5 * v**2 - 1/r
    
    # Plot contours of specific energy
    contour = ax4.contourf(R, V, E, 20, cmap='viridis')
    
    # Add colorbar
    cbar = plt.colorbar(contour, ax=ax4)
    cbar.set_label('Specific Energy')
    
    # Add boundary line between bound and escape orbits (E = 0)
    v_escape = np.sqrt(2/r_values)
    ax4.plot(r_values, v_escape, 'r-', linewidth=2, label='Escape Boundary (E = 0)')
    
    # Add line for circular orbits
    v_circular = np.sqrt(1/r_values)
    ax4.plot(r_values, v_circular, 'g-', linewidth=2, label='Circular Orbits')
    
    # Add markers for the examples used in the other plots
    for i, factor in enumerate(velocity_factors):
        ax4.scatter(2.0, v_circ * factor, color=colors[i], s=50, marker='o')
    
    for i, altitude in enumerate(altitudes):
        v_circ_alt = np.sqrt(1/altitude)
        ax4.scatter(altitude, v_circ_alt, color=colors[i], s=50, marker='s')
    
    # Set axis properties
    ax4.set_xlabel('Distance (Earth radii)')
    ax4.set_ylabel('Velocity (normalized)')
    ax4.set_title('Phase Space: Orbit Types by Distance and Velocity')
    ax4.set_xlim(1.0, 5.0)
    ax4.set_ylim(0.1, 1.5)
    ax4.grid(True)
    ax4.legend(loc='upper right', fontsize=8)
    
    # Add annotations explaining the phase space
    ax4.text(1.5, 0.2, 'Bound Orbits\n(E < 0)', color='white', fontsize=10)
    ax4.text(4.0, 1.3, 'Escape Trajectories\n(E > 0)', color='black', fontsize=10)
    
    # Add overall title
    fig.suptitle('Effects of Release Parameters on Payload Trajectories', fontsize=16)
    
    # Add explanation text
    plt.figtext(0.5, 0.01, 
               "This visualization demonstrates how different release parameters affect payload trajectories:\n"
               "• Velocity magnitude determines whether the orbit is circular, elliptical, or hyperbolic\n"
               "• Release angle affects the orientation and eccentricity of the resulting orbit\n"
               "• Release altitude influences both the size of the orbit and the required velocity for different orbit types\n"
               "The phase space diagram shows the relationship between distance, velocity, and orbit type",
               ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.90, bottom=0.12)
    plt.savefig(os.path.join(figures_dir, 'release_parameter_effects.png'), dpi=300)
    plt.close()
    
    print("Release parameter effects visualization created successfully.")

def create_interactive_trajectory_animation():
    """Create an animation showing the evolution of a payload trajectory over time."""
    # Initial conditions
    r0 = [2.0, 0.0, 0.0]  # 2 Earth radii from center, on x-axis
    v_circ = np.sqrt(1/2)  # Circular velocity at r=2 (in normalized units)
    v0 = [0.0, v_circ * 1.2, 0.0]  # Slightly elliptical orbit
    
    # Simulate trajectory
    _, positions = simulate_trajectory(r0, v0, tmax=50000, dt=500)
    
    # Create figure and 3D axis
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Draw Earth
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)
    ax.plot_surface(x, y, z, color='blue', alpha=0.3)
    
    # Set axis properties
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_zlim(-3, 3)
    ax.set_xlabel('X (Earth radii)')
    ax.set_ylabel('Y (Earth radii)')
    ax.set_zlabel('Z (Earth radii)')
    ax.set_title('Payload Trajectory Evolution')
    
    # Initialize trajectory line and payload marker
    line, = ax.plot([], [], [], 'r-', linewidth=2, label='Trajectory')
    payload, = ax.plot([], [], [], 'ro', markersize=8, label='Payload')
    
    # Add time counter
    time_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes)
    
    # Animation initialization function
    def init():
        line.set_data([], [])
        line.set_3d_properties([])
        payload.set_data([], [])
        payload.set_3d_properties([])
        time_text.set_text('')
        return line, payload, time_text
    
    # Animation update function
    def update(frame):
        # Update trajectory line
        line.set_data(positions[:frame, 0], positions[:frame, 1])
        line.set_3d_properties(positions[:frame, 2])
        
        # Update payload position
        payload.set_data([positions[frame, 0]], [positions[frame, 1]])
        payload.set_3d_properties([positions[frame, 2]])
        
        # Update time text
        time_text.set_text(f'Time: {frame * 500} s')
        
        return line, payload, time_text
    
    # Create animation
    frames = min(100, len(positions))  # Limit to 100 frames for file size
    step = len(positions) // frames
    
    ani = FuncAnimation(fig, update, frames=range(0, len(positions), step),
                        init_func=init, blit=False, interval=50)
    
    # Save animation as a static image showing the full trajectory
    # (actual animation would be too large for a figure)
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'r-', linewidth=2, label='Full Trajectory')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'trajectory_evolution.png'), dpi=300)
    plt.close()
    
    print("Trajectory evolution visualization created successfully.")

if __name__ == "__main__":
    print("Generating additional visualizations for Problem 3 (Gravity)...")
    create_release_parameter_effects()
    create_interactive_trajectory_animation()
    print("Additional visualizations have been generated and saved to the figures folder.")
