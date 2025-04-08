import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, Wedge
from scipy.integrate import solve_ivp
import os
from mpl_toolkits.mplot3d import Axes3D

# Constants
G = 6.67430e-11  # Gravitational constant (m^3 kg^-1 s^-2)
M_EARTH = 5.972e24  # Mass of Earth (kg)
R_EARTH = 6.371e6  # Radius of Earth (m)

# Create output directory if it doesn't exist
output_dir = os.path.dirname(os.path.abspath(__file__))

# Figure 1: Coordinate system for payload trajectories
def create_coordinate_system():
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Draw Earth
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = R_EARTH * np.cos(u) * np.sin(v) / R_EARTH
    y = R_EARTH * np.sin(u) * np.sin(v) / R_EARTH
    z = R_EARTH * np.cos(v) / R_EARTH
    ax.plot_surface(x, y, z, color='blue', alpha=0.3)
    
    # Draw coordinate axes
    ax.quiver(0, 0, 0, 2, 0, 0, color='r', arrow_length_ratio=0.1, label='X')
    ax.quiver(0, 0, 0, 0, 2, 0, color='g', arrow_length_ratio=0.1, label='Y')
    ax.quiver(0, 0, 0, 0, 0, 2, color='b', arrow_length_ratio=0.1, label='Z')
    
    # Draw payload position and velocity vectors
    payload_pos = np.array([1.5, 1.0, 0.5])
    payload_vel = np.array([0.2, 0.3, 0.1])
    
    ax.scatter(payload_pos[0], payload_pos[1], payload_pos[2], color='red', s=100, label='Payload')
    ax.quiver(payload_pos[0], payload_pos[1], payload_pos[2], 
              payload_vel[0], payload_vel[1], payload_vel[2], 
              color='orange', arrow_length_ratio=0.1, label='Velocity')
    
    # Draw gravitational force vector
    r_vec = -payload_pos  # Vector from payload to Earth center
    r_mag = np.linalg.norm(r_vec)
    r_unit = r_vec / r_mag
    grav_force = 0.5 * r_unit  # Scaled for visualization
    
    ax.quiver(payload_pos[0], payload_pos[1], payload_pos[2], 
              grav_force[0], grav_force[1], grav_force[2], 
              color='purple', arrow_length_ratio=0.1, label='Gravitational Force')
    
    # Set axis properties
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-2, 2)
    ax.set_xlabel('X (Earth radii)')
    ax.set_ylabel('Y (Earth radii)')
    ax.set_zlabel('Z (Earth radii)')
    ax.set_title('Coordinate System for Payload Trajectory Analysis')
    
    # Add legend
    ax.legend()
    
    # Add annotations
    ax.text(0, 0, 0, 'Earth', color='blue')
    ax.text(payload_pos[0], payload_pos[1], payload_pos[2] + 0.2, 'Payload', color='red')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'payload_coordinate_system.png'), dpi=300)
    plt.close()

# Figure 2: Types of trajectories
def create_trajectory_types():
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Draw Earth
    earth = Circle((0, 0), 1, color='blue', alpha=0.7, label='Earth')
    ax.add_patch(earth)
    
    # Draw trajectories
    # Elliptical trajectory
    e_ellipse = 0.7  # Eccentricity
    a_ellipse = 3.0  # Semi-major axis
    b_ellipse = a_ellipse * np.sqrt(1 - e_ellipse**2)
    ellipse = Ellipse((a_ellipse * e_ellipse, 0), 2 * a_ellipse, 2 * b_ellipse, 
                      angle=0, fill=False, color='blue', linewidth=2, label='Elliptical')
    ax.add_patch(ellipse)
    
    # Parabolic trajectory
    theta_parabola = np.linspace(-np.pi/2, np.pi/2, 100)
    p_parabola = 3.0  # Semi-latus rectum
    r_parabola = p_parabola / (1 + np.cos(theta_parabola))
    x_parabola = r_parabola * np.cos(theta_parabola)
    y_parabola = r_parabola * np.sin(theta_parabola)
    ax.plot(x_parabola, y_parabola, 'g-', linewidth=2, label='Parabolic')
    
    # Hyperbolic trajectory
    theta_hyperbola = np.linspace(-np.pi/2, np.pi/2, 100)
    e_hyperbola = 1.5  # Eccentricity
    a_hyperbola = 2.0  # Semi-major axis
    p_hyperbola = a_hyperbola * (e_hyperbola**2 - 1)
    r_hyperbola = p_hyperbola / (1 + e_hyperbola * np.cos(theta_hyperbola))
    x_hyperbola = r_hyperbola * np.cos(theta_hyperbola)
    y_hyperbola = r_hyperbola * np.sin(theta_hyperbola)
    ax.plot(x_hyperbola, y_hyperbola, 'r-', linewidth=2, label='Hyperbolic')
    
    # Add spacecraft symbols at key points
    ax.scatter(a_ellipse * (1 - e_ellipse), 0, color='blue', s=100, zorder=5)
    ax.scatter(x_parabola[75], y_parabola[75], color='green', s=100, zorder=5)
    ax.scatter(x_hyperbola[75], y_hyperbola[75], color='red', s=100, zorder=5)
    
    # Add labels
    ax.text(0, 0, 'Earth', ha='center', va='center', color='white')
    ax.text(a_ellipse * (1 - e_ellipse) + 0.3, 0.3, 'Bound Orbit\n$ε < 0$', color='blue')
    ax.text(x_parabola[75] + 0.3, y_parabola[75], 'Parabolic Escape\n$ε = 0$', color='green')
    ax.text(x_hyperbola[75] + 0.3, y_hyperbola[75], 'Hyperbolic Escape\n$ε > 0$', color='red')
    
    # Set axis properties
    ax.set_xlim(-5, 8)
    ax.set_ylim(-6, 6)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_title('Types of Payload Trajectories', fontsize=16)
    ax.set_xlabel('Distance (Earth radii)', fontsize=12)
    ax.set_ylabel('Distance (Earth radii)', fontsize=12)
    
    # Add legend
    ax.legend(loc='upper right', fontsize=10)
    
    # Add explanation text
    plt.figtext(0.5, 0.01, 
               "Trajectory type depends on specific energy $ε = \\frac{v^2}{2} - \\frac{GM}{r}$\n"
               "Elliptical (blue): Payload remains in orbit around Earth\n"
               "Parabolic (green): Payload escapes with zero velocity at infinity\n"
               "Hyperbolic (red): Payload escapes with non-zero velocity at infinity",
               ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'trajectory_types.png'), dpi=300)
    plt.close()

# Figure 3: Initial conditions impact
def create_initial_conditions_impact():
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Draw Earth
    earth = Circle((0, 0), 1, color='blue', alpha=0.7, label='Earth')
    ax.add_patch(earth)
    
    # Draw rocket
    rocket_pos = np.array([2.5, 0])
    rocket_angle = np.pi/6  # 30 degrees
    rocket_length = 0.5
    rocket_dir = np.array([np.cos(rocket_angle), np.sin(rocket_angle)])
    
    # Draw rocket body
    ax.plot([rocket_pos[0], rocket_pos[0] + rocket_length * rocket_dir[0]],
            [rocket_pos[1], rocket_pos[1] + rocket_length * rocket_dir[1]],
            'k-', linewidth=3)
    
    # Draw rocket velocity vector
    rocket_vel = 0.8 * rocket_dir
    ax.arrow(rocket_pos[0], rocket_pos[1], 
             rocket_vel[0], rocket_vel[1], 
             head_width=0.1, head_length=0.2, fc='k', ec='k', label='Rocket Velocity')
    
    # Draw different release scenarios
    # 1. Release in direction of motion
    release_pos1 = rocket_pos + 0.8 * rocket_length * rocket_dir
    release_vel1 = rocket_vel * 1.2
    
    # Draw trajectory 1 (elliptical)
    e1 = 0.5
    a1 = 3.0
    b1 = a1 * np.sqrt(1 - e1**2)
    c1 = a1 * e1
    ellipse1 = Ellipse((c1, 0), 2 * a1, 2 * b1, 
                      angle=0, fill=False, color='green', linewidth=2, label='Forward Release')
    ax.add_patch(ellipse1)
    
    # 2. Release perpendicular to motion
    release_pos2 = rocket_pos + 0.4 * rocket_length * rocket_dir
    release_vel2 = np.array([-rocket_vel[1], rocket_vel[0]]) * 0.8
    
    # Draw trajectory 2 (highly elliptical)
    e2 = 0.8
    a2 = 4.0
    b2 = a2 * np.sqrt(1 - e2**2)
    c2 = a2 * e2
    ellipse2 = Ellipse((c2, 0), 2 * a2, 2 * b2, 
                      angle=0, fill=False, color='red', linewidth=2, label='Perpendicular Release')
    ax.add_patch(ellipse2)
    
    # 3. Release opposite to motion
    release_pos3 = rocket_pos
    release_vel3 = -rocket_vel * 0.5
    
    # Draw trajectory 3 (suborbital)
    theta3 = np.linspace(0, np.pi, 100)
    r3 = 2.0 / (1 + 0.9 * np.cos(theta3))
    x3 = r3 * np.cos(theta3)
    y3 = r3 * np.sin(theta3)
    ax.plot(x3, y3, 'b-', linewidth=2, label='Retrograde Release')
    
    # Add payload symbols
    ax.scatter(release_pos1[0], release_pos1[1], color='green', s=80, marker='^')
    ax.scatter(release_pos2[0], release_pos2[1], color='red', s=80, marker='^')
    ax.scatter(release_pos3[0], release_pos3[1], color='blue', s=80, marker='^')
    
    # Add velocity vectors
    ax.arrow(release_pos1[0], release_pos1[1], 
             release_vel1[0], release_vel1[1], 
             head_width=0.1, head_length=0.2, fc='green', ec='green')
    
    ax.arrow(release_pos2[0], release_pos2[1], 
             release_vel2[0], release_vel2[1], 
             head_width=0.1, head_length=0.2, fc='red', ec='red')
    
    ax.arrow(release_pos3[0], release_pos3[1], 
             release_vel3[0], release_vel3[1], 
             head_width=0.1, head_length=0.2, fc='blue', ec='blue')
    
    # Set axis properties
    ax.set_xlim(-6, 6)
    ax.set_ylim(-5, 5)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_title('Impact of Initial Release Conditions on Payload Trajectory', fontsize=16)
    ax.set_xlabel('Distance (Earth radii)', fontsize=12)
    ax.set_ylabel('Distance (Earth radii)', fontsize=12)
    
    # Add legend
    ax.legend(loc='upper right', fontsize=10)
    
    # Add explanation text
    plt.figtext(0.5, 0.01, 
               "The trajectory of a released payload depends critically on its initial conditions:\n"
               "• Release velocity magnitude and direction\n"
               "• Release altitude\n"
               "• Rocket's state (position and velocity) at the moment of release",
               ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'initial_conditions_impact.png'), dpi=300)
    plt.close()

# Figure 4: Simulation results
def create_simulation_results():
    # Function to simulate trajectory
    def simulate_trajectory(r0, v0, GM, tmax=10000, dt=100):
        # Initial state
        state0 = np.concatenate([r0, v0])
        
        # Define the derivatives function
        def derivatives(t, state):
            r = state[:3]
            v = state[3:]
            r_mag = np.linalg.norm(r)
            
            # Gravitational acceleration
            a = -GM * r / r_mag**3
            
            return np.concatenate([v, a])
        
        # Solve the differential equations
        sol = solve_ivp(
            derivatives,
            [0, tmax],
            state0,
            method='RK45',
            t_eval=np.arange(0, tmax, dt)
        )
        
        return sol.y[0], sol.y[1], sol.y[2]  # x, y, z positions
    
    # Create figure
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Draw Earth
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)
    ax.plot_surface(x, y, z, color='blue', alpha=0.3)
    
    # Simulate different trajectories
    # 1. Elliptical orbit (low energy)
    r0_elliptical = np.array([2.0, 0.0, 0.0])
    v0_elliptical = np.array([0.0, 0.7, 0.0])
    x_elliptical, y_elliptical, z_elliptical = simulate_trajectory(
        r0_elliptical, v0_elliptical, 1.0, tmax=50000, dt=500
    )
    ax.plot(x_elliptical, y_elliptical, z_elliptical, 'b-', linewidth=2, label='Elliptical Orbit')
    
    # 2. Parabolic escape (exact escape energy)
    r0_parabolic = np.array([2.0, 0.0, 0.0])
    v0_parabolic = np.array([0.0, 1.0, 0.0])
    x_parabolic, y_parabolic, z_parabolic = simulate_trajectory(
        r0_parabolic, v0_parabolic, 1.0, tmax=30000, dt=300
    )
    ax.plot(x_parabolic, y_parabolic, z_parabolic, 'g-', linewidth=2, label='Parabolic Escape')
    
    # 3. Hyperbolic escape (high energy)
    r0_hyperbolic = np.array([2.0, 0.0, 0.0])
    v0_hyperbolic = np.array([0.0, 1.3, 0.0])
    x_hyperbolic, y_hyperbolic, z_hyperbolic = simulate_trajectory(
        r0_hyperbolic, v0_hyperbolic, 1.0, tmax=20000, dt=200
    )
    ax.plot(x_hyperbolic, y_hyperbolic, z_hyperbolic, 'r-', linewidth=2, label='Hyperbolic Escape')
    
    # 4. Elliptical orbit with inclination
    r0_inclined = np.array([2.0, 0.0, 0.0])
    v0_inclined = np.array([0.0, 0.65, 0.25])
    x_inclined, y_inclined, z_inclined = simulate_trajectory(
        r0_inclined, v0_inclined, 1.0, tmax=50000, dt=500
    )
    ax.plot(x_inclined, y_inclined, z_inclined, 'purple', linewidth=2, label='Inclined Orbit')
    
    # Set axis properties
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-5, 5)
    ax.set_xlabel('X (Earth radii)')
    ax.set_ylabel('Y (Earth radii)')
    ax.set_zlabel('Z (Earth radii)')
    ax.set_title('Simulated Payload Trajectories for Different Initial Conditions')
    
    # Add legend
    ax.legend(loc='upper right')
    
    # Add annotations
    ax.text(0, 0, 0, 'Earth', color='blue')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'simulation_results.png'), dpi=300)
    plt.close()

# Figure 5: Sensitivity analysis
def create_sensitivity_analysis():
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # Draw Earth in all subplots
    for i in range(2):
        for j in range(2):
            earth = Circle((0, 0), 1, color='blue', alpha=0.7)
            axs[i, j].add_patch(earth)
            axs[i, j].set_aspect('equal')
            axs[i, j].grid(True, linestyle='--', alpha=0.7)
            axs[i, j].set_xlim(-6, 6)
            axs[i, j].set_ylim(-6, 6)
    
    # 1. Velocity magnitude sensitivity
    # Base elliptical orbit
    e_base = 0.5
    a_base = 3.0
    b_base = a_base * np.sqrt(1 - e_base**2)
    c_base = a_base * e_base
    ellipse_base = Ellipse((c_base, 0), 2 * a_base, 2 * b_base, 
                          angle=0, fill=False, color='blue', linewidth=2, label='Base Orbit')
    axs[0, 0].add_patch(ellipse_base)
    
    # Slightly higher velocity
    e_higher = 0.6
    a_higher = 3.5
    b_higher = a_higher * np.sqrt(1 - e_higher**2)
    c_higher = a_higher * e_higher
    ellipse_higher = Ellipse((c_higher, 0), 2 * a_higher, 2 * b_higher, 
                            angle=0, fill=False, color='green', linewidth=2, label='+5% Velocity')
    axs[0, 0].add_patch(ellipse_higher)
    
    # Slightly lower velocity
    e_lower = 0.4
    a_lower = 2.5
    b_lower = a_lower * np.sqrt(1 - e_lower**2)
    c_lower = a_lower * e_lower
    ellipse_lower = Ellipse((c_lower, 0), 2 * a_lower, 2 * b_lower, 
                           angle=0, fill=False, color='red', linewidth=2, label='-5% Velocity')
    axs[0, 0].add_patch(ellipse_lower)
    
    axs[0, 0].set_title('Sensitivity to Velocity Magnitude')
    axs[0, 0].legend(loc='upper right', fontsize=8)
    
    # 2. Release angle sensitivity
    # Base trajectory (horizontal release)
    theta_base = np.linspace(0, 2*np.pi, 100)
    r_base = 3.0 / (1 + 0.5 * np.cos(theta_base))
    x_base = r_base * np.cos(theta_base)
    y_base = r_base * np.sin(theta_base)
    axs[0, 1].plot(x_base, y_base, 'b-', linewidth=2, label='0° Release')
    
    # Angled release (+10 degrees)
    theta_plus = np.linspace(0, 2*np.pi, 100)
    r_plus = 3.0 / (1 + 0.5 * np.cos(theta_plus - np.pi/18))
    x_plus = r_plus * np.cos(theta_plus)
    y_plus = r_plus * np.sin(theta_plus)
    axs[0, 1].plot(x_plus, y_plus, 'g-', linewidth=2, label='+10° Release')
    
    # Angled release (-10 degrees)
    theta_minus = np.linspace(0, 2*np.pi, 100)
    r_minus = 3.0 / (1 + 0.5 * np.cos(theta_minus + np.pi/18))
    x_minus = r_minus * np.cos(theta_minus)
    y_minus = r_minus * np.sin(theta_minus)
    axs[0, 1].plot(x_minus, y_minus, 'r-', linewidth=2, label='-10° Release')
    
    axs[0, 1].set_title('Sensitivity to Release Angle')
    axs[0, 1].legend(loc='upper right', fontsize=8)
    
    # 3. Altitude sensitivity
    # Base altitude
    e_alt_base = 0.5
    a_alt_base = 3.0
    b_alt_base = a_alt_base * np.sqrt(1 - e_alt_base**2)
    c_alt_base = a_alt_base * e_alt_base
    ellipse_alt_base = Ellipse((c_alt_base, 0), 2 * a_alt_base, 2 * b_alt_base, 
                              angle=0, fill=False, color='blue', linewidth=2, label='Base Altitude')
    axs[1, 0].add_patch(ellipse_alt_base)
    
    # Higher altitude
    e_alt_higher = 0.45
    a_alt_higher = 3.5
    b_alt_higher = a_alt_higher * np.sqrt(1 - e_alt_higher**2)
    c_alt_higher = a_alt_higher * e_alt_higher
    ellipse_alt_higher = Ellipse((c_alt_higher, 0), 2 * a_alt_higher, 2 * b_alt_higher, 
                                angle=0, fill=False, color='green', linewidth=2, label='Higher Altitude')
    axs[1, 0].add_patch(ellipse_alt_higher)
    
    # Lower altitude
    e_alt_lower = 0.55
    a_alt_lower = 2.5
    b_alt_lower = a_alt_lower * np.sqrt(1 - e_alt_lower**2)
    c_alt_lower = a_alt_lower * e_alt_lower
    ellipse_alt_lower = Ellipse((c_alt_lower, 0), 2 * a_alt_lower, 2 * b_alt_lower, 
                               angle=0, fill=False, color='red', linewidth=2, label='Lower Altitude')
    axs[1, 0].add_patch(ellipse_alt_lower)
    
    axs[1, 0].set_title('Sensitivity to Release Altitude')
    axs[1, 0].legend(loc='upper right', fontsize=8)
    
    # 4. Combined sensitivity (near escape velocity)
    # Base trajectory (near escape)
    theta_esc_base = np.linspace(-np.pi/2, np.pi/2, 100)
    p_esc_base = 3.0
    e_esc_base = 0.95
    r_esc_base = p_esc_base / (1 + e_esc_base * np.cos(theta_esc_base))
    x_esc_base = r_esc_base * np.cos(theta_esc_base)
    y_esc_base = r_esc_base * np.sin(theta_esc_base)
    axs[1, 1].plot(x_esc_base, y_esc_base, 'b-', linewidth=2, label='Near Escape')
    
    # Slightly higher velocity (escape)
    theta_esc_higher = np.linspace(-np.pi/2, np.pi/2, 100)
    p_esc_higher = 3.0
    e_esc_higher = 1.05
    r_esc_higher = p_esc_higher / (1 + e_esc_higher * np.cos(theta_esc_higher))
    x_esc_higher = r_esc_higher * np.cos(theta_esc_higher)
    y_esc_higher = r_esc_higher * np.sin(theta_esc_higher)
    axs[1, 1].plot(x_esc_higher, y_esc_higher, 'g-', linewidth=2, label='Escape')
    
    # Slightly lower velocity (bound)
    theta_esc_lower = np.linspace(0, 2*np.pi, 100)
    p_esc_lower = 3.0
    e_esc_lower = 0.85
    r_esc_lower = p_esc_lower / (1 + e_esc_lower * np.cos(theta_esc_lower))
    x_esc_lower = r_esc_lower * np.cos(theta_esc_lower)
    y_esc_lower = r_esc_lower * np.sin(theta_esc_lower)
    axs[1, 1].plot(x_esc_lower, y_esc_lower, 'r-', linewidth=2, label='Bound')
    
    axs[1, 1].set_title('Sensitivity Near Escape Velocity')
    axs[1, 1].legend(loc='upper right', fontsize=8)
    
    # Add overall title
    fig.suptitle('Sensitivity Analysis of Payload Trajectories', fontsize=16)
    
    # Add explanation text
    plt.figtext(0.5, 0.01, 
               "This sensitivity analysis shows how small changes in initial conditions affect the resulting trajectory.\n"
               "Even minor variations in velocity, release angle, or altitude can significantly alter the payload's path,\n"
               "especially for trajectories near the escape velocity threshold.",
               ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, bottom=0.1)
    plt.savefig(os.path.join(output_dir, 'sensitivity_analysis.png'), dpi=300)
    plt.close()

# Generate all figures
def generate_all_figures():
    print("Generating figures for Problem 3 (Gravity)...")
    create_coordinate_system()
    create_trajectory_types()
    create_initial_conditions_impact()
    create_simulation_results()
    create_sensitivity_analysis()
    print("All figures for Problem 3 (Gravity) have been generated and saved to the docs/1 Physics/2 Gravity folder.")

if __name__ == "__main__":
    generate_all_figures()
