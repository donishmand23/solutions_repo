import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, Wedge
from scipy.integrate import solve_ivp
import os
from mpl_toolkits.mplot3d import Axes3D

# Constants
G = 6.67430e-11  # Gravitational constant (m^3 kg^-1 s^-2)
M_SUN = 1.989e30  # Mass of the Sun (kg)
M_EARTH = 5.972e24  # Mass of Earth (kg)
R_EARTH = 6.371e6  # Radius of Earth (m)
AU = 1.496e11  # Astronomical Unit in meters
C = 299792458  # Speed of light (m/s)

# Create output directory if it doesn't exist
output_dir = os.path.dirname(os.path.abspath(__file__))

# Figure 1: Cosmic velocities diagram
def create_cosmic_velocities_diagram():
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Draw planet
    planet = Circle((0, 0), 1.0, color='blue', alpha=0.7, label='Planet')
    ax.add_patch(planet)
    
    # Draw trajectories
    # Circular orbit (first cosmic velocity)
    theta = np.linspace(0, 2*np.pi, 100)
    r_orbit = 2.0
    x_orbit = r_orbit * np.cos(theta)
    y_orbit = r_orbit * np.sin(theta)
    ax.plot(x_orbit, y_orbit, 'g-', linewidth=2, label='Circular Orbit (v₁)')
    
    # Parabolic escape trajectory (second cosmic velocity)
    theta = np.linspace(-np.pi/2, np.pi/2, 100)
    r_parabola = 2.0 / (1 - np.cos(theta))
    x_parabola = r_parabola * np.cos(theta)
    y_parabola = r_parabola * np.sin(theta)
    ax.plot(x_parabola, y_parabola, 'r-', linewidth=2, label='Parabolic Escape (v₂)')
    
    # Hyperbolic trajectory (third cosmic velocity)
    theta = np.linspace(-np.pi/2, np.pi/2, 100)
    e = 1.5  # Eccentricity > 1 for hyperbola
    r_hyperbola = 2.0 / (1 - e * np.cos(theta))
    x_hyperbola = r_hyperbola * np.cos(theta)
    y_hyperbola = r_hyperbola * np.sin(theta)
    ax.plot(x_hyperbola, y_hyperbola, 'purple', linewidth=2, label='Hyperbolic Escape (v₃)')
    
    # Add spacecraft symbols at key points
    spacecraft1 = plt.scatter(x_orbit[25], y_orbit[25], color='green', s=100, zorder=5)
    spacecraft2 = plt.scatter(x_parabola[75], y_parabola[75], color='red', s=100, zorder=5)
    spacecraft3 = plt.scatter(x_hyperbola[75], y_hyperbola[75], color='purple', s=100, zorder=5)
    
    # Add velocity vectors
    # First cosmic velocity (tangential to orbit)
    v1_arrow = FancyArrowPatch((x_orbit[25], y_orbit[25]), 
                              (x_orbit[25] - 0.5*y_orbit[25]/r_orbit, y_orbit[25] + 0.5*x_orbit[25]/r_orbit), 
                              arrowstyle='->', color='green', linewidth=2, mutation_scale=20)
    ax.add_patch(v1_arrow)
    
    # Second cosmic velocity
    v2_arrow = FancyArrowPatch((x_parabola[75], y_parabola[75]), 
                              (x_parabola[75] + 0.7*np.cos(theta[75]), y_parabola[75] + 0.7*np.sin(theta[75])), 
                              arrowstyle='->', color='red', linewidth=2, mutation_scale=20)
    ax.add_patch(v2_arrow)
    
    # Third cosmic velocity
    v3_arrow = FancyArrowPatch((x_hyperbola[75], y_hyperbola[75]), 
                              (x_hyperbola[75] + 1.0*np.cos(theta[75]), y_hyperbola[75] + 1.0*np.sin(theta[75])), 
                              arrowstyle='->', color='purple', linewidth=2, mutation_scale=20)
    ax.add_patch(v3_arrow)
    
    # Add labels
    plt.text(x_orbit[25] - 0.3, y_orbit[25] + 0.3, '$v_1$', fontsize=14, color='green')
    plt.text(x_parabola[75] + 0.3, y_parabola[75] + 0.3, '$v_2$', fontsize=14, color='red')
    plt.text(x_hyperbola[75] + 0.3, y_hyperbola[75] + 0.3, '$v_3$', fontsize=14, color='purple')
    
    # Add annotations
    plt.text(0, 0, 'M', ha='center', va='center', fontsize=14, color='white')
    plt.text(2.0, 0.3, 'Circular Orbit\n$v_1 = \\sqrt{\\frac{GM}{r}}$', ha='center', va='center', fontsize=12)
    plt.text(4.0, 2.0, 'Parabolic Escape\n$v_2 = \\sqrt{\\frac{2GM}{r}}$', ha='center', va='center', fontsize=12)
    plt.text(5.0, 4.0, 'Hyperbolic Escape\n$v_3 > v_2$', ha='center', va='center', fontsize=12)
    
    # Set axis properties
    ax.set_xlim(-6, 6)
    ax.set_ylim(-2, 6)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_title('Cosmic Velocities and Resulting Trajectories', fontsize=16)
    ax.set_xlabel('x-position (arbitrary units)', fontsize=12)
    ax.set_ylabel('y-position (arbitrary units)', fontsize=12)
    
    # Add legend
    ax.legend(loc='upper right', fontsize=10)
    
    # Add explanation text
    plt.figtext(0.5, 0.01, 
               "$v_1$: First Cosmic Velocity (Orbital) - Object remains in orbit\n"
               "$v_2$: Second Cosmic Velocity (Escape) - Object escapes the planet\n"
               "$v_3$: Third Cosmic Velocity (Interstellar) - Object escapes the star system",
               ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cosmic_velocities_diagram.png'), dpi=300)
    plt.close()

# Figure 2: Escape velocities comparison
def create_escape_velocities_comparison():
    # Data for celestial bodies
    bodies = [
        ('Moon', 7.35e22, 1.737e6, 2.4),
        ('Mars', 6.42e23, 3.39e6, 5.0),
        ('Earth', 5.97e24, 6.371e6, 11.2),
        ('Jupiter', 1.90e27, 6.9911e7, 59.5),
        ('Sun', 1.99e30, 6.957e8, 617.5)
    ]
    
    names = [body[0] for body in bodies]
    masses = np.array([body[1] for body in bodies])
    radii = np.array([body[2] for body in bodies])
    escape_vels = np.array([body[3] for body in bodies])
    
    # Calculate escape velocities to verify
    calc_escape_vels = np.sqrt(2 * G * masses / radii) / 1000  # Convert to km/s
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    
    # Bar chart of escape velocities
    bars = ax1.bar(names, escape_vels, color=['gray', 'red', 'blue', 'orange', 'yellow'])
    
    # Add value labels on top of bars
    for bar, vel in zip(bars, escape_vels):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{vel:.1f} km/s', ha='center', va='bottom', fontsize=10)
    
    ax1.set_yscale('log')
    ax1.set_ylabel('Escape Velocity (km/s)', fontsize=12)
    ax1.set_title('Escape Velocities for Different Celestial Bodies', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.7, which='both')
    
    # Visual comparison with size and mass
    # Create circles representing the bodies (not to scale, but proportional to log of radius)
    log_radii = np.log10(radii)
    log_radii_normalized = 20 * (log_radii - min(log_radii)) / (max(log_radii) - min(log_radii)) + 10
    
    # Position the circles
    positions = np.linspace(0.2, 0.8, len(bodies))
    
    for i, (name, mass, radius, vel) in enumerate(bodies):
        # Draw circle
        circle = Circle((positions[i], 0.5), log_radii_normalized[i]/100, 
                       color=bars[i].get_facecolor(), alpha=0.7)
        ax2.add_patch(circle)
        
        # Add name and escape velocity
        ax2.text(positions[i], 0.5 + log_radii_normalized[i]/100 + 0.05, 
                name, ha='center', va='bottom', fontsize=12)
        ax2.text(positions[i], 0.5 - log_radii_normalized[i]/100 - 0.05, 
                f'{vel:.1f} km/s', ha='center', va='top', fontsize=10)
    
    # Add formula
    ax2.text(0.5, 0.9, '$v_{esc} = \\sqrt{\\frac{2GM}{r}}$', ha='center', va='center', 
            fontsize=16, bbox=dict(facecolor='white', alpha=0.8))
    
    # Set axis properties
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_title('Relative Size Comparison (not to scale)', fontsize=14)
    ax2.axis('off')
    
    # Add explanation
    plt.figtext(0.5, 0.01, 
               "Escape velocity depends on the mass and radius of the celestial body.\n"
               "More massive bodies with smaller radii have higher escape velocities.",
               ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'escape_velocities_comparison.png'), dpi=300)
    plt.close()

# Figure 3: Altitude effect on escape velocity
def create_altitude_effect_graph():
    # Calculate escape velocity as a function of altitude
    altitudes = np.linspace(0, 40000, 1000)  # km above Earth's surface
    distances = R_EARTH + altitudes * 1000  # Convert to meters
    
    escape_velocities = np.sqrt(2 * G * M_EARTH / distances) / 1000  # km/s
    
    # Calculate orbital velocities for comparison
    orbital_velocities = np.sqrt(G * M_EARTH / distances) / 1000  # km/s
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Plot escape velocity
    ax.plot(altitudes, escape_velocities, 'r-', linewidth=2, label='Escape Velocity (v₂)')
    
    # Plot orbital velocity
    ax.plot(altitudes, orbital_velocities, 'g-', linewidth=2, label='Orbital Velocity (v₁)')
    
    # Add key points
    # Surface
    ax.scatter(0, escape_velocities[0], color='red', s=100, zorder=5)
    ax.text(1000, escape_velocities[0] + 0.2, f'Surface: {escape_velocities[0]:.1f} km/s', fontsize=10)
    
    # LEO (Low Earth Orbit) ~400km
    leo_idx = np.abs(altitudes - 400).argmin()
    ax.scatter(400, escape_velocities[leo_idx], color='red', s=100, zorder=5)
    ax.text(1000, escape_velocities[leo_idx] + 0.2, f'LEO (400km): {escape_velocities[leo_idx]:.1f} km/s', fontsize=10)
    
    # GEO (Geostationary Orbit) ~35,786km
    geo_idx = np.abs(altitudes - 35786).argmin()
    ax.scatter(35786, escape_velocities[geo_idx], color='red', s=100, zorder=5)
    ax.text(30000, escape_velocities[geo_idx] + 0.2, f'GEO: {escape_velocities[geo_idx]:.1f} km/s', fontsize=10)
    
    # Add ISS for reference
    iss_altitude = 400  # km
    iss_idx = np.abs(altitudes - iss_altitude).argmin()
    ax.scatter(iss_altitude, orbital_velocities[iss_idx], color='green', s=100, zorder=5)
    ax.text(1000, orbital_velocities[iss_idx] - 0.3, f'ISS Orbit: {orbital_velocities[iss_idx]:.1f} km/s', fontsize=10)
    
    # Set axis properties
    ax.set_xlim(0, 40000)
    ax.set_ylim(0, 12)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_title('Escape and Orbital Velocities vs. Altitude Above Earth', fontsize=16)
    ax.set_xlabel('Altitude (km)', fontsize=12)
    ax.set_ylabel('Velocity (km/s)', fontsize=12)
    
    # Add legend
    ax.legend(loc='upper right', fontsize=12)
    
    # Add formula
    plt.figtext(0.5, 0.01, 
               "$v_{esc} = \\sqrt{\\frac{2GM}{r}}$ where $r = R_{Earth} + altitude$\n"
               "As altitude increases, both escape velocity and orbital velocity decrease.",
               ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'altitude_effect.png'), dpi=300)
    plt.close()

# Figure 4: Interstellar mission concept
def create_interstellar_mission_concept():
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create Sun
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x_sun = 0.2 * np.cos(u) * np.sin(v)
    y_sun = 0.2 * np.sin(u) * np.sin(v)
    z_sun = 0.2 * np.cos(v)
    ax.plot_surface(x_sun, y_sun, z_sun, color='yellow', alpha=0.8)
    
    # Create Earth's orbit
    theta = np.linspace(0, 2*np.pi, 100)
    x_earth_orbit = np.cos(theta)
    y_earth_orbit = np.sin(theta)
    z_earth_orbit = np.zeros_like(theta)
    ax.plot(x_earth_orbit, y_earth_orbit, z_earth_orbit, 'b-', alpha=0.3)
    
    # Create Earth
    earth_pos = np.pi/4  # Position on orbit
    x_earth = np.cos(earth_pos)
    y_earth = np.sin(earth_pos)
    z_earth = 0
    
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x_earth_sphere = x_earth + 0.05 * np.cos(u) * np.sin(v)
    y_earth_sphere = y_earth + 0.05 * np.sin(u) * np.sin(v)
    z_earth_sphere = z_earth + 0.05 * np.cos(v)
    ax.plot_surface(x_earth_sphere, y_earth_sphere, z_earth_sphere, color='blue', alpha=0.8)
    
    # Create spacecraft trajectory
    t = np.linspace(0, 5, 1000)
    # Spiral out from Earth
    spiral_radius = 0.05 + t * 0.2
    spiral_theta = earth_pos + t * 4
    x_spiral = x_earth + spiral_radius * np.cos(spiral_theta)
    y_spiral = y_earth + spiral_radius * np.sin(spiral_theta)
    z_spiral = 0.1 * t
    
    # Then straight line out of solar system
    x_escape = x_spiral[-1] + 0.5 * t
    y_escape = y_spiral[-1] + 0.5 * t
    z_escape = z_spiral[-1] + 0.2 * t
    
    # Plot trajectory
    ax.plot(x_spiral, y_spiral, z_spiral, 'r-', linewidth=2, label='Spacecraft Trajectory')
    ax.plot(x_escape, y_escape, z_escape, 'r-', linewidth=2)
    
    # Add spacecraft
    ax.scatter(x_escape[-1], y_escape[-1], z_escape[-1], color='silver', s=100, marker='^')
    
    # Add solar system boundary
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x_boundary = 3 * np.cos(u) * np.sin(v)
    y_boundary = 3 * np.sin(u) * np.sin(v)
    z_boundary = 3 * np.cos(v)
    ax.plot_surface(x_boundary, y_boundary, z_boundary, color='gray', alpha=0.1)
    
    # Add labels
    ax.text(0, 0, 0, 'Sun', color='black', fontsize=12)
    ax.text(x_earth, y_earth, z_earth + 0.1, 'Earth', color='black', fontsize=12)
    ax.text(x_escape[-1], y_escape[-1], z_escape[-1] + 0.2, 'Spacecraft', color='black', fontsize=12)
    
    # Add velocity annotations
    ax.text(x_spiral[200], y_spiral[200], z_spiral[200] + 0.2, 
           'v > Earth Escape Velocity\n(Second Cosmic Velocity)', 
           color='red', fontsize=10)
    
    ax.text(x_escape[200], y_escape[200], z_escape[200] + 0.2, 
           'v > Solar System Escape Velocity\n(Third Cosmic Velocity)', 
           color='purple', fontsize=10)
    
    # Set axis properties
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_zlim(-1, 3)
    ax.set_title('Interstellar Mission Concept: Escaping the Solar System', fontsize=16)
    ax.set_xlabel('X (AU)', fontsize=12)
    ax.set_ylabel('Y (AU)', fontsize=12)
    ax.set_zlabel('Z (AU)', fontsize=12)
    
    # Add propulsion system annotations
    plt.figtext(0.2, 0.02, 
               "Propulsion Systems for Interstellar Travel:\n"
               "• Nuclear Propulsion\n"
               "• Ion Drives\n"
               "• Solar Sails\n"
               "• Theoretical: Antimatter, Fusion",
               ha='left', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.figtext(0.7, 0.02, 
               "Third Cosmic Velocity (from Earth):\n"
               "≈ 42.1 km/s relative to Sun\n"
               "≈ 16.7 km/s relative to Earth",
               ha='left', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'interstellar_mission_concept.png'), dpi=300)
    plt.close()

# Figure 5: Trajectory simulations
def create_trajectory_simulations():
    # Function to simulate trajectories
    def simulate_trajectory(initial_velocity_ratio, planet_mass, planet_radius, max_time=100000):
        """Simulate trajectory of an object launched from a planet's surface."""
        # Calculate escape velocity
        v_esc = np.sqrt(2 * G * planet_mass / planet_radius)
        
        # Set initial velocity as a ratio of escape velocity
        initial_velocity = initial_velocity_ratio * v_esc
        
        # Launch angle (tangential to surface for orbital insertion)
        angle_degrees = 0  # Horizontal launch
        angle_radians = np.radians(angle_degrees)
        
        # Initial conditions
        x0 = 0
        y0 = planet_radius
        vx0 = initial_velocity
        vy0 = 0
        
        # Differential equations for motion in gravitational field
        def motion_equations(t, state):
            x, y, vx, vy = state
            r = np.sqrt(x**2 + y**2)
            
            # Check if object has crashed into the planet
            if r < planet_radius:
                return [0, 0, 0, 0]
            
            # Gravitational acceleration components
            ax = -G * planet_mass * x / r**3
            ay = -G * planet_mass * y / r**3
            
            return [vx, vy, ax, ay]
        
        # Solve the differential equations
        solution = solve_ivp(
            motion_equations,
            [0, max_time],
            [x0, y0, vx0, vy0],
            method='RK45',
            rtol=1e-8,
            atol=1e-8
        )
        
        return solution.t, solution.y[0], solution.y[1], initial_velocity_ratio
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Draw Earth
    earth = Circle((0, 0), R_EARTH/R_EARTH, color='blue', alpha=0.7, label='Earth')
    ax.add_patch(earth)
    
    # Simulate trajectories for different initial velocities
    velocity_ratios = [0.7, 0.9, 1.0, 1.2, 1.5]
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    labels = [
        'Sub-orbital (0.7 × v₂)',
        'Elliptical orbit (0.9 × v₂)',
        'Parabolic escape (1.0 × v₂)',
        'Hyperbolic escape (1.2 × v₂)',
        'Fast escape (1.5 × v₂)'
    ]
    
    for i, v_ratio in enumerate(velocity_ratios):
        t, x, y, ratio = simulate_trajectory(v_ratio, M_EARTH, R_EARTH)
        
        # Scale for plotting (in Earth radii)
        x_scaled = x / R_EARTH
        y_scaled = y / R_EARTH
        
        # Plot trajectory
        ax.plot(x_scaled, y_scaled, color=colors[i], linewidth=2, label=labels[i])
        
        # Add spacecraft at end of trajectory
        ax.scatter(x_scaled[-1], y_scaled[-1], color=colors[i], s=80, marker='^')
    
    # Set axis properties
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_title('Trajectories for Different Initial Velocities', fontsize=16)
    ax.set_xlabel('Distance (Earth radii)', fontsize=12)
    ax.set_ylabel('Distance (Earth radii)', fontsize=12)
    
    # Add legend
    ax.legend(loc='upper right', fontsize=10)
    
    # Add explanation
    plt.figtext(0.5, 0.01, 
               "v₂ = Second Cosmic Velocity (Escape Velocity) ≈ 11.2 km/s from Earth's surface\n"
               "At exactly escape velocity, the trajectory is parabolic.\n"
               "Below escape velocity, the object either falls back or enters an elliptical orbit.\n"
               "Above escape velocity, the object escapes on a hyperbolic trajectory.",
               ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'trajectory_simulations.png'), dpi=300)
    plt.close()

# Generate all figures
def generate_all_figures():
    print("Generating figures for Problem 2 (Gravity)...")
    create_cosmic_velocities_diagram()
    create_escape_velocities_comparison()
    create_altitude_effect_graph()
    create_interstellar_mission_concept()
    create_trajectory_simulations()
    print("All figures for Problem 2 (Gravity) have been generated and saved to the docs/1 Physics/2 Gravity folder.")

if __name__ == "__main__":
    generate_all_figures()
