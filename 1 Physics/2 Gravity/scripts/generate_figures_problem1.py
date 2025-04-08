import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch
from scipy.integrate import solve_ivp
import os

# Constants
G = 6.67430e-11  # Gravitational constant (m^3 kg^-1 s^-2)
M_SUN = 1.989e30  # Mass of the Sun (kg)
AU = 1.496e11    # Astronomical Unit in meters

# Create output directory if it doesn't exist
output_dir = os.path.dirname(os.path.abspath(__file__))

# Figure 1: Circular orbit diagram
def create_circular_orbit_diagram():
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Draw central body
    central_body = Circle((0, 0), 0.2, color='orange', zorder=10, label='Central Mass (M)')
    ax.add_patch(central_body)
    
    # Draw orbit
    orbit = Circle((0, 0), 1.0, fill=False, color='blue', linestyle='-', linewidth=2)
    ax.add_patch(orbit)
    
    # Draw orbiting body
    orbiting_body = Circle((1.0, 0), 0.1, color='blue', zorder=10, label='Orbiting Body (m)')
    ax.add_patch(orbiting_body)
    
    # Draw force vectors
    # Gravitational force
    grav_arrow = FancyArrowPatch((1.0, 0), (0.7, 0), arrowstyle='->', 
                                 color='red', linewidth=2, mutation_scale=20,
                                 label='Gravitational Force')
    ax.add_patch(grav_arrow)
    
    # Centripetal acceleration
    cent_arrow = FancyArrowPatch((1.0, 0), (0.7, 0), arrowstyle='->', 
                                 color='green', linewidth=2, mutation_scale=20,
                                 label='Centripetal Acceleration')
    ax.add_patch(cent_arrow)
    
    # Velocity vector
    vel_arrow = FancyArrowPatch((1.0, 0), (1.0, 0.3), arrowstyle='->', 
                               color='purple', linewidth=2, mutation_scale=20,
                               label='Velocity')
    ax.add_patch(vel_arrow)
    
    # Add labels and annotations
    plt.text(0, 0, 'M', ha='center', va='center', fontsize=12, color='white')
    plt.text(1.0, 0, 'm', ha='center', va='center', fontsize=12, color='white')
    plt.text(0.85, 0.1, '$F_g$', ha='center', va='center', fontsize=12, color='red')
    plt.text(0.85, -0.1, '$F_c$', ha='center', va='center', fontsize=12, color='green')
    plt.text(1.05, 0.15, '$v$', ha='center', va='center', fontsize=12, color='purple')
    plt.text(0.5, 0.5, '$r$', ha='center', va='center', fontsize=14)
    
    # Draw radius line
    plt.plot([0, 1.0], [0, 0], 'k--', alpha=0.5)
    
    # Set axis properties
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_title('Circular Orbit Dynamics', fontsize=16)
    ax.set_xlabel('x-position', fontsize=12)
    ax.set_ylabel('y-position', fontsize=12)
    
    # Add legend
    ax.legend(loc='upper right', fontsize=10)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'circular_orbit_diagram.png'), dpi=300)
    plt.close()

# Figure 2: Kepler's Law verification for solar system
def create_kepler_law_verification():
    # Planet data: name, period (years), semi-major axis (AU)
    planets = [
        ('Mercury', 0.241, 0.387),
        ('Venus', 0.615, 0.723),
        ('Earth', 1.000, 1.000),
        ('Mars', 1.881, 1.524),
        ('Jupiter', 11.86, 5.203),
        ('Saturn', 29.46, 9.537),
        ('Uranus', 84.01, 19.19),
        ('Neptune', 164.8, 30.07)
    ]
    
    # Extract data
    names = [p[0] for p in planets]
    periods = np.array([p[1] for p in planets])
    semi_major_axes = np.array([p[2] for p in planets])
    
    # Calculate T^2/a^3 (should be constant)
    t_squared = periods**2
    a_cubed = semi_major_axes**3
    ratio = t_squared / a_cubed
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot T^2 vs a^3
    ax1.scatter(a_cubed, t_squared, s=80, c='blue', alpha=0.7)
    
    # Add planet labels
    for i, name in enumerate(names):
        ax1.annotate(name, (a_cubed[i], t_squared[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=10)
    
    # Add best fit line
    x_fit = np.linspace(0, max(a_cubed)*1.1, 100)
    y_fit = np.mean(ratio) * x_fit
    ax1.plot(x_fit, y_fit, 'r--', label=f'Best fit: $T^2 = {np.mean(ratio):.3f} \\cdot a^3$')
    
    ax1.set_xlabel('Semi-major Axis Cubed ($AU^3$)', fontsize=12)
    ax1.set_ylabel('Orbital Period Squared ($years^2$)', fontsize=12)
    ax1.set_title("Kepler's Third Law: $T^2 \\propto a^3$", fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(fontsize=10)
    
    # Plot T^2/a^3 ratio (should be constant)
    ax2.bar(names, ratio, color='green', alpha=0.7)
    ax2.axhline(y=np.mean(ratio), color='r', linestyle='--', 
               label=f'Mean: {np.mean(ratio):.3f}')
    
    ax2.set_xlabel('Planet', fontsize=12)
    ax2.set_ylabel('$T^2/a^3$ Ratio ($years^2/AU^3$)', fontsize=12)
    ax2.set_title("Consistency of Kepler's Third Law", fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(fontsize=10)
    
    # Rotate x-axis labels for better readability
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'kepler_law_verification.png'), dpi=300)
    plt.close()

# Figure 3: Binary star system
def create_binary_star_diagram():
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Parameters
    m1 = 2.0  # Mass of star 1 (arbitrary units)
    m2 = 1.0  # Mass of star 2 (arbitrary units)
    a1 = m2 / (m1 + m2)  # Distance from center of mass to star 1
    a2 = m1 / (m1 + m2)  # Distance from center of mass to star 2
    
    # Draw center of mass
    plt.plot(0, 0, 'ko', markersize=8, label='Center of Mass')
    
    # Draw stars
    star1 = Circle((-a1, 0), 0.2, color='red', alpha=0.8, label=f'Star 1 (M={m1})')
    star2 = Circle((a2, 0), 0.15, color='blue', alpha=0.8, label=f'Star 2 (M={m2})')
    ax.add_patch(star1)
    ax.add_patch(star2)
    
    # Draw orbits
    orbit1 = Circle((0, 0), a1, fill=False, color='red', linestyle='-', linewidth=2)
    orbit2 = Circle((0, 0), a2, fill=False, color='blue', linestyle='-', linewidth=2)
    ax.add_patch(orbit1)
    ax.add_patch(orbit2)
    
    # Draw relative orbit (elliptical for visual interest)
    relative_orbit = Ellipse((0, 0), 2*(a1+a2), 1.5*(a1+a2), fill=False, 
                            color='purple', linestyle='--', linewidth=2, 
                            label='Relative Orbit')
    ax.add_patch(relative_orbit)
    
    # Add labels
    plt.text(-a1, 0, '$M_1$', ha='center', va='center', fontsize=12, color='white')
    plt.text(a2, 0, '$M_2$', ha='center', va='center', fontsize=12, color='white')
    plt.text(0, 0, 'CM', ha='center', va='bottom', fontsize=10)
    plt.text(-a1/2, 0.1, '$a_1$', ha='center', va='center', fontsize=12)
    plt.text(a2/2, 0.1, '$a_2$', ha='center', va='center', fontsize=12)
    
    # Draw semi-major axes
    plt.plot([0, -a1], [0, 0], 'r--', alpha=0.7)
    plt.plot([0, a2], [0, 0], 'b--', alpha=0.7)
    
    # Set axis properties
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_title('Binary Star System', fontsize=16)
    ax.set_xlabel('x-position', fontsize=12)
    ax.set_ylabel('y-position', fontsize=12)
    
    # Add legend
    ax.legend(loc='upper right', fontsize=10)
    
    # Add explanation text
    plt.figtext(0.5, 0.01, 
               "For binary systems: $T^2 = \\frac{4\\pi^2 a^3}{G(M_1 + M_2)}$ where $a$ is the semi-major axis of the relative orbit",
               ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'binary_star_system.png'), dpi=300)
    plt.close()

# Figure 4: Kepler's Law simulation
def create_kepler_law_simulation():
    # Define the differential equations for orbital motion
    def orbital_motion(t, state, GM):
        x, y, vx, vy = state
        r = np.sqrt(x**2 + y**2)
        ax = -GM * x / r**3
        ay = -GM * y / r**3
        return [vx, vy, ax, ay]
    
    # Parameters
    GM = 1.0  # Gravitational parameter (arbitrary units)
    
    # Different orbital radii to test
    radii = np.linspace(1.0, 5.0, 8)
    periods = []
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Simulate orbits at different radii
    for r in radii:
        # Initial conditions for circular orbit
        x0 = r
        y0 = 0
        vx0 = 0
        vy0 = np.sqrt(GM/r)  # Circular orbit velocity
        
        # Time span (adjust based on expected period)
        expected_period = 2 * np.pi * np.sqrt(r**3 / GM)
        t_span = (0, 2.5 * expected_period)
        t_eval = np.linspace(t_span[0], t_span[1], 1000)
        
        # Solve the differential equations
        sol = solve_ivp(orbital_motion, t_span, [x0, y0, vx0, vy0], 
                        args=(GM,), t_eval=t_eval, method='RK45')
        
        x = sol.y[0]
        y = sol.y[1]
        
        # Plot the orbit
        ax1.plot(x, y, label=f'r = {r:.1f}')
        
        # Calculate the period by finding when y changes from negative to positive
        # (crossing the x-axis from below)
        crossings = []
        for i in range(1, len(t_eval)):
            if y[i-1] < 0 and y[i] >= 0:
                crossings.append(t_eval[i])
        
        if len(crossings) >= 2:
            period = crossings[1] - crossings[0]
            periods.append(period)
        else:
            # If we couldn't detect two crossings, use the theoretical period
            periods.append(expected_period)
    
    # Plot central body
    central_body = Circle((0, 0), 0.2, color='orange', zorder=10)
    ax1.add_patch(central_body)
    
    # Set axis properties for orbit plot
    ax1.set_xlim(-6, 6)
    ax1.set_ylim(-6, 6)
    ax1.set_aspect('equal')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_title('Simulated Orbits at Different Radii', fontsize=14)
    ax1.set_xlabel('x-position', fontsize=12)
    ax1.set_ylabel('y-position', fontsize=12)
    
    # Convert to numpy array
    periods = np.array(periods)
    
    # Plot T^2 vs r^3
    r_cubed = radii**3
    t_squared = periods**2
    
    ax2.scatter(r_cubed, t_squared, s=80, c='blue', alpha=0.7)
    
    # Add best fit line
    ratio = t_squared / r_cubed
    x_fit = np.linspace(0, max(r_cubed)*1.1, 100)
    y_fit = np.mean(ratio) * x_fit
    ax2.plot(x_fit, y_fit, 'r--', label=f'Best fit: $T^2 = {np.mean(ratio):.3f} \\cdot r^3$')
    
    # Theoretical line
    y_theory = (4 * np.pi**2 / GM) * x_fit
    ax2.plot(x_fit, y_theory, 'g-', label=f'Theory: $T^2 = \\frac{{4\\pi^2}}{{GM}} \\cdot r^3$')
    
    # Set axis properties for T^2 vs r^3 plot
    ax2.set_xlabel('Orbital Radius Cubed ($r^3$)', fontsize=12)
    ax2.set_ylabel('Orbital Period Squared ($T^2$)', fontsize=12)
    ax2.set_title("Verification of Kepler's Third Law from Simulation", fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'kepler_law_simulation.png'), dpi=300)
    plt.close()

# Figure 5: Exoplanet transit
def create_exoplanet_transit():
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Time array
    time = np.linspace(0, 10, 1000)
    
    # Create light curve with transit dip
    flux = np.ones_like(time)
    
    # Add transit dips
    transit_centers = [2.5, 7.5]
    transit_width = 0.5
    transit_depth = 0.05
    
    for center in transit_centers:
        transit_mask = np.abs(time - center) < transit_width/2
        # Create a smooth transit shape
        in_transit = np.abs(time - center) < transit_width/2
        depth = transit_depth * (1 - ((time[in_transit] - center) / (transit_width/2))**2)
        flux[in_transit] -= depth
    
    # Plot light curve
    ax.plot(time, flux, 'b-', linewidth=2)
    
    # Add annotations
    ax.annotate('Transit', xy=(2.5, 0.95), xytext=(2.5, 0.9),
                arrowprops=dict(arrowstyle='->'), ha='center')
    
    ax.annotate('Period = 5 time units', xy=(5, 0.98), xytext=(5, 0.85),
                arrowprops=dict(arrowstyle='<->'), ha='center')
    
    # Add illustrations of the transit
    for i, center in enumerate(transit_centers):
        # Star
        star = Circle((center, 1.1), 0.1, color='yellow', alpha=0.8)
        ax.add_patch(star)
        
        # Planet
        planet = Circle((center, 1.1), 0.02, color='blue', alpha=0.8)
        ax.add_patch(planet)
        
        # Add text
        if i == 0:
            ax.text(center, 1.2, "Star + Planet Alignment\nCauses Flux Decrease", 
                   ha='center', va='center', fontsize=10,
                   bbox=dict(facecolor='white', alpha=0.8))
    
    # Set axis properties
    ax.set_xlim(0, 10)
    ax.set_ylim(0.8, 1.25)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Relative Flux', fontsize=12)
    ax.set_title('Exoplanet Transit Light Curve', fontsize=16)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add explanation text
    plt.figtext(0.5, 0.01, 
               "Using Kepler's Third Law: $T^2 = \\frac{4\\pi^2 a^3}{GM_*}$\n"
               "From the transit period (T) and stellar mass (M_*), we can determine the orbital radius (a)",
               ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'exoplanet_transit.png'), dpi=300)
    plt.close()

# Generate all figures
def generate_all_figures():
    print("Generating figures for Problem 1 (Gravity)...")
    create_circular_orbit_diagram()
    create_kepler_law_verification()
    create_binary_star_diagram()
    create_kepler_law_simulation()
    create_exoplanet_transit()
    print("All figures for Problem 1 (Gravity) have been generated and saved to the docs/1 Physics/2 Gravity folder.")

if __name__ == "__main__":
    generate_all_figures()
