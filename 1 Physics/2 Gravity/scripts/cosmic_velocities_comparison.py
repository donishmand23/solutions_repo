import numpy as np
import matplotlib.pyplot as plt
import os

# Constants
G = 6.67430e-11  # Gravitational constant (m^3 kg^-1 s^-2)
M_SUN = 1.989e30  # Mass of the Sun (kg)
AU = 1.496e11  # Astronomical Unit in meters

# Celestial body data: name, mass (kg), radius (m), distance from Sun (AU)
bodies = [
    ('Mercury', 3.3011e23, 2.4397e6, 0.387),
    ('Venus', 4.8675e24, 6.0518e6, 0.723),
    ('Earth', 5.972e24, 6.371e6, 1.0),
    ('Mars', 6.4171e23, 3.3895e6, 1.524),
    ('Jupiter', 1.8982e27, 6.9911e7, 5.203),
    ('Saturn', 5.6834e26, 5.8232e7, 9.537),
    ('Uranus', 8.6810e25, 2.5362e7, 19.191),
    ('Neptune', 1.02413e26, 2.4622e7, 30.069)
]

def calculate_cosmic_velocities(mass, radius, distance_from_sun):
    """Calculate the three cosmic velocities for a celestial body.
    
    Parameters:
    - mass: Mass of the celestial body in kg
    - radius: Radius of the celestial body in meters
    - distance_from_sun: Distance from the Sun in AU
    
    Returns:
    - v1: First cosmic velocity (orbital velocity) in km/s
    - v2: Second cosmic velocity (escape velocity) in km/s
    - v3: Third cosmic velocity (interstellar escape velocity) in km/s
    """
    # First cosmic velocity (orbital velocity)
    v1 = np.sqrt(G * mass / radius) / 1000  # Convert to km/s
    
    # Second cosmic velocity (escape velocity)
    v2 = np.sqrt(2 * G * mass / radius) / 1000  # Convert to km/s
    
    # Third cosmic velocity (interstellar escape velocity)
    # Calculate Sun's escape velocity at the planet's distance
    v_esc_sun = np.sqrt(2 * G * M_SUN / (distance_from_sun * AU)) / 1000  # km/s
    
    # Third cosmic velocity is the vector sum of planet's escape velocity and Sun's escape velocity
    v3 = np.sqrt(v2**2 + v_esc_sun**2)  # km/s
    
    return v1, v2, v3

def create_cosmic_velocities_comparison():
    # Calculate cosmic velocities for each body
    names = []
    v1_values = []
    v2_values = []
    v3_values = []
    
    for name, mass, radius, distance in bodies:
        v1, v2, v3 = calculate_cosmic_velocities(mass, radius, distance)
        names.append(name)
        v1_values.append(v1)
        v2_values.append(v2)
        v3_values.append(v3)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Set width of bars
    bar_width = 0.25
    
    # Set positions of bars on X axis
    r1 = np.arange(len(names))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    
    # Create bars
    bars1 = ax.bar(r1, v1_values, width=bar_width, color='blue', edgecolor='black', label='First Cosmic Velocity (Orbital)')
    bars2 = ax.bar(r2, v2_values, width=bar_width, color='green', edgecolor='black', label='Second Cosmic Velocity (Escape)')
    bars3 = ax.bar(r3, v3_values, width=bar_width, color='red', edgecolor='black', label='Third Cosmic Velocity (Interstellar)')
    
    # Add value labels on top of bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
    
    add_labels(bars1)
    add_labels(bars2)
    add_labels(bars3)
    
    # Add labels and title
    ax.set_xlabel('Celestial Body', fontsize=12)
    ax.set_ylabel('Velocity (km/s)', fontsize=12)
    ax.set_title('Comparison of Cosmic Velocities Across Solar System Bodies', fontsize=16)
    ax.set_xticks([r + bar_width for r in range(len(names))])
    ax.set_xticklabels(names)
    
    # Add legend
    ax.legend()
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # Add explanatory text
    plt.figtext(0.5, 0.01,
               "First Cosmic Velocity (v₁): Minimum velocity for circular orbit\n"
               "Second Cosmic Velocity (v₂): Minimum velocity to escape the planet's gravity\n"
               "Third Cosmic Velocity (v₃): Minimum velocity to escape the solar system from the planet",
               ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    # Save figure
    output_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(output_dir)
    figures_dir = os.path.join(parent_dir, 'figures')
    
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])  # Adjust layout to make room for the text
    plt.savefig(os.path.join(figures_dir, 'cosmic_velocities_comparison_planets.png'), dpi=300)
    plt.close()
    
    print("Cosmic velocities comparison figure created successfully.")

if __name__ == "__main__":
    create_cosmic_velocities_comparison()
