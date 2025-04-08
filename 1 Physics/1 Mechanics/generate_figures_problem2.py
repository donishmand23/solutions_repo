import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib.cm as cm

# Set up figure directory
fig_dir = "docs/1 Physics/1 Mechanics"

# Figure 1: Forced damped pendulum diagram
plt.figure(figsize=(8, 10))
# Draw the pendulum
L = 2  # pendulum length
theta = np.pi/6  # angle (30 degrees)
x_bob = L * np.sin(theta)
y_bob = -L * np.cos(theta)

# Draw the pendulum rod and bob
plt.plot([0, x_bob], [0, y_bob], 'k-', lw=2)
plt.plot(x_bob, y_bob, 'bo', markersize=20)

# Draw the pivot
plt.plot(0, 0, 'ko', markersize=8)

# Draw the forces
# Gravity
plt.arrow(x_bob, y_bob, 0, -0.5, head_width=0.1, head_length=0.1, fc='g', ec='g')
plt.text(x_bob+0.1, y_bob-0.3, 'mg', fontsize=12)

# Damping (opposite to velocity)
v_direction = np.array([np.cos(theta+np.pi/2), np.sin(theta+np.pi/2)])
plt.arrow(x_bob, y_bob, -0.4*v_direction[0], -0.4*v_direction[1], 
          head_width=0.1, head_length=0.1, fc='r', ec='r')
plt.text(x_bob-0.6*v_direction[0], y_bob-0.6*v_direction[1], 'Damping', fontsize=12)

# External force (horizontal)
plt.arrow(x_bob, y_bob, 0.5, 0, head_width=0.1, head_length=0.1, fc='purple', ec='purple')
plt.text(x_bob+0.6, y_bob, 'F = A cos(ωt)', fontsize=12)

# Add angle arc and label
arc_radius = 0.5
theta_rad = np.linspace(0, theta, 30)
plt.plot(arc_radius * np.sin(theta_rad), -arc_radius * np.cos(theta_rad), 'k-')
plt.text(arc_radius/2 * np.sin(theta/2), -arc_radius/2 * np.cos(theta/2), 'θ', fontsize=14)

# Set up the plot
plt.axis('equal')
plt.xlim(-2.5, 2.5)
plt.ylim(-2.5, 1)
plt.grid(True, linestyle='--', alpha=0.7)
plt.title('Forced Damped Pendulum', fontsize=16)
plt.axis('off')

# Add parameter labels
plt.text(-2.3, 0.7, r'Parameters:', fontsize=12)
plt.text(-2.3, 0.5, r'$\theta$ = angular displacement', fontsize=12)
plt.text(-2.3, 0.3, r'$b$ = damping coefficient', fontsize=12)
plt.text(-2.3, 0.1, r'$g$ = gravitational acceleration', fontsize=12)
plt.text(-2.3, -0.1, r'$L$ = pendulum length', fontsize=12)
plt.text(-2.3, -0.3, r'$A$ = driving force amplitude', fontsize=12)
plt.text(-2.3, -0.5, r'$\omega$ = driving frequency', fontsize=12)

plt.savefig(f'{fig_dir}/forced_damped_pendulum_diagram.png', dpi=300, bbox_inches='tight')
plt.close()

# Figure 2: Amplitude vs driving frequency (resonance curves)
plt.figure(figsize=(10, 6))

# Parameters
omega0 = 1.0  # natural frequency
damping_values = [0.1, 0.3, 0.7, 1.5]  # different damping coefficients
A = 1.0  # driving amplitude
omega_range = np.linspace(0, 2.5, 500)  # range of driving frequencies

# Calculate amplitude response for each damping value
for b in damping_values:
    amplitude = A / np.sqrt((omega0**2 - omega_range**2)**2 + (b*omega_range)**2)
    plt.plot(omega_range, amplitude, label=f'b = {b}')

# Mark the natural frequency
plt.axvline(x=omega0, color='k', linestyle='--', alpha=0.5)
plt.text(omega0+0.02, 0.5, r'$\omega_0$', fontsize=12)

# Set up the plot
plt.xlabel('Driving Frequency (ω)', fontsize=12)
plt.ylabel('Amplitude Response', fontsize=12)
plt.title('Resonance Curves for Different Damping Values', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
plt.ylim(0, 10)

plt.savefig(f'{fig_dir}/amplitude_vs_frequency.png', dpi=300)
plt.close()

# Figure 3: Phase shift vs frequency
plt.figure(figsize=(10, 6))

# Calculate phase shift for each damping value
for b in damping_values:
    phase_shift = np.arctan2(b*omega_range, omega0**2 - omega_range**2) * 180/np.pi
    plt.plot(omega_range, phase_shift, label=f'b = {b}')

# Mark the natural frequency
plt.axvline(x=omega0, color='k', linestyle='--', alpha=0.5)
plt.text(omega0+0.02, 0, r'$\omega_0$', fontsize=12)

# Set up the plot
plt.xlabel('Driving Frequency (ω)', fontsize=12)
plt.ylabel('Phase Shift (degrees)', fontsize=12)
plt.title('Phase Shift vs Driving Frequency for Different Damping Values', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
plt.ylim(-180, 180)

plt.savefig(f'{fig_dir}/phase_shift_vs_frequency.png', dpi=300)
plt.close()

# Function to simulate the pendulum dynamics
def pendulum_dynamics(t, y, b, omega0_squared, A, Omega):
    theta, omega = y
    dydt = [
        omega,
        -b * omega - omega0_squared * np.sin(theta) + A * np.cos(Omega * t)
    ]
    return dydt

# Figure 4: Bifurcation diagram
plt.figure(figsize=(10, 8))

# Parameters
omega0_squared = 1.0  # natural frequency squared
b = 0.2  # damping coefficient
Omega = 2/3  # driving frequency
A_values = np.linspace(0.1, 1.5, 100)  # range of driving amplitudes
transient_periods = 100  # periods to discard as transient
sample_periods = 100  # periods to sample after transient

# For each driving amplitude, simulate and record theta values at fixed phase
theta_samples = []
for A in A_values:
    # Simulate for many periods to reach steady state
    t_span = [0, 2*np.pi/Omega * (transient_periods + sample_periods)]
    t_eval = np.linspace(t_span[0], t_span[1], 10000)
    
    sol = solve_ivp(
        lambda t, y: pendulum_dynamics(t, y, b, omega0_squared, A, Omega),
        t_span,
        [0.1, 0],  # initial conditions
        t_eval=t_eval,
        method='RK45',
        rtol=1e-6,
        atol=1e-9
    )
    
    # Extract theta values at fixed phase (once per period)
    t_samples = np.linspace(
        2*np.pi/Omega * transient_periods,
        t_span[1],
        sample_periods
    )
    
    # Find closest time points in the solution
    idx_samples = [np.argmin(np.abs(sol.t - t)) for t in t_samples]
    theta_at_samples = sol.y[0, idx_samples]
    
    # Add to our collection
    for theta in theta_at_samples:
        theta_samples.append((A, theta))

# Plot the bifurcation diagram
A_plot, theta_plot = zip(*theta_samples)
plt.scatter(A_plot, theta_plot, s=0.5, c='blue', alpha=0.5)

plt.xlabel('Driving Amplitude (A)', fontsize=12)
plt.ylabel('θ at Fixed Phase', fontsize=12)
plt.title('Bifurcation Diagram: Transition to Chaos', fontsize=14)
plt.grid(True, alpha=0.3)

plt.savefig(f'{fig_dir}/bifurcation_diagram.png', dpi=300)
plt.close()

# Figure 5: Phase portraits for different parameters
plt.figure(figsize=(15, 10))

# Define different parameter sets to show different behaviors
param_sets = [
    {'b': 0.1, 'A': 0.5, 'Omega': 0.9, 'title': 'Regular Motion (Small Forcing)'},
    {'b': 0.1, 'A': 1.2, 'Omega': 0.9, 'title': 'Chaotic Motion (Large Forcing)'},
    {'b': 0.5, 'A': 1.0, 'Omega': 1.0, 'title': 'Near Resonance'},
    {'b': 0.1, 'A': 0.7, 'Omega': 2.0, 'title': 'High Frequency Driving'}
]

for i, params in enumerate(param_sets):
    # Simulate the system
    b = params['b']
    A = params['A']
    Omega = params['Omega']
    
    t_span = [0, 100]
    sol = solve_ivp(
        lambda t, y: pendulum_dynamics(t, y, b, omega0_squared, A, Omega),
        t_span,
        [0.1, 0],
        t_eval=np.linspace(t_span[0], t_span[1], 10000),
        method='RK45'
    )
    
    # Skip the first part (transient)
    start_idx = len(sol.t) // 5
    
    # Plot phase portrait
    plt.subplot(2, 2, i+1)
    plt.plot(sol.y[0, start_idx:], sol.y[1, start_idx:], 'b-', linewidth=0.5)
    plt.xlabel('θ (angle)', fontsize=10)
    plt.ylabel('ω (angular velocity)', fontsize=10)
    plt.title(params['title'], fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add parameter text
    plt.text(0.05, 0.95, f"b={b}, A={A}, Ω={Omega}", transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top')

plt.tight_layout()
plt.savefig(f'{fig_dir}/phase_portrait.png', dpi=300)
plt.close()

# Figure 6: Poincaré sections
plt.figure(figsize=(15, 10))

# Parameters for Poincaré sections
A_values = [0.3, 0.65, 0.9, 1.3]  # different driving amplitudes
b = 0.2  # fixed damping
Omega = 2/3  # fixed driving frequency

for i, A in enumerate(A_values):
    # Simulate for a long time
    t_span = [0, 500]
    sol = solve_ivp(
        lambda t, y: pendulum_dynamics(t, y, b, omega0_squared, A, Omega),
        t_span,
        [0.1, 0],
        method='RK45',
        rtol=1e-6,
        atol=1e-9
    )
    
    # Extract points at fixed phase (once per period)
    period = 2*np.pi/Omega
    num_periods = int(t_span[1] / period)
    t_samples = np.array([period * i for i in range(50, num_periods)])
    
    # Find closest time points in the solution
    theta_samples = []
    omega_samples = []
    for t in t_samples:
        idx = np.argmin(np.abs(sol.t - t))
        theta_samples.append(sol.y[0, idx])
        omega_samples.append(sol.y[1, idx])
    
    # Plot Poincaré section
    plt.subplot(2, 2, i+1)
    plt.scatter(theta_samples, omega_samples, s=5, c='blue', alpha=0.7)
    plt.xlabel('θ (angle)', fontsize=10)
    plt.ylabel('ω (angular velocity)', fontsize=10)
    plt.title(f'Poincaré Section (A = {A})', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add parameter text
    plt.text(0.05, 0.95, f"b={b}, Ω={Omega}", transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top')

plt.tight_layout()
plt.savefig(f'{fig_dir}/poincare_section.png', dpi=300)
plt.close()

print("All figures for Problem 2 have been generated and saved to the docs/1 Physics/1 Mechanics folder.")
