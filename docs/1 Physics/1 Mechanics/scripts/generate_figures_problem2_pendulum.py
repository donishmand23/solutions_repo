import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os

# Create figures directory if it doesn't exist
figures_dir = 'figures'
if not os.path.exists(figures_dir):
    os.makedirs(figures_dir)

# Common parameters
g = 9.81  # gravitational acceleration (m/s^2)
L = 1.0   # pendulum length (m)
omega0_squared = g/L  # natural frequency squared
omega0 = np.sqrt(omega0_squared)  # natural frequency

# Time settings
t_max = 20  # maximum simulation time (s)
dt = 0.01   # time step for plotting

# 1. Simple Pendulum
def simple_pendulum(t, y):
    theta, omega = y
    dtheta_dt = omega
    domega_dt = -omega0_squared * np.sin(theta)
    return [dtheta_dt, domega_dt]

def plot_simple_pendulum():
    print("Generating simple pendulum plot...")
    # Initial conditions: theta = 30 degrees, omega = 0
    y0 = [np.radians(30), 0]
    
    # Solve the ODE
    t_span = (0, t_max)
    t_eval = np.arange(0, t_max, dt)
    sol = solve_ivp(simple_pendulum, t_span, y0, method='RK45', t_eval=t_eval)
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(sol.t, np.degrees(sol.y[0]), 'b-', linewidth=2)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Angle (degrees)', fontsize=12)
    plt.title('Simple Pendulum Motion', fontsize=14)
    plt.xlim(0, t_max)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'simple_pendulum.png'), dpi=300)
    plt.close()

# 2. Damped Pendulum
def damped_pendulum(t, y, b=0.5):
    theta, omega = y
    dtheta_dt = omega
    domega_dt = -b * omega - omega0_squared * np.sin(theta)
    return [dtheta_dt, domega_dt]

def plot_damped_pendulum():
    print("Generating damped pendulum plot...")
    # Initial conditions: theta = 30 degrees, omega = 0
    y0 = [np.radians(30), 0]
    
    # Different damping coefficients
    damping_coefficients = [0.1, 0.5, 1.0]
    labels = [f'b = {b}' for b in damping_coefficients]
    colors = ['b', 'g', 'r']
    
    plt.figure(figsize=(10, 6))
    
    # Solve and plot for each damping coefficient
    t_span = (0, t_max)
    t_eval = np.arange(0, t_max, dt)
    
    for i, b in enumerate(damping_coefficients):
        sol = solve_ivp(lambda t, y: damped_pendulum(t, y, b), t_span, y0, method='RK45', t_eval=t_eval)
        plt.plot(sol.t, np.degrees(sol.y[0]), color=colors[i], linewidth=2, label=labels[i])
    
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Angle (degrees)', fontsize=12)
    plt.title('Damped Pendulum Motion with Different Damping Coefficients', fontsize=14)
    plt.legend(fontsize=10)
    plt.xlim(0, t_max)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'damped_pendulum.png'), dpi=300)
    plt.close()

# 3. Forced Pendulum (no damping)
def forced_pendulum(t, y, A=1.0, Omega=0.667*omega0):
    theta, omega = y
    dtheta_dt = omega
    domega_dt = -omega0_squared * np.sin(theta) + A * np.cos(Omega * t)
    return [dtheta_dt, domega_dt]

def plot_forced_pendulum():
    print("Generating forced pendulum plot...")
    # Initial conditions: theta = 0 degrees, omega = 0
    y0 = [0, 0]
    
    # Different driving frequencies
    driving_frequencies = [0.5*omega0, 0.667*omega0, omega0]
    labels = [f'Ω = {freq:.2f}ω₀' for freq in [0.5, 0.667, 1.0]]
    colors = ['b', 'g', 'r']
    
    plt.figure(figsize=(10, 6))
    
    # Solve and plot for each driving frequency
    t_span = (0, t_max)
    t_eval = np.arange(0, t_max, dt)
    
    for i, Omega in enumerate(driving_frequencies):
        sol = solve_ivp(lambda t, y: forced_pendulum(t, y, A=0.5, Omega=Omega), 
                        t_span, y0, method='RK45', t_eval=t_eval)
        plt.plot(sol.t, np.degrees(sol.y[0]), color=colors[i], linewidth=2, label=labels[i])
    
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Angle (degrees)', fontsize=12)
    plt.title('Forced Pendulum Motion with Different Driving Frequencies', fontsize=14)
    plt.legend(fontsize=10)
    plt.xlim(0, t_max)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'forced_pendulum.png'), dpi=300)
    plt.close()

# 4 & 5. Forced Damped Pendulum (two scenarios)
def forced_damped_pendulum(t, y, b=0.5, A=1.0, Omega=0.667*omega0):
    theta, omega = y
    dtheta_dt = omega
    domega_dt = -b * omega - omega0_squared * np.sin(theta) + A * np.cos(Omega * t)
    return [dtheta_dt, domega_dt]

def plot_forced_damped_pendulum_scenario1():
    print("Generating forced damped pendulum scenario 1 plot...")
    # Initial conditions: theta = 0 degrees, omega = 0
    y0 = [0, 0]
    
    # Fixed damping, different driving frequencies
    b = 0.2
    driving_frequencies = [0.5*omega0, 0.667*omega0, omega0]
    labels = [f'Ω = {freq:.2f}ω₀' for freq in [0.5, 0.667, 1.0]]
    colors = ['b', 'g', 'r']
    
    plt.figure(figsize=(10, 6))
    
    # Solve and plot for each driving frequency
    t_span = (0, t_max)
    t_eval = np.arange(0, t_max, dt)
    
    for i, Omega in enumerate(driving_frequencies):
        sol = solve_ivp(lambda t, y: forced_damped_pendulum(t, y, b=b, A=0.5, Omega=Omega), 
                        t_span, y0, method='RK45', t_eval=t_eval)
        plt.plot(sol.t, np.degrees(sol.y[0]), color=colors[i], linewidth=2, label=labels[i])
    
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Angle (degrees)', fontsize=12)
    plt.title(f'Forced Damped Pendulum (b = {b}) with Different Driving Frequencies', fontsize=14)
    plt.legend(fontsize=10)
    plt.xlim(0, t_max)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'forced_damped_pendulum_scenario1.png'), dpi=300)
    plt.close()

def plot_forced_damped_pendulum_scenario2():
    print("Generating forced damped pendulum scenario 2 plot...")
    # Initial conditions: theta = 0 degrees, omega = 0
    y0 = [0, 0]
    
    # Fixed driving frequency (near resonance), different damping
    Omega = 0.95 * omega0  # Near resonance
    damping_coefficients = [0.05, 0.2, 0.5]
    labels = [f'b = {b}' for b in damping_coefficients]
    colors = ['b', 'g', 'r']
    
    plt.figure(figsize=(10, 6))
    
    # Solve and plot for each damping coefficient
    t_span = (0, t_max)
    t_eval = np.arange(0, t_max, dt)
    
    for i, b in enumerate(damping_coefficients):
        sol = solve_ivp(lambda t, y: forced_damped_pendulum(t, y, b=b, A=0.5, Omega=Omega), 
                        t_span, y0, method='RK45', t_eval=t_eval)
        plt.plot(sol.t, np.degrees(sol.y[0]), color=colors[i], linewidth=2, label=labels[i])
    
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Angle (degrees)', fontsize=12)
    plt.title(f'Forced Damped Pendulum (Ω = {Omega:.2f}ω₀) with Different Damping', fontsize=14)
    plt.legend(fontsize=10)
    plt.xlim(0, t_max)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'forced_damped_pendulum_scenario2.png'), dpi=300)
    plt.close()

if __name__ == "__main__":
    print("Generating pendulum plots...")
    
    # Generate all plots
    plot_simple_pendulum()
    plot_damped_pendulum()
    plot_forced_pendulum()
    plot_forced_damped_pendulum_scenario1()
    plot_forced_damped_pendulum_scenario2()
    
    print("All pendulum plots generated successfully!")
