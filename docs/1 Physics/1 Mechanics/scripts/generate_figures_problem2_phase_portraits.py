#!/usr/bin/env python3
"""
Script to generate pendulum motion figures for Problem 2, including time series and phase portraits.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os

# Create figures directory if it doesn't exist
figures_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'figures')
os.makedirs(figures_dir, exist_ok=True)

# Common parameters
g = 9.81  # gravitational acceleration (m/s^2)
L = 1.0   # pendulum length (m)
omega0_squared = g/L  # natural frequency squared
omega0 = np.sqrt(omega0_squared)  # natural frequency

# Time settings
t_max = 40  # maximum simulation time (s)
dt = 0.05   # time step for plotting

# 1. Simple Pendulum (no damping, no external force)
def simple_pendulum(t, y):
    """ODE for simple pendulum motion (no damping, no external force)"""
    theta, omega = y
    dtheta_dt = omega
    domega_dt = -omega0_squared * np.sin(theta)
    return [dtheta_dt, domega_dt]

def plot_simple_pendulum():
    """Generate time series and phase portrait for simple pendulum"""
    print("Generating simple pendulum plots...")
    
    # Initial conditions: theta = 30 degrees, omega = 0
    y0 = [np.radians(30), 0]
    
    # Solve the ODE
    t_span = (0, t_max)
    t_eval = np.arange(0, t_max, dt)
    sol = solve_ivp(simple_pendulum, t_span, y0, method='RK45', t_eval=t_eval)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Time series plot
    ax1.plot(sol.t, sol.y[0], 'r-', linewidth=1.5)
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('Angle (rad)', fontsize=12)
    ax1.set_title('Time Series: Simple Pendulum', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_xlim(0, t_max)
    max_angle = np.max(np.abs(sol.y[0])) * 1.1
    ax1.set_ylim(-max_angle, max_angle)
    
    # Phase portrait
    ax2.plot(sol.y[0], sol.y[1], 'r-', linewidth=1.5)
    ax2.set_xlabel('Angle (rad)', fontsize=12)
    ax2.set_ylabel('Angular Velocity (rad/s)', fontsize=12)
    ax2.set_title('Phase Portrait: Simple Pendulum', fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.7)
    max_velocity = np.max(np.abs(sol.y[1])) * 1.1
    ax2.set_xlim(-max_angle, max_angle)
    ax2.set_ylim(-max_velocity, max_velocity)
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'simple_pendulum_phase.png'), dpi=300)
    plt.close()

# 2. Damped Pendulum
def damped_pendulum(t, y, b=0.2):
    """ODE for damped pendulum motion (with damping, no external force)"""
    theta, omega = y
    dtheta_dt = omega
    domega_dt = -b * omega - omega0_squared * np.sin(theta)
    return [dtheta_dt, domega_dt]

def plot_damped_pendulum():
    """Generate time series and phase portrait for damped pendulum"""
    print("Generating damped pendulum plots...")
    
    # Initial conditions: theta = 30 degrees, omega = 0
    y0 = [np.radians(30), 0]
    
    # Damping coefficient
    b = 0.2
    
    # Solve the ODE
    t_span = (0, t_max)
    t_eval = np.arange(0, t_max, dt)
    sol = solve_ivp(lambda t, y: damped_pendulum(t, y, b), t_span, y0, method='RK45', t_eval=t_eval)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Time series plot
    ax1.plot(sol.t, sol.y[0], 'b-', linewidth=1.5)
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('Angle (rad)', fontsize=12)
    ax1.set_title(f'Time Series: Damped Pendulum (b={b})', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_xlim(0, t_max)
    max_angle = np.max(np.abs(sol.y[0])) * 1.1
    ax1.set_ylim(-max_angle, max_angle)
    
    # Phase portrait
    ax2.plot(sol.y[0], sol.y[1], 'b-', linewidth=1.5)
    ax2.set_xlabel('Angle (rad)', fontsize=12)
    ax2.set_ylabel('Angular Velocity (rad/s)', fontsize=12)
    ax2.set_title(f'Phase Portrait: Damped Pendulum (b={b})', fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.7)
    max_velocity = np.max(np.abs(sol.y[1])) * 1.1
    ax2.set_xlim(-max_angle, max_angle)
    ax2.set_ylim(-max_velocity, max_velocity)
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'damped_pendulum_phase.png'), dpi=300)
    plt.close()

# 3. Forced Pendulum (no damping)
def forced_pendulum(t, y, A=0.5, Omega=0.667*omega0):
    """ODE for forced pendulum motion (no damping, with external force)"""
    theta, omega = y
    dtheta_dt = omega
    domega_dt = -omega0_squared * np.sin(theta) + A * np.cos(Omega * t)
    return [dtheta_dt, domega_dt]

def plot_forced_pendulum():
    """Generate time series and phase portrait for forced pendulum"""
    print("Generating forced pendulum plots...")
    
    # Initial conditions: theta = 0, omega = 0
    y0 = [0, 0]
    
    # Force parameters
    A = 0.5  # amplitude
    Omega = 0.667 * omega0  # driving frequency
    
    # Solve the ODE
    t_span = (0, t_max)
    t_eval = np.arange(0, t_max, dt)
    sol = solve_ivp(lambda t, y: forced_pendulum(t, y, A, Omega), t_span, y0, method='RK45', t_eval=t_eval)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Time series plot
    ax1.plot(sol.t, sol.y[0], 'g-', linewidth=1.5)
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('Angle (rad)', fontsize=12)
    ax1.set_title(f'Time Series: Forced Pendulum (A={A}, Ω={Omega:.2f})', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_xlim(0, t_max)
    max_angle = np.max(np.abs(sol.y[0])) * 1.1
    ax1.set_ylim(-max_angle, max_angle)
    
    # Phase portrait
    ax2.plot(sol.y[0], sol.y[1], 'g-', linewidth=1.5)
    ax2.set_xlabel('Angle (rad)', fontsize=12)
    ax2.set_ylabel('Angular Velocity (rad/s)', fontsize=12)
    ax2.set_title(f'Phase Portrait: Forced Pendulum (A={A}, Ω={Omega:.2f})', fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.7)
    max_velocity = np.max(np.abs(sol.y[1])) * 1.1
    ax2.set_xlim(-max_angle, max_angle)
    ax2.set_ylim(-max_velocity, max_velocity)
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'forced_pendulum_phase.png'), dpi=300)
    plt.close()

# Combined plot showing all three scenarios
def plot_combined_pendulum_scenarios():
    """Generate a combined plot showing all three pendulum scenarios"""
    print("Generating combined pendulum scenarios plot...")
    
    # Initial conditions: theta = 30 degrees (or 0 for forced), omega = 0
    y0_simple = [np.radians(30), 0]
    y0_damped = [np.radians(30), 0]
    y0_forced = [0, 0]
    
    # Parameters
    b = 0.2  # damping coefficient
    A = 0.5  # force amplitude
    Omega = 0.667 * omega0  # driving frequency
    
    # Time settings
    t_span = (0, t_max)
    t_eval = np.arange(0, t_max, dt)
    
    # Solve ODEs
    sol_simple = solve_ivp(simple_pendulum, t_span, y0_simple, method='RK45', t_eval=t_eval)
    sol_damped = solve_ivp(lambda t, y: damped_pendulum(t, y, b), t_span, y0_damped, method='RK45', t_eval=t_eval)
    sol_forced = solve_ivp(lambda t, y: forced_pendulum(t, y, A, Omega), t_span, y0_forced, method='RK45', t_eval=t_eval)
    
    # Create figure with 3 rows and 2 columns
    fig, axs = plt.subplots(3, 2, figsize=(12, 15))
    
    # Set common limits for better comparison
    # For time series plots
    max_time = t_max
    max_angle = max(
        np.max(np.abs(sol_simple.y[0])),
        np.max(np.abs(sol_damped.y[0])),
        np.max(np.abs(sol_forced.y[0]))
    ) * 1.1
    
    # For phase portraits
    max_velocity = max(
        np.max(np.abs(sol_simple.y[1])),
        np.max(np.abs(sol_damped.y[1])),
        np.max(np.abs(sol_forced.y[1]))
    ) * 1.1
    
    # 1. Simple Pendulum
    axs[0, 0].plot(sol_simple.t, sol_simple.y[0], 'r-', linewidth=1.5)
    axs[0, 0].set_xlabel('Time (s)', fontsize=12)
    axs[0, 0].set_ylabel('Angle (rad)', fontsize=12)
    axs[0, 0].set_title('1) Simple Pendulum', fontsize=14)
    axs[0, 0].grid(True, linestyle='--', alpha=0.7)
    axs[0, 0].set_xlim(0, max_time)
    axs[0, 0].set_ylim(-max_angle, max_angle)
    
    axs[0, 1].plot(sol_simple.y[0], sol_simple.y[1], 'r-', linewidth=1.5)
    axs[0, 1].set_xlabel('Angle (rad)', fontsize=12)
    axs[0, 1].set_ylabel('Angular Velocity (rad/s)', fontsize=12)
    axs[0, 1].set_title('Phase Portrait', fontsize=14)
    axs[0, 1].grid(True, linestyle='--', alpha=0.7)
    axs[0, 1].set_xlim(-max_angle, max_angle)
    axs[0, 1].set_ylim(-max_velocity, max_velocity)
    
    # 2. Damped Pendulum
    axs[1, 0].plot(sol_damped.t, sol_damped.y[0], 'b-', linewidth=1.5)
    axs[1, 0].set_xlabel('Time (s)', fontsize=12)
    axs[1, 0].set_ylabel('Angle (rad)', fontsize=12)
    axs[1, 0].set_title(f'2) Damped Pendulum (b={b})', fontsize=14)
    axs[1, 0].grid(True, linestyle='--', alpha=0.7)
    axs[1, 0].set_xlim(0, max_time)
    axs[1, 0].set_ylim(-max_angle, max_angle)
    
    axs[1, 1].plot(sol_damped.y[0], sol_damped.y[1], 'b-', linewidth=1.5)
    axs[1, 1].set_xlabel('Angle (rad)', fontsize=12)
    axs[1, 1].set_ylabel('Angular Velocity (rad/s)', fontsize=12)
    axs[1, 1].set_title('Phase Portrait', fontsize=14)
    axs[1, 1].grid(True, linestyle='--', alpha=0.7)
    axs[1, 1].set_xlim(-max_angle, max_angle)
    axs[1, 1].set_ylim(-max_velocity, max_velocity)
    
    # 3. Forced Pendulum
    axs[2, 0].plot(sol_forced.t, sol_forced.y[0], 'g-', linewidth=1.5)
    axs[2, 0].set_xlabel('Time (s)', fontsize=12)
    axs[2, 0].set_ylabel('Angle (rad)', fontsize=12)
    axs[2, 0].set_title(f'3) Forced Pendulum (A={A}, Ω={Omega:.2f})', fontsize=14)
    axs[2, 0].grid(True, linestyle='--', alpha=0.7)
    axs[2, 0].set_xlim(0, max_time)
    axs[2, 0].set_ylim(-max_angle, max_angle)
    
    axs[2, 1].plot(sol_forced.y[0], sol_forced.y[1], 'g-', linewidth=1.5)
    axs[2, 1].set_xlabel('Angle (rad)', fontsize=12)
    axs[2, 1].set_ylabel('Angular Velocity (rad/s)', fontsize=12)
    axs[2, 1].set_title('Phase Portrait', fontsize=14)
    axs[2, 1].grid(True, linestyle='--', alpha=0.7)
    axs[2, 1].set_xlim(-max_angle, max_angle)
    axs[2, 1].set_ylim(-max_velocity, max_velocity)
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'pendulum_combined_phase_portraits.png'), dpi=300)
    plt.close()

if __name__ == "__main__":
    print("Generating pendulum phase portraits...")
    
    # Generate only the combined plot
    plot_combined_pendulum_scenarios()
    
    print("Pendulum combined phase portraits generated successfully!")
