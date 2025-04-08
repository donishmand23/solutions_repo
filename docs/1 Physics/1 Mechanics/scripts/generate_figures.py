import numpy as np
import matplotlib.pyplot as plt

# Figure 3: Range vs Angle
angles = np.linspace(0, 90, 91)
v0 = 10  # m/s
g = 9.8  # m/s²

ranges = (v0**2 * np.sin(2 * angles * np.pi/180)) / g

plt.figure(figsize=(10, 6))
plt.plot(angles, ranges)
plt.xlabel('Angle (degrees)')
plt.ylabel('Range (m)')
plt.title('Range vs Angle of Projection')
plt.grid(True)
plt.savefig('docs/1 Physics/1 Mechanics/range_vs_angle.png', dpi=300)

# Figure 2: Range vs Initial Velocity
velocities = np.linspace(5, 30, 100)
angles_deg = [15, 30, 45, 60, 75]

plt.figure(figsize=(10, 6))
for angle in angles_deg:
    angle_rad = angle * np.pi / 180
    ranges = (velocities**2 * np.sin(2 * angle_rad)) / g
    plt.plot(velocities, ranges, label=f'{angle}°')

plt.xlabel('Initial Velocity (m/s)')
plt.ylabel('Range (m)')
plt.title('Range vs Initial Velocity for Different Angles')
plt.grid(True)
plt.legend()
plt.savefig('docs/1 Physics/1 Mechanics/range_vs_velocity.png', dpi=300)

# Figure 1: Projectile Motion Trajectory
angle_deg = 45
v0 = 10
angle_rad = angle_deg * np.pi / 180
vx = v0 * np.cos(angle_rad)
vy = v0 * np.sin(angle_rad)

# Time of flight
t_flight = 2 * vy / g
t = np.linspace(0, t_flight, 100)

x = vx * t
y = vy * t - 0.5 * g * t**2

plt.figure(figsize=(10, 6))
plt.plot(x, y)
plt.grid(True)
plt.xlabel('Horizontal Distance (m)')
plt.ylabel('Height (m)')
plt.title(f'Projectile Motion Trajectory (v₀={v0} m/s, θ={angle_deg}°)')
plt.axis('equal')
# Add arrows to show initial velocity
plt.arrow(0, 0, vx/2, vy/2, head_width=0.5, head_length=0.8, fc='red', ec='red')
plt.text(vx/2 + 0.5, vy/2, r'$v_0$', fontsize=12)
plt.savefig('docs/1 Physics/1 Mechanics/projectile_motion_diagram.png', dpi=300)

# Figure 4: Range vs Angle with Different Parameters
plt.figure(figsize=(10, 6))

# Ideal case
ranges_ideal = (v0**2 * np.sin(2 * angles * np.pi/180)) / g
plt.plot(angles, ranges_ideal, label='Ideal')

# With air resistance (simplified model)
# Simulate with a simple drag coefficient
def simulate_with_drag(v0, angle_deg, drag_coef=0.01):
    angle_rad = angle_deg * np.pi / 180
    vx = v0 * np.cos(angle_rad)
    vy = v0 * np.sin(angle_rad)
    x, y = 0, 0
    dt = 0.01
    
    while y >= 0:
        v = np.sqrt(vx**2 + vy**2)
        drag_x = -drag_coef * v * vx
        drag_y = -drag_coef * v * vy
        
        vx += drag_x * dt
        vy += (drag_y - g) * dt
        
        x += vx * dt
        y += vy * dt
        
        if y < 0:
            # Linear interpolation to find exact range
            t_frac = -y / (vy * dt)
            x -= vx * dt * (1 - t_frac)
            break
            
    return x

ranges_drag = [simulate_with_drag(v0, angle) for angle in angles]
plt.plot(angles, ranges_drag, label='With Air Resistance')

# From elevated position
h = 10  # meters
def range_from_height(v0, angle_deg, height):
    angle_rad = angle_deg * np.pi / 180
    vx = v0 * np.cos(angle_rad)
    vy = v0 * np.sin(angle_rad)
    
    # Time to hit ground from height h
    # Solve: h + vy*t - 0.5*g*t^2 = 0
    # Using quadratic formula
    a = -0.5 * g
    b = vy
    c = height
    
    # We want the positive root (future time)
    t = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)
    
    return vx * t

ranges_height = [range_from_height(v0, angle, h) for angle in angles]
plt.plot(angles, ranges_height, label=f'From Height {h}m')

plt.xlabel('Angle (degrees)')
plt.ylabel('Range (m)')
plt.title('Range vs Angle Under Different Conditions')
plt.grid(True)
plt.legend()
plt.savefig('docs/1 Physics/1 Mechanics/range_vs_angle_parameters.png', dpi=300)

print("All figures have been generated and saved to the docs/1 Physics/1 Mechanics folder.")
