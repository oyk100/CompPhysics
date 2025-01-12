import numpy as np
import matplotlib.pyplot as plt

def simulate_chase(R=100, u=1, V=1, dt=0.01):
    """
    Simulate predator-prey chase where prey moves in a circle and predator chases directly.
    
    Parameters:
        R (float): Radius of prey's circular path (meters)
        u (float): Speed of prey (m/s)
        V (float): Speed of predator (m/s)
        dt (float): Time step for simulation (seconds)
        
    Returns:
        Tc (float): Time until capture (seconds)
        positions (list): List of (prey_pos, predator_pos) tuples over time
    """
    # Initial positions
    prey_pos = np.array([R, 0])  # Start prey at (R, 0)
    predator_pos = np.array([0, 0])  # Start predator at origin
    
    positions = [(prey_pos.copy(), predator_pos.copy())]
    t = 0
    
    while True:
        # Calculate prey's new position (circular motion)
        theta = u * t / R  # Angle based on time
        prey_pos = np.array([R * np.cos(theta), R * np.sin(theta)])
        
        # Calculate predator's direction vector
        direction = prey_pos - predator_pos
        distance = np.linalg.norm(direction)
        
        # Check for capture
        if distance < 0.1:  # Capture condition
            return t, positions
        
        # Normalize direction and move predator
        predator_pos += (direction / distance) * V * dt
        
        # Store positions
        positions.append((prey_pos.copy(), predator_pos.copy()))
        t += dt

def plot_trajectory(positions):
    """Plot the trajectories of prey and predator"""
    prey_x, prey_y = zip(*[p[0] for p in positions])
    pred_x, pred_y = zip(*[p[1] for p in positions])
    
    plt.figure(figsize=(8, 8))
    plt.plot(prey_x, prey_y, label='Prey Path')
    plt.plot(pred_x, pred_y, label='Predator Path')
    plt.scatter([0], [0], c='red', label='Start')
    plt.gca().set_aspect('equal')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.legend()
    plt.title('Predator-Prey Chase Trajectories')
    plt.show()

def analyze_capture_time(u_range=(0.1, 2.0), num_points=20):
    """Analyze how capture time varies with prey speed"""
    u_values = np.linspace(u_range[0], u_range[1], num_points)
    Tc_values = []
    
    for u in u_values:
        Tc, _ = simulate_chase(u=u)
        Tc_values.append(Tc)
    
    plt.figure()
    plt.plot(u_values, Tc_values, 'o-')
    plt.xlabel('Prey Speed u (m/s)')
    plt.ylabel('Capture Time Tc (s)')
    plt.title('Capture Time vs Prey Speed')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Run simulation with default parameters
    Tc, positions = simulate_chase()
    print(f"Capture time: {Tc:.2f} seconds")
    
    # Plot the trajectories
    plot_trajectory(positions)
    
    # Analyze how capture time varies with prey speed
    analyze_capture_time()
