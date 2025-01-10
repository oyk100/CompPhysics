import numpy as np
import matplotlib.pyplot as plt

def rubber_bug_simulation(V=1.0, v=1.0, L0=1.0, dt=0.001, t_max=10.0):
    """
    Simulate the bug on a stretching rubber band.
    
    Parameters:
    V - speed of the end of the rubber band (m/s)
    v - speed of the bug relative to the rubber band (m/s)
    L0 - initial length of the rubber band (m)
    dt - time step for simulation (s)
    t_max - maximum simulation time (s)
    """
    
    # Initialize arrays to store results
    times = np.arange(0, t_max, dt)
    L = np.zeros_like(times)  # Length of rubber band
    x_bug = np.zeros_like(times)  # Bug's position
    distance = np.zeros_like(times)  # Distance from bug to end
    
    # Initial conditions
    L[0] = L0
    x_bug[0] = 0.0  # Bug starts at wall
    distance[0] = L0
    
    # Simulation loop
    for i in range(1, len(times)):
        t = times[i]
        
        # Update rubber band length
        L[i] = L0 + V * t
        
        # Calculate bug's new position
        # The bug moves relative to the stretching rubber band
        dx_bug = (v + x_bug[i-1] * V/L[i-1]) * dt
        x_bug[i] = x_bug[i-1] + dx_bug
        
        # Calculate distance to end
        distance[i] = L[i] - x_bug[i]
        
        # Check if bug has reached the end
        if distance[i] <= 0:
            print(f"Bug reached the end at t = {t:.2f} seconds")
            break
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(times[:i+1], distance[:i+1])
    plt.xlabel('Time (s)')
    plt.ylabel('Distance to end (m)')
    plt.title('Distance between Bug and End of Rubber Band')
    plt.grid(True)
    plt.show()

# Run the simulation
rubber_bug_simulation()
