import numpy as np
import matplotlib.pyplot as plt

def rubber_bug_simulation(
    V=1.0,
    v=1.0,
    L0=1.0,
    dt=0.001,
    t_max=10.0,
    plot=True,
    ax=None,
):
    """
    Simulate the bug on a stretching rubber band.

    Parameters:
    V - speed of the end of the rubber band (m/s)
    v - speed of the bug relative to the rubber band (m/s)
    L0 - initial length of the rubber band (m)
    dt - time step for simulation (s)
    t_max - maximum simulation time (s)
    plot - if True, plot distance vs time
    ax - optional matplotlib axis to draw on
    """

    if dt <= 0:
        raise ValueError("dt must be > 0")
    if t_max <= 0:
        raise ValueError("t_max must be > 0")
    if L0 <= 0:
        raise ValueError("L0 must be > 0")

    # Include the endpoint so short runs still have at least two samples.
    times = np.arange(0.0, t_max + dt, dt)
    n = len(times)
    if n < 2:
        raise ValueError("Choose larger t_max or smaller dt")

    # Initialize arrays to store results.
    L = L0 + V * times
    if np.any(L <= 0):
        raise ValueError("Rubber band length became non-positive; choose different V/t_max")

    x_bug = np.zeros_like(times)  # Bug's position
    distance = np.zeros_like(times)  # Distance from bug to end
    distance[0] = L[0]

    reached = False
    reach_time = None

    # Simulation loop
    for i in range(1, n):
        # Calculate bug's new position.
        # The bug moves relative to the stretching rubber band
        dx_bug = (v + x_bug[i - 1] * V / L[i - 1]) * dt
        x_bug[i] = x_bug[i - 1] + dx_bug

        # Calculate distance to end
        distance[i] = L[i] - x_bug[i]

        # Check if bug has reached the end
        if distance[i] <= 0:
            # Linear interpolation gives a better crossing estimate.
            d0 = distance[i - 1]
            d1 = distance[i]
            t0 = times[i - 1]
            t1 = times[i]
            frac = d0 / (d0 - d1) if d0 != d1 else 1.0
            reach_time = t0 + frac * (t1 - t0)
            print(f"Bug reached the end at t = {reach_time:.6f} seconds")
            reached = True
            break

    if reached:
        # Keep computed samples up to crossing and append the interpolated endpoint.
        times_out = np.append(times[: i + 1], reach_time)
        L_out = np.append(L[: i + 1], L0 + V * reach_time)
        x_bug_out = np.append(x_bug[: i + 1], L0 + V * reach_time)
        distance_out = np.append(distance[: i + 1], 0.0)
    else:
        print("Bug did not reach the end within t_max")
        times_out = times
        L_out = L
        x_bug_out = x_bug
        distance_out = distance

    if plot:
        created_ax = ax is None
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))
        ax.plot(times_out, distance_out)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Distance to end (m)")
        ax.set_title("Distance between Bug and End of Rubber Band")
        ax.grid(True)
        if created_ax:
            plt.show()
        else:
            plt.draw()

    return {
        "times": times_out,
        "length": L_out,
        "x_bug": x_bug_out,
        "distance": distance_out,
        "reached": reached,
        "reach_time": reach_time,
    }

# Run the simulation
if __name__ == "__main__":
    rubber_bug_simulation()
