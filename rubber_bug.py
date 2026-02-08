from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class SimulationResult:
    times: np.ndarray
    length: np.ndarray
    x_bug: np.ndarray
    distance: np.ndarray
    reached: bool
    reach_time_numeric: Optional[float]
    reach_time_theory: Optional[float]


def theoretical_reach_time(V: float, v: float, L0: float) -> Optional[float]:
    """
    Closed-form reach time for a stretching rubber band.

    Assumptions:
    - band stretches uniformly with L(t) = L0 + V*t
    - V >= 0, L0 > 0, v > 0
    """
    if L0 <= 0:
        raise ValueError("L0 must be > 0")
    if V < 0:
        raise ValueError("V must be >= 0 for this model")
    if v <= 0:
        return None

    if np.isclose(V, 0.0):
        return L0 / v
    return (L0 / V) * (np.exp(V / v) - 1.0)


def rubber_bug_simulation(
    V: float = 1.0,
    v: float = 1.0,
    L0: float = 1.0,
    dt: float = 0.001,
    t_max: Optional[float] = None,
    plot: bool = True,
    ax=None,
    verbose: bool = True,
) -> SimulationResult:
    """
    Simulate a bug crawling on a uniformly stretching rubber band.

    Returns both numeric and theoretical reach times.
    """
    if dt <= 0:
        raise ValueError("dt must be > 0")
    if L0 <= 0:
        raise ValueError("L0 must be > 0")
    if V < 0:
        raise ValueError("V must be >= 0 for this model")
    if v < 0:
        raise ValueError("v must be >= 0")

    t_theory = theoretical_reach_time(V, v, L0)

    if t_max is None:
        if t_theory is None:
            t_max = 10.0
        else:
            t_max = 1.2 * t_theory
    if t_max <= 0:
        raise ValueError("t_max must be > 0")

    times = np.arange(0.0, t_max + dt, dt)
    length = L0 + V * times
    x_bug = np.zeros_like(times)
    distance = np.zeros_like(times)
    distance[0] = L0

    reached = False
    t_numeric = None

    for i in range(1, len(times)):
        # Euler integration of:
        # dx/dt = v + (x/L)*dL/dt = v + x*V/L
        x_bug[i] = x_bug[i - 1] + (v + x_bug[i - 1] * V / length[i - 1]) * dt
        distance[i] = length[i] - x_bug[i]

        if distance[i] <= 0:
            d0 = distance[i - 1]
            d1 = distance[i]
            t0 = times[i - 1]
            t1 = times[i]
            frac = d0 / (d0 - d1) if d0 != d1 else 1.0
            t_numeric = t0 + frac * (t1 - t0)
            reached = True
            break

    if reached:
        times_out = np.append(times[: i + 1], t_numeric)
        length_out = np.append(length[: i + 1], L0 + V * t_numeric)
        x_bug_out = np.append(x_bug[: i + 1], L0 + V * t_numeric)
        distance_out = np.append(distance[: i + 1], 0.0)
    else:
        times_out = times
        length_out = length
        x_bug_out = x_bug
        distance_out = distance

    if verbose:
        if reached:
            msg = f"Bug reached the end at t = {t_numeric:.6f} s (numeric)"
            if t_theory is not None:
                rel_err = abs(t_numeric - t_theory) / t_theory
                msg += f"; theory = {t_theory:.6f} s; rel err = {rel_err:.2e}"
            print(msg)
        else:
            if t_theory is None:
                print("Bug cannot reach the end for v <= 0")
            else:
                print(f"Bug did not reach the end within t_max={t_max:.6f} s")

    if plot:
        created_ax = ax is None
        if created_ax:
            _, ax = plt.subplots(figsize=(10, 6))
        ax.plot(times_out, distance_out, label="Distance to end")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Distance (m)")
        ax.set_title("Bug on Stretching Rubber Band")
        ax.grid(True)
        ax.legend()
        if created_ax:
            backend = plt.get_backend().lower()
            if "agg" in backend:
                plt.close(ax.figure)
            else:
                plt.show()
        else:
            plt.draw()

    return SimulationResult(
        times=times_out,
        length=length_out,
        x_bug=x_bug_out,
        distance=distance_out,
        reached=reached,
        reach_time_numeric=t_numeric,
        reach_time_theory=t_theory,
    )


if __name__ == "__main__":
    rubber_bug_simulation()
