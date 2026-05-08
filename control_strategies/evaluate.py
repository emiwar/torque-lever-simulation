import jax
import jax.numpy as jp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from simulation import TorqueLeverSimulationJAX, SIM_PARAMS

# Evaluation target: the lever's minimum angle should land near TARGET_DEG.
# Score per trial is 1 if the minimum is within BOUND_DEG, falling linearly
# to 0 further away. Total score = sum over all evaluated torque levels.
TARGET_DEG = 67.5
BOUND_DEG  = 5.0

# Range of motor extra-torques (added on top of baseline torque)
EXTRA_TORQUE_MIN = -0.05   # Nm
EXTRA_TORQUE_MAX =  0.15   # Nm

EVAL_TORQUES = jp.arange(EXTRA_TORQUE_MIN, EXTRA_TORQUE_MAX, 0.005)   # 40 levels, fine grid
PLOT_TORQUES = jp.arange(EXTRA_TORQUE_MIN, EXTRA_TORQUE_MAX, 0.02)    # 10 levels, one curve each

EVAL_DURATION = 1.0   # seconds
PLOT_DURATION = 1.0   # seconds


def _make_sim(extra_torque):
    return TorqueLeverSimulationJAX(**dict(SIM_PARAMS, motor_extra_torque=extra_torque))


def evaluate_strategy(strategy):
    """Score a strategy across the range of motor torques. Higher is better.

    Returns the sum of per-trial reward scores. Maximum = len(EVAL_TORQUES).
    """
    target = jp.deg2rad(TARGET_DEG)
    bound  = jp.deg2rad(BOUND_DEG)

    def eval_one(extra_torque):
        sim = _make_sim(extra_torque)
        thetas = sim.run(strategy, duration=EVAL_DURATION)
        min_theta = jp.min(thetas)
        reward = 1.0 - jp.abs(min_theta - target) / bound
        return jp.clip(reward, 0.0, 1.0)

    return jp.sum(jax.vmap(eval_one)(EVAL_TORQUES))


def sample_trajectories(strategy, duration=PLOT_DURATION):
    """Run the strategy under each torque in PLOT_TORQUES.

    Returns an array of shape (n_torques, n_steps) with lever angles in radians.
    """
    def run_one(extra_torque):
        sim = _make_sim(extra_torque)
        return sim.run(strategy, duration=duration)

    return jax.vmap(run_one)(PLOT_TORQUES)


def plot_trajectory_panel(ax, thetas, title):
    """Plot a set of trajectories on ax, coloured by total motor torque.

    thetas : array of shape (n_torques, n_steps), angles in radians
    Returns a ScalarMappable suitable for adding a shared colorbar.
    """
    n_torques, n_steps = thetas.shape
    seconds = np.arange(n_steps) * SIM_PARAMS['dt']
    total_torques = np.array(PLOT_TORQUES) + SIM_PARAMS['motor_baseline_torque']

    for i in range(n_torques):
        color = plt.cm.viridis(i / (n_torques - 1))
        ax.plot(seconds, np.rad2deg(np.array(thetas[i])), color=color, linewidth=0.9)

    ax.axhline(TARGET_DEG, color='black', linewidth=0.8, linestyle='--', alpha=0.4,
               label=f'Target ({TARGET_DEG}°)')
    ax.set_xlim(0, seconds[-1])
    ax.set_ylim(45, 100)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Lever angle (degrees)')
    ax.set_title(title)
    sns.despine(ax=ax)

    norm = plt.Normalize(vmin=total_torques.min(), vmax=total_torques.max())
    return plt.cm.ScalarMappable(norm=norm, cmap='viridis')
