import numpy as np
import matplotlib.pyplot as plt

import rat_strategies

def plot_simulation(sim_result, title=None):
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(5, 8))
    axs[0].plot(sim_result["time"], np.rad2deg(sim_result["theta"]))
    axs[0].set_ylabel("Theta (degrees)")
    axs[0].set_ylim(0, 100)
    #axs[0].axhline(80, lw=1, ls='--', color='C1')
    #axs[0].axhline(75, lw=1, ls='--', color='C1')
    #axs[0].axhline(30, lw=1, ls='--', color='C2')
    axs[1].plot(sim_result["time"], np.rad2deg(sim_result["theta_dot"]))
    axs[1].set_ylabel("Theta vel\n(degrees/s)")
    axs[2].plot(sim_result["time"], np.rad2deg(sim_result["theta_dotdot"]))
    axs[2].set_ylabel("Theta acc\n(degrees/s^2)")
    axs[2].set_xlabel("Time (s)")
    if title is not None:
        fig.suptitle(title)

sim = rat_strategies.FixedForce(
        lever_mass = 3.5e-3,    #kg
        lever_length = 15e-2,   #meter
        lever_range = (30, 100), #degrees
        motor_baseline_torque = 3e-3,
        motor_extra_torque = 2e-2,
        motor_onset_angle = (75, 80),
        friction_coeff = 0.0e-3,
        fixed_force = -0.2
)
sim_result = sim.simulate_trial()
plot_simulation(sim_result, "Fixed force")


sim = rat_strategies.TwoStepForce(
        lever_mass = 3.5e-3,    #kg
        lever_length = 15e-2,   #meter
        lever_range = (30, 100), #degrees
        motor_baseline_torque = 3e-3,
        motor_extra_torque = 2e-2,
        motor_onset_angle = (75, 80),
        friction_coeff = 0.0e-3,
        force_1 = -0.1,
        force_2 = -0.4,
        switch_ang = 70
)
sim_result = sim.simulate_trial()
plot_simulation(sim_result, f"Two-step force\n(switch at {np.rad2deg(sim.switch_rad)} degrees)")

