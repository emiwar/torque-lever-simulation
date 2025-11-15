import numpy as np
import jax
import jax.numpy as jp
from fit_force import TorqueLeverSimulationJAX
from rat_strategies import DummyForce

# Run numpy implementation with more details
numpy_sim = DummyForce()
result_dict = numpy_sim.simulate_trial(duration=0.5)
numpy_thetas = result_dict["theta"]
numpy_theta_dots = result_dict["theta_dot"]

# Extract dummy force for JAX
dummy_force = -np.sin(4*np.linspace(0, 1, 500))**2*np.linspace(0, 1, 500) + 0.1*np.linspace(0, 1, 500)**2
n_steps = int(0.5 / 1e-3)
jax_dummy_force = dummy_force[:n_steps]

# Run JAX implementation with debugging
jax_sim = TorqueLeverSimulationJAX(lever_mass = 3.5e-3,
                                   lever_length = 15e-2,
                                   lever_range = (30, 100),
                                   motor_baseline_torque = 3e-3,
                                   motor_extra_torque = 2e-2,
                                   motor_onset_angle = (75, 80),
                                   friction_coeff = 0.0e-3,
                                   dt=1e-3)

start_theta = jax_sim.lever_max - jp.deg2rad(5)
print("Initial theta (numpy):", numpy_sim.theta)
print("Initial theta (JAX):", start_theta)
print("Lever max:", jax_sim.lever_max, "rad =", np.rad2deg(jax_sim.lever_max), "deg")
print("Lever min:", jax_sim.lever_min, "rad =", np.rad2deg(jax_sim.lever_min), "deg")

loss, jax_thetas = jax_sim.eval_candidate((jax_dummy_force, start_theta, jp.array(0.0)), 
                                           jax_dummy_force)

jax_thetas = np.array(jax_thetas)
numpy_thetas_trimmed = numpy_thetas[:len(jax_thetas)]

# Look at where they diverge
print("\n--- Checking where they diverge ---")
for i in range(0, min(len(jax_thetas), 100), 10):
    diff = abs(jax_thetas[i] - numpy_thetas_trimmed[i])
    print(f"Step {i}: NumPy={numpy_thetas_trimmed[i]:.8f}, JAX={jax_thetas[i]:.8f}, diff={diff:.2e}, NumPy_dot={numpy_theta_dots[i]:.8f}")

print("\n--- Checking the clipping region ---")
for i in range(len(jax_thetas)-20, len(jax_thetas)):
    if i >= 0:
        in_bounds_np = numpy_thetas_trimmed[i] > numpy_sim.lever_min and numpy_thetas_trimmed[i] < numpy_sim.lever_max
        in_bounds_jax = jax_thetas[i] > jax_sim.lever_min and jax_thetas[i] < jax_sim.lever_max
        print(f"Step {i}: NumPy={numpy_thetas_trimmed[i]:.8f} (in_bounds={in_bounds_np}), JAX={jax_thetas[i]:.8f} (in_bounds={in_bounds_jax})")
