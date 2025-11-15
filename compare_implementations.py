import numpy as np
import jax
import jax.numpy as jp
from fit_force import TorqueLeverSimulationJAX
from rat_strategies import DummyForce

# Run numpy implementation
numpy_sim = DummyForce()
result_dict = numpy_sim.simulate_trial(duration=0.5)
numpy_thetas = result_dict["theta"]

# Extract dummy force for JAX
dummy_force = -np.sin(4*np.linspace(0, 1, 500))**2*np.linspace(0, 1, 500) + 0.1*np.linspace(0, 1, 500)**2
# Truncate to match simulation duration
n_steps = int(0.5 / 1e-3)
jax_dummy_force = dummy_force[:n_steps]

# Run JAX implementation
jax_sim = TorqueLeverSimulationJAX(lever_mass = 3.5e-3,
                                   lever_length = 15e-2,
                                   lever_range = (30, 100),
                                   motor_baseline_torque = 3e-3,
                                   motor_extra_torque = 2e-2,
                                   motor_onset_angle = (75, 80),
                                   friction_coeff = 0.0e-3,
                                   dt=1e-3)

# Call eval_candidate with the dummy force
start_theta = jax_sim.lever_max - jp.deg2rad(5)  # Same as numpy reset
loss, jax_thetas = jax_sim.eval_candidate((jax_dummy_force, start_theta, jp.array(0.0)), 
                                           jax_dummy_force)

# Convert JAX arrays to numpy for comparison
jax_thetas = np.array(jax_thetas)
numpy_thetas_trimmed = numpy_thetas[:len(jax_thetas)]

print("NumPy simulation shape:", numpy_thetas_trimmed.shape)
print("JAX simulation shape:", jax_thetas.shape)
print("\nFirst 10 theta values (NumPy):", numpy_thetas_trimmed[:10])
print("First 10 theta values (JAX):", jax_thetas[:10])
print("\nLast 10 theta values (NumPy):", numpy_thetas_trimmed[-10:])
print("Last 10 theta values (JAX):", jax_thetas[-10:])
print("\nMax difference:", np.max(np.abs(numpy_thetas_trimmed - jax_thetas)))
print("Mean difference:", np.mean(np.abs(numpy_thetas_trimmed - jax_thetas)))
