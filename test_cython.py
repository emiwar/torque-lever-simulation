import numpy as np
from torque_lever_cython import CythonTorqueLeverStep, fit_force_cython
import fit_force
from rat_strategies import DummyForce

# Create test data using the actual DummyForce simulation
dummy_force_sim = DummyForce()
result_dict = dummy_force_sim.simulate_trial(duration=0.5)
ref_thetas = result_dict["theta"]
dummy_force = dummy_force_sim.dummy_force[:len(ref_thetas)]

# Initialize Cython step function
cython_step = CythonTorqueLeverStep(
    lever_mass=3.5e-3,
    lever_length=15e-2,
    lever_range=(30, 100),
    motor_baseline_torque=3e-3,
    motor_extra_torque=2e-2,
    motor_onset_angle=(75, 80),
    friction_coeff=0.0e-3,
    dt=1e-3
)

# Test single step
theta = 1.655
theta_dot = 0.0
force = dummy_force[0]

print("Testing single step function:")
result = cython_step.step(theta, theta_dot, force)
print(f"Input: theta={theta:.6f}, theta_dot={theta_dot:.6f}, force={force:.6f}")
print(f"Output: theta={result[0]:.6f}, theta_dot={result[1]:.6f}")

# Test dynamic programming fit
print("\nTesting dynamic programming fit:")
try:
    force_est = fit_force_cython(ref_thetas, cython_step)
    print(f"Force estimation completed successfully!")
    print(f"Estimated force shape: {force_est.shape}")
    print(f"First 10 estimated forces: {force_est[:10]}")
    print(f"Last 10 estimated forces: {force_est[-10:]}")
except Exception as e:
    print(f"Error during fit: {e}")
    import traceback
    traceback.print_exc()
