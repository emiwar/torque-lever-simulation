import jax.numpy as jp
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import BSpline
import fit_force

NOISE_SIGMA = 0.005   # radians (~0.3 degrees)
SMOOTH_SIGMA = 20     # Gaussian pre-smoothing for analytical inversion (samples)
N_BASIS = 30          # number of cubic B-spline basis functions
N_STEPS = 500
AUTOREGRESSIVE_PENALTY = 1e-2

sim = fit_force.TorqueLeverSimulationJAX(
    lever_mass=3.5e-3,
    lever_length=15e-2,
    lever_range=(30, 100),
    motor_baseline_torque=3e-3,
    motor_extra_torque=2e-2,
    motor_onset_angle=(75, 80),
    friction_coeff=0.0e-3,
    autoregressive_penalty=AUTOREGRESSIVE_PENALTY,
    dt=1e-3,
)

tt = np.linspace(0, 1, N_STEPS)
true_force = jp.array(-np.sin(4 * tt) ** 2 * tt - 0.1 * tt ** 2)
true_theta = sim.run(true_force)

rng = np.random.default_rng(42)
noisy_theta = true_theta + jp.array(rng.normal(0.0, NOISE_SIGMA, true_theta.shape))

start_theta     = sim.lever_max - jp.deg2rad(5.0)
start_theta_dot = jp.array(0.0)

# Build cubic B-spline basis matrix: shape (N_STEPS, N_BASIS)
# Uniformly spaced interior knots with clamped endpoints (degree k=3)
k = 3
n_interior = N_BASIS - k - 1
interior_knots = np.linspace(tt[0], tt[-1], n_interior + 2)[1:-1]
knots = np.concatenate([[tt[0]] * (k + 1), interior_knots, [tt[-1]] * (k + 1)])
basis = BSpline.design_matrix(tt, knots, k).toarray()   # (N_STEPS, N_BASIS)

# Analytical inversion (smooth noisy theta first)
smoothed_theta = jp.array(gaussian_filter1d(np.array(noisy_theta), sigma=SMOOTH_SIGMA))
F_analytical = sim.analytical_inversion(smoothed_theta, start_theta=start_theta)

# Initialise basis coefficients by projecting analytical estimate onto the basis
coeffs_init = jp.array(np.linalg.lstsq(basis, np.array(F_analytical), rcond=None)[0])

# Basis fit — L-BFGS, warm-started from analytical projection
print("Basis fit (L-BFGS, warm start from analytical)...")
losses_basis, F_basis = sim.fit_force_basis(
    noisy_theta,
    start_theta=start_theta,
    start_theta_dot=start_theta_dot,
    basis=basis,
    max_iter=200,
    coeffs_init=coeffs_init,
)

def rmse(a, b):
    return float(jp.sqrt(jp.mean(jp.square(a - b))))

# GP fit — linearised around basis estimate
print("\nGP fit (linearised around basis fit)...")
F_gp_mean, F_gp_std = sim.fit_force_gp(
    noisy_theta,
    start_theta=start_theta,
    start_theta_dot=start_theta_dot,
    tt=tt,
    sigma_obs=NOISE_SIGMA,
    forces_linearize_around=F_basis,
)

print(f"\nNoise sigma:                    {NOISE_SIGMA:.4f} rad")
print(f"RMSE analytical (smoothed):     {rmse(F_analytical, true_force):.4f} N")
print(f"RMSE basis fit  ({N_BASIS} splines): {rmse(F_basis,     true_force):.4f} N")
print(f"RMSE GP mean:                   {rmse(F_gp_mean,   true_force):.4f} N")

_, theta_basis = sim.eval_candidate(F_basis,    noisy_theta, start_theta, start_theta_dot)
_, theta_gp    = sim.eval_candidate(F_gp_mean,  noisy_theta, start_theta, start_theta_dot)

fig, axes = plt.subplots(3, 1, figsize=(10, 12))

# Force panel — all three estimates + GP confidence interval
ax = axes[0]
ax.plot(tt, true_force,   label='True force',                                          color='black',      linewidth=2)
ax.plot(tt, F_basis,      label=f'Basis fit ({N_BASIS} splines)  RMSE={rmse(F_basis,    true_force):.3f}', color='tomato',     linewidth=1.5)
ax.plot(tt, F_gp_mean,    label=f'GP mean                 RMSE={rmse(F_gp_mean, true_force):.3f}', color='seagreen',   linewidth=1.5)
ax.fill_between(tt,
                F_gp_mean - 1.96 * F_gp_std,
                F_gp_mean + 1.96 * F_gp_std,
                color='seagreen', alpha=0.2, label='GP 95% CI')
ax.set_ylabel('Force (N)')
ax.set_title(f'Force recovery  [noise σ={NOISE_SIGMA} rad]')
ax.legend(fontsize=8)

# Theta panel
ax = axes[1]
ax.plot(tt, jp.rad2deg(true_theta),  label='True θ (clean)',    color='black',    linewidth=2)
ax.plot(tt, jp.rad2deg(noisy_theta), label='Observed (noisy)',   color='gray',     linewidth=0.6, alpha=0.6)
ax.plot(tt, jp.rad2deg(theta_basis), label='Basis fit predicted', color='tomato',  linewidth=1.2, alpha=0.9)
ax.plot(tt, jp.rad2deg(theta_gp),    label='GP predicted',        color='seagreen', linewidth=1.2, alpha=0.9)
ax.set_ylabel('Angle (degrees)')
ax.set_title('Lever angle: true vs noisy vs predicted')
ax.legend(fontsize=8)

# Loss panel
ax = axes[2]
ax.semilogy(losses_basis, label=f'Basis fit L-BFGS ({N_BASIS} splines)', color='tomato')
ax.set_xlabel('L-BFGS step')
ax.set_ylabel('Loss')
ax.set_title('Basis fit loss curve')
ax.legend()

plt.tight_layout()
plt.savefig('test_gradient_refinement.png', dpi=120)
print("\nPlot saved to test_gradient_refinement.png")
plt.show()
