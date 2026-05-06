"""
MCMC force inference using BlackJAX NUTS.

Parameterises F(t) = basis @ coeffs (same 30-spline basis as the basis fit),
places an independent Gaussian prior on the coefficients, and samples the
posterior with NUTS.  This gives calibrated confidence intervals that account
for the full nonlinear posterior rather than the linearised GP approximation.
"""

import jax
import jax.numpy as jp
import numpy as np
import blackjax
import matplotlib.pyplot as plt
from scipy.interpolate import BSpline
from scipy.ndimage import gaussian_filter1d
import fit_force

# ── Hyper-parameters ──────────────────────────────────────────────────────────
NOISE_SIGMA  = 0.005   # observation noise (rad)
SMOOTH_SIGMA = 20      # Gaussian pre-smooth for analytical warm-start (samples)
N_BASIS      = 30      # cubic B-spline coefficients
N_STEPS      = 500
PRIOR_STD    = 1.0     # Gaussian prior std on each basis coefficient
N_WARMUP     = 500     # NUTS window-adaptation steps
N_SAMPLES    = 2000    # posterior samples to draw after warmup
# ─────────────────────────────────────────────────────────────────────────────

sim = fit_force.TorqueLeverSimulationJAX(
    lever_mass=3.5e-3,
    lever_length=15e-2,
    lever_range=(30, 100),
    motor_baseline_torque=3e-3,
    motor_extra_torque=2e-2,
    motor_onset_angle=(75, 80),
    friction_coeff=0.0e-3,
    autoregressive_penalty=0.0,
    dt=1e-3,
)

tt = np.linspace(0, 1, N_STEPS)
true_force  = jp.array(-np.sin(4 * tt) ** 2 * tt - 0.1 * tt ** 2)
true_theta  = sim.run(true_force)

rng_np = np.random.default_rng(42)
noisy_theta = true_theta + jp.array(rng_np.normal(0.0, NOISE_SIGMA, true_theta.shape))

start_theta     = sim.lever_max - jp.deg2rad(5.0)
start_theta_dot = jp.array(0.0)

# ── Build B-spline basis (identical to test_gradient_refinement.py) ───────────
k = 3
n_interior = N_BASIS - k - 1
interior_knots = np.linspace(tt[0], tt[-1], n_interior + 2)[1:-1]
knots = np.concatenate([[tt[0]] * (k + 1), interior_knots, [tt[-1]] * (k + 1)])
basis_np = BSpline.design_matrix(tt, knots, k).toarray()   # (N_STEPS, N_BASIS)
basis    = jp.array(basis_np)

# ── Warm-start coefficients from the analytical inversion + basis fit ─────────
print("Computing warm-start via analytical inversion + basis fit...")
smoothed   = jp.array(gaussian_filter1d(np.array(noisy_theta), sigma=SMOOTH_SIGMA))
F_anal     = sim.analytical_inversion(smoothed, start_theta=start_theta)
coeffs_init = jp.array(np.linalg.lstsq(basis_np, np.array(F_anal), rcond=None)[0])
_, F_basis = sim.fit_force_basis(
    noisy_theta, start_theta, start_theta_dot,
    basis=basis_np, max_iter=200, coeffs_init=coeffs_init,
)
coeffs_basis = jp.array(np.linalg.lstsq(basis_np, np.array(F_basis), rcond=None)[0])

# ── Log-density for NUTS ──────────────────────────────────────────────────────
@jax.jit
def log_density(coeffs):
    forces     = basis @ coeffs
    theta_pred = sim.run(forces, start_theta, start_theta_dot)
    log_lik    = -0.5 * jp.sum(jp.square(noisy_theta - theta_pred)) / NOISE_SIGMA ** 2
    log_prior  = -0.5 * jp.sum(jp.square(coeffs)) / PRIOR_STD ** 2
    return log_lik + log_prior

# ── NUTS with window adaptation (warmup) ──────────────────────────────────────
print(f"\nRunning NUTS warmup ({N_WARMUP} steps)...")
rng_key = jax.random.PRNGKey(0)
warmup  = blackjax.window_adaptation(blackjax.nuts, log_density)
(warmup_state, nuts_params), warmup_info = warmup.run(
    rng_key, coeffs_basis, num_steps=N_WARMUP
)
print(f"  Adapted step size:  {nuts_params['step_size']:.5f}")
print(f"  Warmup acceptance:  {float(warmup_info.acceptance_rate.mean()):.3f}")

# ── Posterior sampling ────────────────────────────────────────────────────────
print(f"\nDrawing {N_SAMPLES} posterior samples...")
nuts_kernel = blackjax.nuts(log_density, **nuts_params)

@jax.jit
def one_step(state, rng_key):
    state, info = nuts_kernel.step(rng_key, state)
    return state, (state.position, info.acceptance_rate)

sample_keys = jax.random.split(rng_key, N_SAMPLES)
final_state, (coeffs_samples, acceptance_rates) = jax.lax.scan(
    one_step, warmup_state, sample_keys
)

mean_acceptance = float(acceptance_rates.mean())
print(f"  Mean acceptance rate: {mean_acceptance:.3f}")

# ── Convert coefficient samples → force samples ───────────────────────────────
# coeffs_samples: (N_SAMPLES, N_BASIS)  →  forces: (N_SAMPLES, N_STEPS)
forces_samples = np.array(coeffs_samples) @ basis_np.T

F_mcmc_mean = forces_samples.mean(axis=0)
F_mcmc_lo   = np.percentile(forces_samples,  2.5, axis=0)
F_mcmc_hi   = np.percentile(forces_samples, 97.5, axis=0)
F_mcmc_std  = forces_samples.std(axis=0)

# ── Coverage ──────────────────────────────────────────────────────────────────
true_force_np = np.array(true_force)
in_ci    = (true_force_np >= F_mcmc_lo) & (true_force_np <= F_mcmc_hi)
coverage = in_ci.mean()

def rmse(a, b):
    return float(np.sqrt(np.mean((np.array(a) - np.array(b)) ** 2)))

print(f"\nRMSE basis fit:  {rmse(F_basis,     true_force):.4f} N")
print(f"RMSE MCMC mean:  {rmse(F_mcmc_mean, true_force):.4f} N")
print(f"95% CI coverage: {coverage:.3f}  (nominal 0.95)")

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(10, 12))

# Force panel
ax = axes[0]
ax.plot(tt, true_force_np, label='True force',  color='black',  linewidth=2)
ax.plot(tt, np.array(F_basis),   label=f'Basis fit (L-BFGS)  RMSE={rmse(F_basis, true_force):.3f}',
        color='tomato', linewidth=1.5, alpha=0.8)
ax.plot(tt, F_mcmc_mean,  label=f'MCMC mean           RMSE={rmse(F_mcmc_mean, true_force):.3f}',
        color='seagreen', linewidth=1.5)
ax.fill_between(tt, F_mcmc_lo, F_mcmc_hi,
                color='seagreen', alpha=0.25,
                label=f'MCMC 95% CI  (coverage={coverage:.2f})')
ax.set_ylabel('Force (N)')
ax.set_title(f'Force recovery — NUTS MCMC  [noise σ={NOISE_SIGMA} rad, {N_BASIS} splines, {N_SAMPLES} samples]')
ax.legend(fontsize=8)

# Theta panel
_, theta_mcmc = sim.eval_candidate(
    jp.array(F_mcmc_mean), noisy_theta, start_theta, start_theta_dot
)
ax = axes[1]
ax.plot(tt, np.degrees(np.array(true_theta)),   label='True θ (clean)',    color='black',    linewidth=2)
ax.plot(tt, np.degrees(np.array(noisy_theta)),  label='Observed (noisy)',  color='gray',     linewidth=0.6, alpha=0.6)
ax.plot(tt, np.degrees(np.array(theta_mcmc)),   label='MCMC mean predicted', color='seagreen', linewidth=1.5, alpha=0.9)
ax.set_ylabel('Angle (degrees)')
ax.set_title('Lever angle: true vs noisy vs MCMC prediction')
ax.legend(fontsize=8)

# Coverage panel
ax = axes[2]
ax.axhline(0.95, color='black', linestyle='--', linewidth=1, label='Nominal 95%')
window = 25
coverage_rolling = np.convolve(in_ci.astype(float), np.ones(window)/window, mode='same')
ax.plot(tt, coverage_rolling, color='seagreen', linewidth=1.5,
        label=f'Rolling coverage (window={window})')
ax.set_ylim(0, 1.05)
ax.set_xlabel('t')
ax.set_ylabel('Coverage')
ax.set_title('Local 95% CI coverage across trajectory')
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('mcmc_force.png', dpi=120)
print("\nPlot saved to mcmc_force.png")
plt.show()
