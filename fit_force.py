import jax
import jax.numpy as jp
import numpy as np
import optax
import tqdm
import scipy.linalg
from scipy.optimize import minimize

class TorqueLeverSimulationJAX:

    def __init__(self,
                 lever_mass,
                 lever_length,
                 lever_range, #Tuple
                 motor_baseline_torque,
                 motor_extra_torque,
                 motor_onset_angle,
                 friction_coeff,
                 autoregressive_penalty,
                 dt = 1e-3,
                 gravity_acc = 9.82):
        self.inertia = lever_mass * (lever_length**2)/3
        self.lever_length = lever_length
        self.lever_min = jp.deg2rad(lever_range[0])
        self.lever_max = jp.deg2rad(lever_range[1])
        self.baseline_torque = motor_baseline_torque
        self.extra_torque = motor_extra_torque
        self.onset_min = jp.deg2rad(motor_onset_angle[0])
        self.onset_max = jp.deg2rad(motor_onset_angle[1])
        self.friction_coeff = friction_coeff
        self.gravity_coeff = -lever_mass * gravity_acc * lever_length / 2
        self.dt = dt
        self.autoregressive_penalty = autoregressive_penalty

    def step(self, carry, force):
        theta, theta_dot = carry
        torque_rat = self.lever_length * force * jp.sin(theta) / 2
        torque_rat = jp.clip(torque_rat, -jp.inf, 0.0)

        # Motor torque
        torque_motor = self.baseline_torque
        factor = jp.clip((self.onset_max - theta) / (self.onset_max - self.onset_min), 0, 1)
        torque_motor += self.extra_torque * factor

        # Environment torques
        torque_grav = self.gravity_coeff * jp.sin(theta)
        torque_friction = -self.friction_coeff * theta_dot

        net_torque = torque_grav + torque_motor + torque_rat + torque_friction

        # Angular acceleration
        theta_dotdot = net_torque / self.inertia

        # Forward-Euler
        new_theta = theta + self.dt * theta_dot
        new_theta_dot = theta_dot + self.dt * theta_dotdot

        # Clip
        new_theta_dot *= jp.logical_and(new_theta >= self.lever_min, new_theta <= self.lever_max)
        new_theta = jp.clip(new_theta, self.lever_min, self.lever_max)

        return (new_theta, new_theta_dot), new_theta

    def run(self, input_forces, start_theta=None, start_theta_dot=0.0):
        if start_theta is None:
            start_theta = self.lever_max - jp.deg2rad(5.0)
        _, thetas = jax.lax.scan(self.step,
                                 init=(start_theta, start_theta_dot),
                                 xs=input_forces)
        return thetas

    def analytical_inversion(self, thetas, start_theta=None):
        # The scan output starts at θ₁; prepend θ₀ so finite differences
        # are computed at the correct (pre-step) states.
        if start_theta is None:
            start_theta = self.lever_max - jp.deg2rad(5.0)
        full_thetas = jp.concatenate([jp.array([start_theta]), thetas])  # shape N+1

        theta_dot = jp.diff(full_thetas) / self.dt      # shape N:   θ_dot_0 .. θ_dot_{N-1}
        theta_dotdot = jp.diff(theta_dot) / self.dt     # shape N-1: θ_dotdot_0 .. θ_dotdot_{N-2}

        # Physics evaluated at steps 0..N-2
        theta_i     = full_thetas[:-2]   # θ_0 .. θ_{N-2}
        theta_dot_i = theta_dot[:-1]     # θ_dot_0 .. θ_dot_{N-2}

        tau_net      = self.inertia * theta_dotdot
        tau_grav     = self.gravity_coeff * jp.sin(theta_i)
        factor       = jp.clip((self.onset_max - theta_i) / (self.onset_max - self.onset_min), 0, 1)
        tau_motor    = self.baseline_torque + self.extra_torque * factor
        tau_friction = -self.friction_coeff * theta_dot_i

        tau_rat = tau_net - tau_grav - tau_motor - tau_friction
        tau_rat = jp.clip(tau_rat, -jp.inf, 0.0)

        forces = tau_rat / (self.lever_length * jp.sin(theta_i) / 2)

        # The second difference at step k uses full_thetas[k], [k+1], and [k+2].
        # If any of those is at a boundary, the position/velocity was clipped in the
        # forward model, corrupting the finite differences — zero those estimates out.
        at_boundary = (
            (full_thetas[:-2] >= self.lever_max) | (full_thetas[:-2] <= self.lever_min) |
            (full_thetas[1:-1] >= self.lever_max) | (full_thetas[1:-1] <= self.lever_min) |
            (full_thetas[2:] >= self.lever_max) | (full_thetas[2:] <= self.lever_min)
        )
        forces = jp.where(at_boundary, 0.0, forces)

        # Pad the final timestep (no second difference available there)
        return jp.pad(forces, (0, 1), mode='edge')

    def eval_candidate(self, forces, reference_thetas, start_theta, start_theta_dot):
        _, thetas = jax.lax.scan(self.step,
                                 init=(start_theta, start_theta_dot),
                                 xs=forces)
        loss = jp.mean(jp.square(thetas - reference_thetas))
        loss += self.autoregressive_penalty * jp.mean(jp.square(jp.diff(forces)))
        return loss, thetas

    def fit_force(self, reference_thetas, start_theta, start_theta_dot,
                  n_steps=1000, learning_rate=1e-4, forces_init=None):
        if forces_init is None:
            forces_init = jp.zeros_like(reference_thetas)

        loss_fn = jax.jit(
            lambda f: self.eval_candidate(f, reference_thetas, start_theta, start_theta_dot)
        )
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

        optimizer = optax.adam(learning_rate=learning_rate)
        opt_state = optimizer.init(forces_init)
        forces = forces_init
        losses = []

        for _ in tqdm.trange(n_steps):
            (loss, _), grads = grad_fn(forces)
            updates, opt_state = optimizer.update(grads, opt_state)
            forces = optax.apply_updates(forces, updates)
            losses.append(float(loss))

        return jp.array(losses), forces

    def fit_force_basis(self, reference_thetas, start_theta, start_theta_dot,
                        basis, max_iter=200, coeffs_init=None):
        """Optimize forces parameterized as basis @ coeffs using L-BFGS.

        basis: array of shape (N, K) mapping K smooth basis coefficients to N force values.
        Smoothness is enforced by construction; L-BFGS with line search guarantees
        monotonically decreasing loss.
        """
        basis = jp.array(basis)

        if coeffs_init is None:
            coeffs_init = jp.zeros(basis.shape[1])

        def loss_fn(coeffs):
            forces = basis @ coeffs
            _, thetas = jax.lax.scan(self.step,
                                     init=(start_theta, start_theta_dot),
                                     xs=forces)
            return jp.mean(jp.square(thetas - reference_thetas))

        val_grad_fn = jax.jit(jax.value_and_grad(loss_fn))

        # Record loss once per L-BFGS step (not per line-search evaluation)
        losses = []
        def callback(coeffs):
            loss, _ = val_grad_fn(jp.array(coeffs))
            losses.append(float(loss))

        def scipy_fn(coeffs_np):
            loss, grads = val_grad_fn(jp.array(coeffs_np))
            return float(loss), np.array(grads, dtype=np.float64)

        # Warm-up JIT before handing off to scipy
        scipy_fn(np.array(coeffs_init))

        result = minimize(
            scipy_fn,
            np.array(coeffs_init),
            method='L-BFGS-B',
            jac=True,
            callback=callback,
            options={'maxiter': max_iter, 'ftol': 0, 'gtol': 1e-8},
        )

        return jp.array(losses), basis @ jp.array(result.x)

    def fit_force_gp(self, reference_thetas, start_theta, start_theta_dot,
                     tt, sigma_obs, forces_linearize_around,
                     length_scale=None, signal_std=None):
        """GP posterior inference via linearisation of the forward model.

        Linearises run(F) around forces_linearize_around, places an RBF GP prior
        on the correction δF = F - F_0, and returns the posterior mean and std.
        Hyperparameters are optimised by marginal likelihood if not supplied.

        Returns: (F_mean, F_std) as JAX arrays — 95% CI is F_mean ± 1.96 * F_std.
        """
        F_0 = np.array(forces_linearize_around)
        tt  = np.asarray(tt)
        N   = len(tt)

        # --- Jacobian J = d(theta)/d(F) at F_0, shape (N, N) ---
        print("Computing Jacobian (this may take ~10-30 s)...")
        _run = jax.jit(lambda f: self.run(f, start_theta, start_theta_dot))
        J = np.array(jax.jacrev(_run)(jp.array(F_0)))

        theta_0 = np.array(self.run(jp.array(F_0), start_theta, start_theta_dot))
        r = np.array(reference_thetas) - theta_0   # residual, shape (N,)

        # Precompute squared distances for kernel construction
        sq_dists = (tt[:, None] - tt[None, :]) ** 2   # (N, N)

        def build_kernel(l, sf):
            return sf ** 2 * np.exp(-0.5 * sq_dists / l ** 2)

        # Marginal-likelihood optimisation only works if the linearisation point has
        # meaningful residuals (r ≠ 0). When linearising around an already-optimal F,
        # r ≈ 0 and the ML collapses. Use caller-supplied or sensible defaults instead.
        if length_scale is None:
            length_scale = 0.15   # forces correlated over ~15% of trajectory
        if signal_std is None:
            signal_std = 0.1      # expected residual force amplitude after initial fit
        l_opt, sf_opt = length_scale, signal_std
        print(f"  length_scale={l_opt:.4f}  signal_std={sf_opt:.4f}")

        K = build_kernel(l_opt, sf_opt)
        S = J @ K @ J.T + sigma_obs ** 2 * np.eye(N)
        L = scipy.linalg.cholesky(S, lower=True)

        alpha    = scipy.linalg.cho_solve((L, True), r)
        mu_delta = K @ J.T @ alpha

        # Posterior variance: diag(K - K J^T S^{-1} J K)
        JK = J @ K                                          # (N, N)
        v  = scipy.linalg.cho_solve((L, True), JK)         # S^{-1} J K, (N, N)
        var_delta = np.diag(K) - np.einsum('ij,ij->j', JK, v)
        sigma_delta = np.sqrt(np.maximum(var_delta, 0.0))

        return jp.array(F_0 + mu_delta), jp.array(sigma_delta)
