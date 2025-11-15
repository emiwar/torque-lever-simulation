import jax
import jax.numpy as jp
import tqdm
import optax

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
        
    def eval_candidate(self, candidate, reference_thetas):
        force, start_theta, start_theta_dot = candidate
        _, thetas = jax.lax.scan(self.step,
                              init=(start_theta, start_theta_dot),
                              xs=force)
        loss = -jp.mean(jp.square(thetas - reference_thetas))
        loss -= self.autoregressive_penalty * jp.mean(jp.abs(jp.ediff1d(force)))
        return loss, thetas
    
    def fit_force(self, reference_thetas, n_steps=1000, eps=1e-4, start_candidate = None):
        if start_candidate is None:
            start_candidate = (jp.zeros_like(reference_thetas), jp.array(reference_thetas[0]), jp.array(0.0))
        loss_fn = jax.jit(lambda cand: self.eval_candidate(cand, reference_thetas))
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        losses = []
        candidate = start_candidate
        optimizer = optax.adam(learning_rate=eps)
        opt_state = optimizer.init(candidate)

        # Replace the gradient descent loop with:
        for i in tqdm.trange(n_steps):
            (loss, thetas), grad = grad_fn(candidate)
            updates, opt_state = optimizer.update(grad, opt_state)
            candidate = jax.tree_util.tree_map(lambda x, u: x - u, candidate, updates)
            losses.append(loss)
        final_loss, theta = loss_fn(candidate)
        return jp.array(losses), candidate, theta
