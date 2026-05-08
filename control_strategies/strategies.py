import jax
import jax.numpy as jp

# Strategy interface
# ------------------
# A strategy is a callable object that decides the force to apply at each
# simulation step. It must implement two methods:
#
#   __call__(self, theta, theta_dot, carry) -> (force, new_carry)
#       theta      : current lever angle (radians)
#       theta_dot  : current angular velocity (rad/s)
#       carry      : any persistent state from the previous step (integral
#                    accumulator, history, etc.)
#       force      : force to apply this step (Newtons; negative pulls lever down)
#       new_carry  : updated persistent state, passed to the next step
#
#   init_carry(self) -> carry
#       Returns the initial carry value before the simulation starts.
#
# To add a new strategy, implement these two methods and add the class below.
# The strategy will work with evaluate_strategy() and sample_trajectories()
# in evaluate.py without any other changes.


class FixedForce:
    """Applies a constant downward force regardless of lever state."""

    def __init__(self, force):
        self.force = force

    def __call__(self, theta, theta_dot, carry):
        return self.force, carry

    def init_carry(self):
        return None


class PIDController:
    """Feedback controller tracking a target lever angle.

    Set ki=kd=0 for proportional-only (P), ki=0 for PD.

    The integral accumulates angle error over time (units: rad · steps).
    The derivative term uses -theta_dot rather than differencing the setpoint,
    which avoids spikes when the target changes.
    """

    def __init__(self, target_deg, kp, ki=0.0, kd=0.0,
                 force_min=-jp.inf, force_max=jp.inf):
        self.target = jp.deg2rad(target_deg)
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.force_min = force_min
        self.force_max = force_max

    def __call__(self, theta, theta_dot, carry):
        integral = carry
        error = self.target - theta
        integral = integral + error
        derivative = -theta_dot
        force = self.kp * error + self.ki * integral + self.kd * derivative
        force = jp.clip(force, self.force_min, self.force_max)
        return force, integral

    def init_carry(self):
        return 0.0


class PulseForce:
    """Applies a constant force for the first `duration_ms` milliseconds, then nothing.

    Parameters
    ----------
    force        : force magnitude (Newtons; negative pulls the lever down)
    duration_ms  : how long to apply the force (milliseconds = steps at dt=1 ms)
    """

    def __init__(self, force, duration_ms):
        self.force = force
        self.duration_ms = jp.asarray(duration_ms, dtype=jp.int32)

    def __call__(self, _theta, _theta_dot, carry):
        step = carry
        applied_force = jp.where(step < self.duration_ms, self.force, 0.0)
        return applied_force, step + 1

    def init_carry(self):
        return jp.array(0, dtype=jp.int32)


class NoisyDelayedPIDController:
    """PID controller with actuator delay and force noise.

    The controller computes a force from the current lever state each step, but
    the force is not applied immediately — it enters a delay buffer and is
    applied `delay_ms` milliseconds later. Gaussian noise can optionally be
    added to the force command before it is buffered. The first `delay_ms`
    steps naturally receive zero force.

    Parameters
    ----------
    target_deg  : target lever angle in degrees
    kp, ki, kd  : PID gains
    delay_ms    : actuator delay in milliseconds (= steps at dt=1 ms); must be >= 1
    noise_std   : std of Gaussian noise added to the force command (Newtons)
    seed        : PRNG seed for the noise
    """

    def __init__(self, target_deg, kp, ki=0.0, kd=0.0,
                 delay_ms=1, noise_std=0.0,
                 force_min=-jp.inf, force_max=jp.inf,
                 seed=0):
        self.target = jp.deg2rad(target_deg)
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.delay_ms = delay_ms
        self.noise_std = noise_std
        self.force_min = force_min
        self.force_max = force_max
        self.seed = seed

    def __call__(self, theta, theta_dot, carry):
        integral, force_buffer, write_idx, rng_key = carry

        # Apply the force that was computed delay_ms steps ago
        applied_force = force_buffer[write_idx]

        # Compute the new PID force from current state
        error = self.target - theta
        integral = integral + error
        new_force = self.kp * error + self.ki * integral + self.kd * (-theta_dot)
        new_force = jp.clip(new_force, self.force_min, self.force_max)

        # Optionally add noise, then store in buffer
        rng_key, subkey = jax.random.split(rng_key)
        new_force = new_force + self.noise_std * jax.random.normal(subkey)

        force_buffer = force_buffer.at[write_idx].set(new_force)
        write_idx = (write_idx + 1) % self.delay_ms

        return applied_force, (integral, force_buffer, write_idx, rng_key)

    def init_carry(self):
        return (
            jp.array(0.0),
            jp.zeros(self.delay_ms),
            jp.array(0, dtype=jp.int32),
            jax.random.PRNGKey(self.seed),
        )
