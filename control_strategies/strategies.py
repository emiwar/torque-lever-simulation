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
