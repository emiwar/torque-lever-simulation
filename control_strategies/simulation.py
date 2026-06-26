import jax
import jax.numpy as jp

# Hardware constants for the rat lever apparatus.
# All other code imports from here so parameters are defined exactly once.
SIM_PARAMS = dict(
    lever_mass=33e-3,                        # kg
    lever_length=14.5e-2,                    # m
    lever_range=(50, 92.012),                # degrees (hard stop at each end; 478 clicks)
    motor_baseline_torque=0.1,               # Nm, always active
    motor_extra_torque=0.0,                  # Nm, varied across trials
    motor_onset_angle=(80.76172, 85.15625),  # degrees (ramp start, ramp end)
    friction_coeff=0.0,
    dt=1e-3,                                 # s
)


class TorqueLeverSimulationJAX:
    """Forward-Euler physics simulation of a motorised torque lever.

    The lever rotates under gravity, a motor torque, friction, and an
    external force applied by the agent. All angles are in radians internally.

    Parameters mirror SIM_PARAMS; construct with TorqueLeverSimulationJAX(**SIM_PARAMS)
    or supply a modified copy to vary individual parameters.
    """

    def __init__(self,
                 lever_mass,
                 lever_length,
                 lever_range,
                 motor_baseline_torque,
                 motor_extra_torque,
                 motor_onset_angle,
                 friction_coeff,
                 dt=1e-3,
                 gravity_acc=9.82):
        self.inertia = lever_mass * (lever_length ** 2) / 3
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

    def step(self, carry, strategy):
        theta, theta_dot, min_theta, strategy_carry = carry
        force, strategy_carry = strategy(theta, theta_dot, strategy_carry)

        # Deepest angle reached so far this trial (only ever decreases).
        min_theta = jp.minimum(min_theta, theta)

        # Agent can only pull the lever downward (negative torque)
        torque_agent = self.lever_length * force * jp.sin(theta) / 2
        torque_agent = jp.clip(torque_agent, -jp.inf, 0.0)

        # Motor torque ramps up linearly as the lever approaches the onset window.
        # The ramp is driven by the deepest angle reached, so once the torque has
        # shifted toward the challenge it never relaxes back (it only moves toward
        # the challenge), as in the real task.
        torque_motor = self.baseline_torque
        ramp = jp.clip((self.onset_max - min_theta) / (self.onset_max - self.onset_min), 0, 1)
        torque_motor += self.extra_torque * ramp

        torque_grav = self.gravity_coeff * jp.sin(theta)
        torque_friction = -self.friction_coeff * theta_dot

        theta_dotdot = (torque_grav + torque_motor + torque_agent + torque_friction) / self.inertia

        new_theta = theta + self.dt * theta_dot
        new_theta_dot = theta_dot + self.dt * theta_dotdot

        # Zero velocity when the lever hits an end-stop
        new_theta_dot *= jp.logical_and(new_theta >= self.lever_min, new_theta <= self.lever_max)
        new_theta = jp.clip(new_theta, self.lever_min, self.lever_max)

        return (new_theta, new_theta_dot, min_theta, strategy_carry), new_theta

    def run(self, strategy, duration, start_theta_dot=0.0):
        """Simulate for `duration` seconds and return the angle trajectory."""
        start_theta = self.lever_max

        def step(carry, _):
            return self.step(carry, strategy)

        _, thetas = jax.lax.scan(
            step,
            init=(start_theta, start_theta_dot, start_theta, strategy.init_carry()),
            length=int(duration / self.dt),
        )
        return thetas
