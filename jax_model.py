import jax
import jax.numpy as jp

class TorqueLeverSimulationJAX:

    def __init__(self,
                 lever_mass,
                 lever_length,
                 lever_range, #Tuple
                 motor_baseline_torque,
                 motor_extra_torque,
                 motor_onset_angle,
                 friction_coeff,
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
            start_theta = self.lever_max - jp.deg2rad(5.0) #5 degrees below max
        _, thetas = jax.lax.scan(self.step,
                              init=(start_theta, start_theta_dot),
                              xs=input_forces)
        return thetas
