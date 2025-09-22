import abc
import numpy as np

class TorqueLeverSimulation(abc.ABC):

    def __init__(self,
                 lever_mass,
                 lever_length,
                 lever_range, #Tuple
                 motor_baseline_torque,
                 motor_extra_torque,
                 motor_onset_angle,
                 friction_coeff,
                 dt = 1e-4,
                 gravity_acc = 9.82):
        self.inertia = lever_mass * (lever_length**2)/3
        self.lever_length = lever_length
        self.lever_min = np.deg2rad(lever_range[0])
        self.lever_max = np.deg2rad(lever_range[1])
        self.baseline_torque = motor_baseline_torque
        self.extra_torque = motor_extra_torque
        self.onset_min = np.deg2rad(motor_onset_angle[0])
        self.onset_max = np.deg2rad(motor_onset_angle[1])
        self.friction_coeff = friction_coeff
        self.gravity_coeff = -lever_mass * gravity_acc * lever_length / 2
        self.dt = dt

        self.reset()

    def step(self):
        # Agent torque
        torque_rat = self.get_torque_rat()
        torque_rat = np.clip(torque_rat, -np.inf, 0.0)

        # Motor torque
        torque_motor = self.baseline_torque
        factor = np.clip((self.onset_max - self.theta) / (self.onset_max - self.onset_min), 0, 1)
        torque_motor += self.extra_torque * factor

        # Environment torques
        torque_grav = self.gravity_coeff * np.sin(self.theta)
        torque_friction = -self.friction_coeff * self.theta_dot

        self.torque = torque_grav + torque_motor + torque_rat + torque_friction
            
        # Angular acceleration
        self.theta_dotdot = self.torque / self.inertia
        
        # Forward-Euler
        self.theta += self.dt * self.theta_dot
        self.theta_dot += self.dt * self.theta_dotdot
        self.time += self.dt

        # Clip
        if self.theta < self.lever_min or self.theta > self.lever_max:
            self.theta_dot = 0.0
            self.theta = np.clip(self.theta, self.lever_min, self.lever_max)

    def reset(self):
        self.time = 0.0
        self.theta = self.lever_max - np.deg2rad(5) #5 degrees from max
        self.theta_dot = 0.0
        self.theta_dotdot = 0.0
        self.torque = 0.0
        
    def simulate_trial(self, duration=2.0,
                       record=["time", "theta", "theta_dot", "theta_dotdot"]):
        n_steps = int(np.ceil(duration / self.dt))
        self.reset()
        result_dict = {r: np.zeros(n_steps) for r in record}
        for t in np.arange(n_steps):
            self.step()
            for rec in record:
                result_dict[rec][t] = getattr(self, rec)
        return result_dict

    @abc.abstractmethod
    def get_torque_rat(self):
        raise NotImplemented("Use a subclass to define the rat's behavior.")
