import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate

class TorqueLeverSimulation:

    def __init__(self,
                 lever_mass,
                 lever_length,
                 lever_range, #Tuple
                 motor_baseline_torque,
                 motor_extra_torque,
                 motor_onset_angle,
                 friction_coeff,
                 gravity_acc=9.82):
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

    def motion_equations(self, t, x):
        #Note: All angles should be in radians in this method
        
        #Angle + angular speed
        theta, theta_dot = x
    
        torque_motor = self.baseline_torque
        factor = np.clip((self.onset_max - theta) / (self.onset_max - self.onset_min), 0, 1)
        torque_motor += self.extra_torque * factor
        torque_rat = self.get_torque_rat(t, theta, theta_dot)
        torque_rat = np.clip(torque_rat, -np.inf, 0.0)
        torque_grav = self.gravity_coeff * np.sin(theta)
        torque_friction = -self.friction_coeff * theta_dot
        torque = torque_grav + torque_motor + torque_rat + torque_friction
            
        #Angular acceleration
        theta_dotdot = torque / self.inertia
    
        return theta_dot, theta_dotdot

    def clip_values(self, theta, theta_dot):
        if theta < self.lever_min or theta > self.lever_max:
            theta_dot = 0.0
            theta = np.clip(theta, self.lever_min, self.lever_max)
        return theta, theta_dot

    def simulate_trial(self, duration=2.0, dt=1e-4):
        theta_start = self.lever_max - np.deg2rad(5)
        time_start = 0.0
        time_range = np.arange(time_start, time_start+duration, dt)
        
        theta = theta_start
        theta_dot = 0.0
        thetas = []
        theta_dots = []
        theta_dotdots = []
        for t in time_range:
            theta_dot, theta_dotdot = self.motion_equations(t, (theta, theta_dot))
            theta += dt*theta_dot
            theta_dot += dt*theta_dotdot
            theta, theta_dot = self.clip_values(theta, theta_dot)
            thetas.append(theta)
            theta_dots.append(theta_dot)
            theta_dotdots.append(theta_dotdot)
        return {
            't': time_range,
            'theta': np.array(thetas),
            'theta_dot': np.array(theta_dots),
            'theta_dotdot': np.array(theta_dotdots),
        }
        #time_span = (time_start, time_start + duration)
        #y_start = (theta_start, theta_dot_start)
        #return scipy.integrate.solve_ivp(self.motion_equations, time_span, y_start,
        #                                 rtol=1e-2, atol=1e-3)

    def get_torque_rat(self, theta, theta_dot):
        raise NotImplemented("Use a subclass to define the rat's behavior")

class FixedTorque(TorqueLeverSimulation):
    def __init__(self, *args, fixed_torque, **kwargs):
        super().__init__(*args, **kwargs)
        self.fixed_torque = fixed_torque

    def get_torque_rat(self, time, theta, theta_dot):
        return self.fixed_torque

class FixedForce(TorqueLeverSimulation):
    def __init__(self, *args, fixed_force, **kwargs):
        super().__init__(*args, **kwargs)
        self.fixed_force = fixed_force

    def get_torque_rat(self, time, theta, theta_dot):
        torque = self.lever_length * self.fixed_force * np.sin(theta) / 2
        return torque

class TwoStepForce(TorqueLeverSimulation):
    def __init__(self, *args, force_1, force_2, switch_ang, **kwargs):
        super().__init__(*args, **kwargs)
        self.force_1 = force_1
        self.force_2 = force_2
        self.switch_rad = np.deg2rad(switch_ang)

    def get_torque_rat(self, time, theta, theta_dot):
        force = self.force_1 if theta < self.switch_rad else self.force_2
        torque = self.lever_length * force * np.sin(theta) / 2
        return torque

sim = TwoStepForce(lever_mass = 3.5e-3,    #kg
                   lever_length = 15e-2,   #meter
                   lever_range = (30, 100), #degrees
                   motor_baseline_torque = 3e-3,
                   motor_extra_torque = 2e-2,
                   motor_onset_angle = (75, 80),
                   friction_coeff = 0.0e-3,
                   fixed_force = -0.2)
                   #fixed_torque = -1e-2)
sim_result = sim.simulate_trial(dt=1e-4)
plt.plot(sim_result["t"], np.rad2deg(sim_result["theta"]))
#plt.ylim(0, 120)
plt.xlabel("Time (s)")
plt.ylabel("Theta (degrees)")
plt.ylim(0, 100)
plt.axhline(80, lw=1, ls='--', color='C1')
plt.axhline(75, lw=1, ls='--', color='C1')
plt.axhline(30, lw=1, ls='--', color='C2')
#1.2 * 15e-2 = 
plt.show()

plt.plot(sim_result["t"], np.rad2deg(sim_result["theta_dot"]))
plt.xlabel("Time (s)")
plt.ylabel("Theta vel (degrees/s)")
plt.show()

plt.plot(sim_result["t"], np.rad2deg(sim_result["theta_dotdot"]))
plt.xlabel("Time (s)")
plt.ylabel("Theta acc (degrees/s^2)")