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

    def step(self, carry, strategy):
        theta, theta_dot, strategy_carry = carry
        force, strategy_carry = strategy(theta, theta_dot, strategy_carry)
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

        return (new_theta, new_theta_dot, strategy_carry), new_theta

    def run(self, strategy, duration, start_theta=None, start_theta_dot=0.0):
        length = 1000#jp.astype(duration / self.dt, int)
        if start_theta is None:
            start_theta = self.lever_max - jp.deg2rad(5.0) #5 degrees below max
        def step(carry, _):
            return self.step(carry, strategy)
        _, thetas = jax.lax.scan(step, init=(start_theta, start_theta_dot, strategy.init_carry()), length=length)
        return thetas

def evaluate_strategy(strategy):
    sim = TorqueLeverSimulationJAX(
        lever_mass=33e-3, # kg
        lever_length=14.5e-2, # m
        lever_range=(50,93), #Tuple
        motor_baseline_torque=0.1,
        motor_extra_torque=0.0,
        motor_onset_angle=(85.15625, 89.55078),
        friction_coeff = 0.0,
        dt=1e-3
    )
    duration = 1.0
    target = jp.deg2rad(67.5)
    bound_size = jp.deg2rad(5)

    extra_torques = jp.arange(-0.05, 0.15, 0.005)
    def eval_on_torque(extra_torque):
        sim.extra_torque = extra_torque
        thetas = sim.run(strategy, duration=duration)
        min_theta = jp.min(thetas)
        reward = 1.0 - jp.abs(min_theta - target) / bound_size
        reward = jp.clip(reward, 0.0, 1.0)
        return reward

    total_reward = jp.sum(jax.vmap(eval_on_torque)(extra_torques))
    return total_reward

def sample_trajectories(strategy, duration=1.0):
    sim = TorqueLeverSimulationJAX(
        lever_mass=33e-3, # kg
        lever_length=14.5e-2, # m
        lever_range=(50,93), #Tuple
        motor_baseline_torque=0.1,
        motor_extra_torque=0.0,
        motor_onset_angle=(85.15625, 89.55078),
        friction_coeff = 0.0,
        dt=1e-3
    )

    @jax.vmap
    def sim_torque(extra_torque):
        sim.extra_torque = extra_torque
        thetas = sim.run(strategy, duration=duration)
        return thetas
    
    extra_torques = jp.arange(-0.05, 0.15, 0.02)
    return sim_torque(extra_torques)    

class FixedForce:
    def __init__(self, force):
        self.force = force

    def __call__(self, theta, theta_dot, carry):
        return self.force, carry

    def init_carry(self):
        return None


class PIDController:
    def __init__(self, kp, ki, kd, force_min=-jp.inf, force_max=jp.inf):
        self.target = jp.deg2rad(67.5)
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.force_min = force_min
        self.force_max = force_max

    def __call__(self, theta, theta_dot, carry):
        integral = carry
        error = self.target - theta
        integral = integral + error
        # Derivative estimated from angular velocity: d(error)/dt = -theta_dot
        derivative = -theta_dot
        force = self.kp * error + self.ki * integral + self.kd * derivative
        force = jp.clip(force, self.force_min, self.force_max)
        return force, integral

    def init_carry(self):
        return 0.0
