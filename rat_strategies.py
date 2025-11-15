import numpy as np

from physics_model import TorqueLeverSimulation

class FixedTorque(TorqueLeverSimulation):
    def __init__(self, *args, fixed_torque, **kwargs):
        super().__init__(*args, **kwargs)
        self.fixed_torque = fixed_torque

    def get_torque_rat(self):
        return self.fixed_torque

class FixedForce(TorqueLeverSimulation):
    def __init__(self, *args, fixed_force, **kwargs):
        super().__init__(*args, **kwargs)
        self.fixed_force = fixed_force

    def get_torque_rat(self):
        torque = self.lever_length * self.fixed_force * np.sin(self.theta) / 2
        return torque

class TwoStepForce(TorqueLeverSimulation):
    def __init__(self, *args, force_1, force_2, switch_ang, **kwargs):
        super().__init__(*args, **kwargs)
        self.force_1 = force_1
        self.force_2 = force_2
        self.switch_rad = np.deg2rad(switch_ang)

    def get_torque_rat(self):
        force = self.force_1 if self.theta > self.switch_rad else self.force_2
        torque_rat = self.lever_length * force * np.sin(self.theta) / 2
        return torque_rat

class DummyForce(TorqueLeverSimulation):
    def __init__(self):
        tt = np.linspace(0, 1, 500)
        super().__init__(lever_mass = 3.5e-3,    #kg
                         lever_length = 15e-2,   #meter
                         lever_range = (30, 100), #degrees
                         motor_baseline_torque = 3e-3,
                         motor_extra_torque = 2e-2,
                         motor_onset_angle = (75, 80),
                         friction_coeff = 0.0e-3,
                         dt=1e-3)
        self.dummy_force = -np.sin(4*tt)**2*tt + 0.1*tt**2
    
    def get_torque_rat(self):
        force = self.dummy_force[int(self.time / self.dt)]
        torque = force * self.lever_length * np.sin(self.theta) / 2
        return torque
