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