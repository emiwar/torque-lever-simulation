# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
import numpy as np
cimport numpy as np
from libc.math cimport sin, fabs, isfinite

ctypedef np.float64_t DTYPE_t

cdef class CythonTorqueLeverStep:
    """Cython implementation of TorqueLeverSimulationJAX.step for performance."""
    
    cdef double inertia
    cdef double lever_length
    cdef double lever_min
    cdef double lever_max
    cdef double baseline_torque
    cdef double extra_torque
    cdef double onset_min
    cdef double onset_max
    cdef double friction_coeff
    cdef double gravity_coeff
    cdef double dt
    
    def __init__(self, 
                 double lever_mass,
                 double lever_length,
                 tuple lever_range,
                 double motor_baseline_torque,
                 double motor_extra_torque,
                 tuple motor_onset_angle,
                 double friction_coeff,
                 double dt=1e-3,
                 double gravity_acc=9.82):
        self.inertia = lever_mass * (lever_length**2) / 3.0
        self.lever_length = lever_length
        self.lever_min = np.deg2rad(lever_range[0])
        self.lever_max = np.deg2rad(lever_range[1])
        self.baseline_torque = motor_baseline_torque
        self.extra_torque = motor_extra_torque
        self.onset_min = np.deg2rad(motor_onset_angle[0])
        self.onset_max = np.deg2rad(motor_onset_angle[1])
        self.friction_coeff = friction_coeff
        self.gravity_coeff = -lever_mass * gravity_acc * lever_length / 2.0
        self.dt = dt
    
    cdef tuple step_c(self, double theta, double theta_dot, double force):
        """Internal C-level step function for performance."""
        cdef double torque_rat, torque_motor, factor, torque_grav, torque_friction
        cdef double net_torque, theta_dotdot, new_theta, new_theta_dot
        cdef bint in_bounds
        
        # Agent torque
        torque_rat = self.lever_length * force * sin(theta) / 2.0
        if torque_rat > 0.0:
            torque_rat = 0.0
        
        # Motor torque
        torque_motor = self.baseline_torque
        factor = (self.onset_max - theta) / (self.onset_max - self.onset_min)
        if factor < 0.0:
            factor = 0.0
        elif factor > 1.0:
            factor = 1.0
        torque_motor += self.extra_torque * factor
        
        # Environment torques
        torque_grav = self.gravity_coeff * sin(theta)
        torque_friction = -self.friction_coeff * theta_dot
        
        net_torque = torque_grav + torque_motor + torque_rat + torque_friction
        
        # Angular acceleration
        theta_dotdot = net_torque / self.inertia
        
        # Forward-Euler
        new_theta = theta + self.dt * theta_dot
        new_theta_dot = theta_dot + self.dt * theta_dotdot
        
        # Clip
        in_bounds = new_theta >= self.lever_min and new_theta <= self.lever_max
        if not in_bounds:
            new_theta_dot = 0.0
        
        if new_theta < self.lever_min:
            new_theta = self.lever_min
        elif new_theta > self.lever_max:
            new_theta = self.lever_max
        
        return (new_theta, new_theta_dot)
    
    def step(self, double theta, double theta_dot, double force):
        """Python-facing step function."""
        return self.step_c(theta, theta_dot, force)


cdef class BinnedValue:
    """Cython implementation of BinnedValue for bin quantization."""
    
    cdef int n_bins
    cdef double min_val
    cdef double max_val
    cdef double range_val
    
    def __init__(self, int n_bins, double min_val, double max_val):
        self.n_bins = n_bins
        self.min_val = min_val
        self.max_val = max_val
        self.range_val = max_val - min_val
    
    cdef double bin2val_c(self, int bin_idx):
        """Convert bin index to value."""
        return self.min_val + bin_idx / <double>self.n_bins * self.range_val
    
    cdef int val2bin_c(self, double val):
        """Convert value to bin index."""
        if val < self.min_val or val > self.max_val:
            return -1
        return <int>((val - self.min_val) / self.range_val * self.n_bins)
    
    def bin2val(self, int bin_idx):
        return self.bin2val_c(bin_idx)
    
    def val2bin(self, double val):
        return self.val2bin_c(val)


def fit_force_cython(double[:] ref_thetas, 
                     CythonTorqueLeverStep step_fcn,
                     int force_n_bins=100,
                     double force_min=-1.0,
                     double force_max=0.0,
                     int theta_dot_n_bins=200,
                     double theta_dot_min=-50.0,
                     double theta_dot_max=50.0,
                     int theta_offset_n_bins=20,
                     double theta_offset_min=-1.0,
                     double theta_offset_max=1.0):
    """
    Cython implementation of dynamic programming force fitting algorithm.
    
    Parameters:
    -----------
    ref_thetas : array of shape (n_steps,)
        Reference theta values
    step_fcn : CythonTorqueLeverStep
        Step function for dynamics
    force_n_bins, theta_dot_n_bins, theta_offset_n_bins : int
        Number of bins for each dimension
    """
    
    cdef int n_steps = len(ref_thetas)
    cdef BinnedValue force_bins = BinnedValue(force_n_bins, force_min, force_max)
    cdef BinnedValue theta_dot_bins = BinnedValue(theta_dot_n_bins, theta_dot_min, theta_dot_max)
    cdef BinnedValue theta_offset_bins = BinnedValue(theta_offset_n_bins, theta_offset_min, theta_offset_max)
    
    # Allocate cost and backref arrays
    cdef double[:,:,:,:] lowest_cost = np.full(
        (n_steps, force_n_bins, theta_dot_n_bins, theta_offset_n_bins), 
        np.inf, 
        dtype=np.float64
    )
    cdef int[:,:,:,:,:] backref = np.full(
        (n_steps, force_n_bins, theta_dot_n_bins, theta_offset_n_bins, 3),
        -1,
        dtype=np.int32
    )
    
    lowest_cost[0, :, :, :] = 0.0
    
    # Dynamic programming forward pass
    cdef int t, force_b, theta_dot_b, theta_offset_b, next_force_b, next_theta_dot_b, next_theta_offset_b
    cdef double force, theta_dot, theta_offset, theta, next_force, next_theta, next_theta_dot
    cdef double cost, current_cost
    cdef bint is_valid_next_state
    
    for t in range(n_steps - 1):
        for force_b in range(force_n_bins):
            force = force_bins.bin2val_c(force_b)
            for theta_dot_b in range(theta_dot_n_bins):
                theta_dot = theta_dot_bins.bin2val_c(theta_dot_b)
                for theta_offset_b in range(theta_offset_n_bins):
                    theta_offset = theta_offset_bins.bin2val_c(theta_offset_b)
                    
                    # Skip if current state is unreachable
                    if not isfinite(lowest_cost[t, force_b, theta_dot_b, theta_offset_b]):
                        continue
                    
                    current_cost = lowest_cost[t, force_b, theta_dot_b, theta_offset_b]
                    theta = ref_thetas[t] + theta_offset
                    
                    # Try all next force values
                    for next_force_b in range(force_n_bins):
                        next_force = force_bins.bin2val_c(next_force_b)
                        
                        # Step the dynamics
                        next_theta, next_theta_dot = step_fcn.step_c(theta, theta_dot, force)
                        
                        # Quantize next state
                        next_theta_offset_b = theta_offset_bins.val2bin_c(next_theta - ref_thetas[t + 1])
                        next_theta_dot_b = theta_dot_bins.val2bin_c(next_theta_dot)
                        
                        is_valid_next_state = next_theta_offset_b >= 0 and next_theta_dot_b >= 0
                        
                        if is_valid_next_state:
                            # Compute transition cost
                            cost = current_cost + fabs(force - next_force)
                            
                            # Update if this is a better path
                            if cost < lowest_cost[t + 1, next_force_b, next_theta_dot_b, next_theta_offset_b]:
                                lowest_cost[t + 1, next_force_b, next_theta_dot_b, next_theta_offset_b] = cost
                                backref[t + 1, next_force_b, next_theta_dot_b, next_theta_offset_b, 0] = force_b
                                backref[t + 1, next_force_b, next_theta_dot_b, next_theta_offset_b, 1] = theta_dot_b
                                backref[t + 1, next_force_b, next_theta_dot_b, next_theta_offset_b, 2] = theta_offset_b
    
    # Backtracking to find best path
    cdef double[:] force_est = np.full(n_steps, np.nan, dtype=np.float64)
    cdef double min_final_cost = np.inf
    cdef int best_force_b = -1, best_theta_dot_b = -1, best_theta_offset_b = -1
    
    # Find best final state
    for force_b in range(force_n_bins):
        for theta_dot_b in range(theta_dot_n_bins):
            for theta_offset_b in range(theta_offset_n_bins):
                if lowest_cost[n_steps - 1, force_b, theta_dot_b, theta_offset_b] < min_final_cost:
                    min_final_cost = lowest_cost[n_steps - 1, force_b, theta_dot_b, theta_offset_b]
                    best_force_b = force_b
                    best_theta_dot_b = theta_dot_b
                    best_theta_offset_b = theta_offset_b
    
    if best_force_b < 0:
        raise RuntimeError("Failed to find any compatible force sequence")
    
    # Backtrack through solution
    force_b = best_force_b
    theta_dot_b = best_theta_dot_b
    theta_offset_b = best_theta_offset_b
    
    for t in range(n_steps - 1, -1, -1):
        force_est[t] = force_bins.bin2val_c(force_b)
        if t > 0:
            # Read previous-state indices using the current (force_b, theta_dot_b, theta_offset_b)
            prev_force_b = backref[t, force_b, theta_dot_b, theta_offset_b, 0]
            prev_theta_dot_b = backref[t, force_b, theta_dot_b, theta_offset_b, 1]
            prev_theta_offset_b = backref[t, force_b, theta_dot_b, theta_offset_b, 2]
            force_b = prev_force_b
            theta_dot_b = prev_theta_dot_b
            theta_offset_b = prev_theta_offset_b
    
    return np.asarray(force_est)
