import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax
import jax.numpy as jp
from dataclasses import dataclass, replace as _dc_replace

from simulation import TorqueLeverSimulationJAX, SIM_PARAMS
from strategies import FixedForce
from evaluate import TARGET_DEG, EXTRA_TORQUE_MIN, EXTRA_TORQUE_MAX

EPISODE_STEPS = 1000   # 1 second at dt=1 ms
FORCE_SCALE   = 5.0    # tanh action in [-1, 1] → force in [-5, 5] N

_TARGET_RAD = jp.deg2rad(TARGET_DEG)


@dataclass
class TorqueLeverEnvState:
    theta:        jax.Array   # lever angle (radians)
    theta_dot:    jax.Array   # angular velocity (rad/s)
    extra_torque: jax.Array   # motor extra torque for this episode (Nm)
    step_count:   jax.Array   # steps taken in current episode
    reward:       jax.Array   # reward from the last step
    done:         jax.Array   # True at end of episode

    def replace(self, **kwargs):
        return _dc_replace(self, **kwargs)

    @property
    def obs(self):
        return jp.stack([self.theta - _TARGET_RAD, self.theta_dot], axis=-1)

    @property
    def info(self):
        return {}

    @property
    def metrics(self):
        return {
            'theta_deg':     jp.rad2deg(self.theta),
            'extra_torque':  self.extra_torque,
        }


jax.tree_util.register_pytree_node(
    TorqueLeverEnvState,
    lambda s: (
        [s.theta, s.theta_dot, s.extra_torque, s.step_count, s.reward, s.done],
        None,
    ),
    lambda _, xs: TorqueLeverEnvState(*xs),
)


class TorqueLeverEnv:
    """RL environment wrapping TorqueLeverSimulationJAX.

    The motor extra-torque is randomised uniformly at each reset, so the agent
    must learn to be robust across the full range of resistance levels.
    """

    def reset(self, rng):
        torque_rng, _ = jax.random.split(rng)
        extra_torque = jax.random.uniform(
            torque_rng, minval=EXTRA_TORQUE_MIN, maxval=EXTRA_TORQUE_MAX,
        )
        return TorqueLeverEnvState(
            theta=jp.deg2rad(SIM_PARAMS['lever_range'][1]),  # lever_max
            theta_dot=jp.array(0.0),
            extra_torque=extra_torque,
            step_count=jp.array(0),
            reward=jp.array(0.0),
            done=jp.array(0.0),  # float32 to match rollout's .astype(float) on done
        )

    def step(self, state, action):
        force = action[0] * FORCE_SCALE
        sim = TorqueLeverSimulationJAX(
            **dict(SIM_PARAMS, motor_extra_torque=state.extra_torque)
        )
        (new_theta, new_theta_dot, _), _ = sim.step(
            (state.theta, state.theta_dot, None),
            FixedForce(force),
        )
        reward = -jp.abs(new_theta - _TARGET_RAD)
        new_step_count = state.step_count + 1
        done = (new_step_count >= EPISODE_STEPS).astype(jp.float32)
        return TorqueLeverEnvState(
            theta=new_theta,
            theta_dot=new_theta_dot,
            extra_torque=state.extra_torque,
            step_count=new_step_count,
            reward=reward,
            done=done,
        )
