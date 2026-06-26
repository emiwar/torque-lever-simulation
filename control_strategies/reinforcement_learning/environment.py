import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax
import jax.numpy as jp
from dataclasses import dataclass, replace as _dc_replace

from simulation import TorqueLeverSimulationJAX, SIM_PARAMS
from strategies import FixedForce
from evaluate import TARGET_DEG, BOUND_DEG, EXTRA_TORQUE_MIN, EXTRA_TORQUE_MAX

# --- Trial structure (one trial = one episode = one press attempt) ------------
MAX_PRESS_STEPS = 3000   # 3 s at dt=1 ms (protocol maxPressMs)
MIN_PRESS_STEPS = 100    # presses shorter than 100 ms don't count as a release

# Lever boundaries (protocol.md, converted to radians).
ONSET_RAD     = jp.deg2rad(85.156)   # 400 clicks: press onset (trial becomes active)
OFFSET_RAD    = jp.deg2rad(89.551)   # 450 clicks: press offset (lever sprung back up)
OVERSWING_RAD = jp.deg2rad(17.822)   # 200 clicks: rebound above deepest point ends trial

# --- Reward (protocol CalcReward: integer 1..5 by closeness, then x3) ---------
FORCE_SCALE  = 5.0   # tanh action in [-1, 1] -> force in [-5, 5] N
REWARD_SCALE = 3     # rewardScaleFactor: rewards in {3, 6, 9, 12, 15}

_TARGET_RAD = jp.deg2rad(TARGET_DEG)
_BOUND_RAD  = jp.deg2rad(BOUND_DEG)   # half-width of the target zone (boundSize)


@dataclass
class TorqueLeverEnvState:
    theta:        jax.Array   # lever angle (radians)
    theta_dot:    jax.Array   # angular velocity (rad/s)
    min_theta:    jax.Array   # deepest angle reached this trial (radians)
    pressed:      jax.Array   # latched: lever has crossed the press-onset boundary
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
        dist = jp.abs(self.min_theta - _TARGET_RAD)
        return {
            'theta_deg':     jp.rad2deg(self.theta),
            'min_theta_deg': jp.rad2deg(self.min_theta),
            'extra_torque':  self.extra_torque,
            'in_zone':       (dist <= _BOUND_RAD).astype(jp.float32),
        }


jax.tree_util.register_pytree_node(
    TorqueLeverEnvState,
    lambda s: (
        [s.theta, s.theta_dot, s.min_theta, s.pressed,
         s.extra_torque, s.step_count, s.reward, s.done],
        None,
    ),
    lambda _, xs: TorqueLeverEnvState(*xs),
)


def _calc_reward(min_theta):
    """Reward for a finished trial, mirroring the microcontroller's CalcReward.

    The deepest press point determines the reward: an integer 1..5 by closeness
    to the target, scaled by REWARD_SCALE -> {3, 6, 9, 12, 15}, or 0 if the
    deepest point fell outside the target zone (target +/- boundSize).
    """
    dist  = jp.abs(_TARGET_RAD - min_theta)
    ratio = jp.clip(1.0 - dist / _BOUND_RAD, 0.0, 1.0)
    base  = jp.clip(1.0 + jp.floor(ratio * 5.0), 1.0, 5.0)
    in_zone = dist <= _BOUND_RAD
    return jp.where(in_zone, base * REWARD_SCALE, 0.0)


class TorqueLeverEnv:
    """RL environment wrapping TorqueLeverSimulationJAX (Torque Lever Task).

    One episode is one trial: the lever starts raised and the agent presses it
    down against a motor resistance that ramps from a 0.1 Nm baseline to a
    per-trial challenge torque once the press-onset boundary is crossed. The
    trial ends when the lever springs back up past the offset boundary, rebounds
    OVERSWING_RAD above its deepest point, or after MAX_PRESS_STEPS. A single
    reward is delivered on the terminal step, sized by how close the deepest
    press point came to the target zone (zero on every other step).

    The motor challenge torque is randomised at each reset, so the agent must
    learn to be robust across the full range of resistance levels.
    """

    def reset(self, rng):
        torque_rng, _ = jax.random.split(rng)
        extra_torque = jax.random.uniform(
            torque_rng, minval=EXTRA_TORQUE_MIN, maxval=EXTRA_TORQUE_MAX,
        )
        lever_max = jp.deg2rad(SIM_PARAMS['lever_range'][1])
        return TorqueLeverEnvState(
            theta=lever_max,                 # lever starts raised
            theta_dot=jp.array(0.0),
            min_theta=lever_max,
            pressed=jp.array(False),
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
        (new_theta, new_theta_dot, new_min_theta, _), _ = sim.step(
            (state.theta, state.theta_dot, state.min_theta, None),
            FixedForce(force),
        )

        # Trial becomes (and stays) "pressed" once it crosses the onset boundary.
        pressed = jp.logical_or(state.pressed, new_theta < ONSET_RAD)

        new_step_count = state.step_count + 1
        long_enough = new_step_count >= MIN_PRESS_STEPS

        # Termination: lever sprung back up past the offset boundary, rebounded
        # OVERSWING_RAD above its deepest point, or the press timed out.
        released  = pressed & long_enough & (new_theta >= OFFSET_RAD)
        overswing = pressed & long_enough & (new_theta >= new_min_theta + OVERSWING_RAD)
        timeout   = new_step_count >= MAX_PRESS_STEPS
        done = (released | overswing | timeout).astype(jp.float32)

        # A single reward is delivered on the terminal step only.
        reward = jp.where(done > 0, _calc_reward(new_min_theta), 0.0)

        return TorqueLeverEnvState(
            theta=new_theta,
            theta_dot=new_theta_dot,
            min_theta=new_min_theta,
            pressed=pressed,
            extra_torque=state.extra_torque,
            step_count=new_step_count,
            reward=reward,
            done=done,
        )
