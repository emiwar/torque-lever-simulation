"""Train the TorqueLeverEnv agent with PPO, logging to Weights & Biases.

Mirror of train.py, but metrics are streamed to WandB instead of stdout.
Modelled on nnx-ppo's examples/wandb_logging.py: PPO calls `log_fn(metrics, step)`
each iteration, and `wandb.log` accepts exactly that signature, so it is passed
straight through. No video is logged — TorqueLeverEnv has no renderer.

Run with:  WANDB_MODE=online python train_wandb.py
(use WANDB_MODE=offline or =disabled to run without a network/account).
"""

import sys
import os
import dataclasses
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, '/home/emil/Development/vnl/nnx-ppo')

import wandb
from flax import nnx
from nnx_ppo.algorithms.ppo import train_ppo, default_config
from nnx_ppo.algorithms.checkpointing import make_checkpoint_fn
from nnx_ppo.algorithms.types import LoggingLevel
from nnx_ppo.networks.factories import make_mlp_actor_critic

from environment import TorqueLeverEnv, MAX_PRESS_STEPS

OBS_SIZE    = 2   # [theta - target, theta_dot]
ACTION_SIZE = 1   # [force]
SEED        = 0

env = TorqueLeverEnv()

networks = make_mlp_actor_critic(
    obs_size=OBS_SIZE,
    action_size=ACTION_SIZE,
    actor_hidden_sizes=[64, 64],
    critic_hidden_sizes=[64, 64],
    rngs=nnx.Rngs(SEED),
    normalize_obs=True,
    entropy_weight=1e-2,
)

config = default_config()
config.seed               = SEED
config.ppo.n_envs         = 2048
config.ppo.total_steps    = 500_000_000
# Episodes are variable length (a press to the target then release; up to the
# 3 s MAX_PRESS_STEPS cap). The reward is sparse (one per trial, at release), and
# the rollout auto-resets on `done`, so this length just sets the rollout horizon.
config.ppo.rollout_length = 1000
config.ppo.learning_rate  = 3e-4

config.ppo.logging_level        = LoggingLevel.ALL
config.ppo.logging_percentiles  = (0, 25, 50, 75, 100)
config.eval.logging_level       = LoggingLevel.ALL
config.eval.logging_percentiles = (0, 25, 50, 75, 100)
# Eval episodes must be allowed to run to the full trial length, not truncated.
config.eval.max_episode_length  = MAX_PRESS_STEPS


CHECKPOINT_ROOT = os.path.join(os.path.dirname(__file__), 'checkpoints')

if __name__ == '__main__':
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    run = wandb.init(
        project='torque-lever-rl',
        name=f'torque-lever-{timestamp}',
        config={'config': dataclasses.asdict(config)},
        tags=[],
    )

    # De-conflict checkpoints per run, keyed by the WandB run name (falling back
    # to the timestamp if WandB is disabled and assigns no name).
    run_name = run.name or f'torque-lever-{timestamp}'
    CHECKPOINT_DIR = os.path.join(CHECKPOINT_ROOT, run_name)

    checkpoint_fn = make_checkpoint_fn(CHECKPOINT_DIR, config=config)
    config.checkpoint_every_steps = 100_000_000

    result = train_ppo(
        env=env, networks=networks, config=config,
        log_fn=wandb.log, checkpoint_fn=checkpoint_fn,
    )

    # Save final weights (train_ppo doesn't guarantee a save at the last step)
    checkpoint_fn(result.training_state, result.total_steps)
    print(f'Done. Checkpoint saved to {CHECKPOINT_DIR}/step_{result.total_steps:010d}/')

    final = result.eval_history[-1].get('eval/episode_reward/mean', 'N/A') if result.eval_history else 'N/A'
    print(f'Final eval reward: {final}')
    wandb.finish()
