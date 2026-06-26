import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, '/home/emil/Development/vnl/nnx-ppo')

from flax import nnx
from nnx_ppo.algorithms.ppo import train_ppo, default_config
from nnx_ppo.algorithms.checkpointing import make_checkpoint_fn
from nnx_ppo.networks.factories import make_mlp_actor_critic

from environment import TorqueLeverEnv

OBS_SIZE    = 2   # [theta - target, theta_dot]
ACTION_SIZE = 1   # [force]

env = TorqueLeverEnv()

networks = make_mlp_actor_critic(
    obs_size=OBS_SIZE,
    action_size=ACTION_SIZE,
    actor_hidden_sizes=[64, 64],
    critic_hidden_sizes=[64, 64],
    rngs=nnx.Rngs(0),
    normalize_obs=True,
    entropy_weight=1e-2,
)

config = default_config()
config.ppo.n_envs         = 2048
config.ppo.total_steps    = 500_000_000
# Episodes are now variable length (a press to the target then release; up to the
# 3 s MAX_PRESS_STEPS cap). The reward is sparse (one per trial, at release), and
# the rollout auto-resets on `done`, so this length just sets the rollout horizon.
config.ppo.rollout_length = 1000
config.ppo.learning_rate  = 3e-4


def log_fn(metrics, step):
    # nnx-ppo >=0.3.0: all eval metrics are `eval/`-prefixed, and lifespan now
    # follows the percentile convention (p{N}, not `lifespan_mean`).
    if 'eval/episode_reward/p50' in metrics:
        r50  = metrics['eval/episode_reward/p50']
        r0   = metrics['eval/episode_reward/p0']
        r100 = metrics['eval/episode_reward/p100']
        span = metrics['eval/lifespan/p50']
        print(f'step {step:>8d}  reward median={r50:.3f}  [{r0:.3f}, {r100:.3f}]  lifespan(median)={span:.1f}')


CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), 'checkpoints')

if __name__ == '__main__':
    checkpoint_fn = make_checkpoint_fn(CHECKPOINT_DIR, config=config)
    config.checkpoint_every_steps = 100_000_000

    result = train_ppo(
        env=env, networks=networks, config=config,
        log_fn=log_fn, checkpoint_fn=checkpoint_fn,
    )

    # Save final weights (train_ppo doesn't guarantee a save at the last step)
    checkpoint_fn(result.training_state, result.total_steps)
    print(f'Done. Checkpoint saved to {CHECKPOINT_DIR}/step_{result.total_steps:010d}/')
