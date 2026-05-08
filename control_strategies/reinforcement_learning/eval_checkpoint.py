import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, '/home/emil/Development/vnl/nnx-ppo')

import jax
import jax.numpy as jp
import numpy as np
import matplotlib.pyplot as plt
from flax import nnx
from nnx_ppo.algorithms.checkpointing import load_checkpoint
from nnx_ppo.algorithms.ppo import new_training_state, default_config
from nnx_ppo.networks.factories import make_mlp_actor_critic

from evaluate import PLOT_TORQUES, plot_trajectory_panel
from environment import TorqueLeverEnv, TorqueLeverEnvState, EPISODE_STEPS
from simulation import SIM_PARAMS

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), 'checkpoints')
OBS_SIZE    = 2
ACTION_SIZE = 1


def latest_checkpoint(directory):
    steps = sorted(
        int(d.split('_')[1])
        for d in os.listdir(directory)
        if d.startswith('step_')
    )
    return os.path.join(directory, f'step_{steps[-1]:010d}'), steps[-1]


def sample_trajectories_rl(networks, env):
    """Run the network on all PLOT_TORQUES in parallel using nnx.scan.

    Returns an array of shape (n_torques, n_steps) with lever angles in radians.
    """
    n_torques = len(PLOT_TORQUES)

    # Build one initial env state per torque level (no randomisation)
    initial_states = jax.vmap(lambda et: TorqueLeverEnvState(
        theta=jp.deg2rad(SIM_PARAMS['lever_range'][1]),
        theta_dot=jp.array(0.0),
        extra_torque=et,
        step_count=jp.array(0),
        reward=jp.array(0.0),
        done=jp.array(0.0),
    ))(jp.array(PLOT_TORQUES, dtype=jp.float32))

    net_states = networks.initialize_state(n_torques)

    def step_fn(networks, carry):
        env_state, net_state = carry
        next_net_state, network_output = networks(net_state, env_state.obs)
        next_env_state = jax.vmap(env.step)(env_state, network_output.actions)
        return (next_env_state, next_net_state), next_env_state.theta

    scan_fn = nnx.scan(
        step_fn,
        in_axes=(nnx.StateAxes({...: nnx.Carry}), nnx.Carry),
        out_axes=(nnx.Carry, 0),
        length=EPISODE_STEPS,
    )

    _, thetas = scan_fn(networks, (initial_states, net_states))
    return thetas.T   # (n_steps, n_torques) → (n_torques, n_steps)


if __name__ == '__main__':
    ckpt_path, step = latest_checkpoint(CHECKPOINT_DIR)
    print(f'Loading checkpoint: {ckpt_path}')

    networks = make_mlp_actor_critic(
        obs_size=OBS_SIZE, action_size=ACTION_SIZE,
        actor_hidden_sizes=[64, 64], critic_hidden_sizes=[64, 64],
        rngs=nnx.Rngs(0), normalize_obs=True, entropy_weight=1e-2,
    )
    env = TorqueLeverEnv()
    ts = new_training_state(env, networks, n_envs=1, seed=0,
                            learning_rate=default_config().ppo.learning_rate)
    load_checkpoint(ckpt_path, networks=ts.networks, optimizer=ts.optimizer)
    networks = ts.networks
    networks.eval()

    print('Running trajectories...')
    thetas = sample_trajectories_rl(networks, env)

    fig, ax = plt.subplots(figsize=(7, 4))
    plot_trajectory_panel(ax, thetas, title=f'RL agent (step {step:,})')
    plt.tight_layout()
    out = os.path.join(os.path.dirname(__file__), 'eval_checkpoint.png')
    plt.savefig(out, dpi=120)
    print(f'Saved {out}')
    plt.show()
