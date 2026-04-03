# =============================================================================
# quetsal/src/agent/ppo_agent.py
# Thin SB3 PPO wrapper for the Quetsal RL agent.
#
# Responsibilities:
#   - Instantiate SB3 PPO with QuetsalGNNPolicy
#   - Enforce the batch_size == n_steps constraint (single minibatch per update)
#   - Provide save / load helpers
#   - Expose a predict() method for eval/benchmarking use
#
# Training entry point is training/train.py — this file only wraps the model.
# =============================================================================

from __future__ import annotations

__all__ = ["make_ppo_agent", "save_agent", "load_agent", "predict_action"]

from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO

from quetsal.src.agent.gnn_policy import QuetsalGNNPolicy


# ── Defaults ──────────────────────────────────────────────────────────────────

# n_steps: number of env steps collected per rollout before each PPO update.
# batch_size must equal n_steps (workaround for variable-size graph obs — see gnn_policy.py).
# See gnn_policy.py SB3 batching note for explanation.
DEFAULT_N_STEPS    = 512
DEFAULT_N_EPOCHS   = 10       # PPO gradient steps per rollout
DEFAULT_GAMMA      = 0.99     # discount factor
DEFAULT_LR         = 3e-4     # learning rate
DEFAULT_CLIP_RANGE    = 0.2    # PPO clip epsilon
DEFAULT_ENT_COEF      = 0.01  # entropy bonus (encourages exploration)
DEFAULT_GAE_LAMBDA    = 0.95  # GAE smoothing (0=TD, 1=Monte Carlo)
DEFAULT_VF_COEF       = 0.5   # value function loss weight
DEFAULT_MAX_GRAD_NORM = 0.5   # gradient clipping threshold

# GNN architecture defaults — must match what GNNFeaturesExtractor expects
DEFAULT_HIDDEN_DIM = 64
DEFAULT_NUM_LAYERS = 3
DEFAULT_LATENT_DIM = 64


# ── Factory ───────────────────────────────────────────────────────────────────


def make_ppo_agent(
    env: gym.Env,
    n_steps: int = DEFAULT_N_STEPS,
    n_epochs: int = DEFAULT_N_EPOCHS,
    gamma: float = DEFAULT_GAMMA,
    learning_rate: float = DEFAULT_LR,
    clip_range: float = DEFAULT_CLIP_RANGE,
    ent_coef: float = DEFAULT_ENT_COEF,
    gae_lambda: float = DEFAULT_GAE_LAMBDA,
    vf_coef: float = DEFAULT_VF_COEF,
    max_grad_norm: float = DEFAULT_MAX_GRAD_NORM,
    hidden_dim: int = DEFAULT_HIDDEN_DIM,
    num_layers: int = DEFAULT_NUM_LAYERS,
    latent_dim: int = DEFAULT_LATENT_DIM,
    seed: int = 42,
    verbose: int = 1,
    device: str = "auto",
) -> PPO:
    """Create a PPO agent with the QuetsalGNNPolicy.

    batch_size is set equal to n_steps (one minibatch per
    update avoids SB3 trying to stack variable-size graph tensors).

    Parameters
    ----------
    env           : PassManagerEnv instance (or VecEnv wrapper).
    n_steps       : rollout length before each PPO update.
    n_epochs      : gradient update iterations per rollout.
    gamma         : discount factor for future rewards.
    learning_rate : Adam learning rate for the GNN + heads.
    clip_range    : PPO clipping parameter (epsilon).
    ent_coef      : entropy coefficient — higher = more exploration.
    gae_lambda    : GAE smoothing factor (0=TD, 1=Monte Carlo).
    vf_coef       : weight of value function loss in total loss.
    max_grad_norm : gradient clipping threshold.
    hidden_dim    : GINEConv MLP hidden width.
    num_layers    : number of GINEConv message-passing layers.
    latent_dim    : output size of the GNN encoder.
    seed          : RNG seed for reproducibility.
    verbose       : SB3 verbosity level (0=silent, 1=info, 2=debug).
    device        : "auto" selects GPU if available, else CPU.

    Returns
    -------
    SB3 PPO model ready for .learn() calls.
    """
    return PPO(
        policy           = QuetsalGNNPolicy,
        env              = env,
        n_steps          = n_steps,
        batch_size       = n_steps,   # single minibatch = full rollout (see gnn_policy.py)
        n_epochs         = n_epochs,
        gamma            = gamma,
        learning_rate    = learning_rate,
        clip_range       = clip_range,
        ent_coef         = ent_coef,
        gae_lambda       = gae_lambda,
        vf_coef          = vf_coef,
        max_grad_norm    = max_grad_norm,
        policy_kwargs    = dict(
            hidden_dim = hidden_dim,
            num_layers = num_layers,
            latent_dim = latent_dim,
        ),
        seed             = seed,
        verbose          = verbose,
        device           = device,
    )


# ── Save / load helpers ───────────────────────────────────────────────────────


def save_agent(model: PPO, path: str | Path) -> None:
    """Save PPO model weights to disk.

    SB3 appends .zip automatically if not present.

    Parameters
    ----------
    model : trained PPO model.
    path  : file path (e.g. "checkpoints/quetsal_10k").
    """
    model.save(str(path))


def load_agent(path: str | Path, env: gym.Env, device: str = "auto") -> PPO:
    """Load a saved PPO model from disk.

    Parameters
    ----------
    path   : file path to the saved .zip (with or without extension).
    env    : environment to bind the model to (needed for predict()).
    device : device to load weights onto.

    Returns
    -------
    PPO model with weights loaded, ready for .predict() or continued .learn().
    """
    return PPO.load(str(path), env=env, device=device)


# ── Predict helper ────────────────────────────────────────────────────────────


def predict_action(
    model: PPO,
    obs: dict,
    deterministic: bool = True,
) -> int:
    """Select an action for a single observation.

    Wraps model.predict() and returns just the action integer.

    Parameters
    ----------
    model         : trained PPO model.
    obs           : dict observation from PassManagerEnv.reset() or .step().
    deterministic : True = take argmax of logits (eval mode).
                    False = sample from policy distribution (exploration).

    Returns
    -------
    int action index in [0, NUM_ACTIONS).
    """
    action, _ = model.predict(obs, deterministic=deterministic)
    return int(action)
