# =============================================================================
# quetsal/src/agent/gnn_policy.py
# GINEConv-based GNN feature extractor + ActorCriticPolicy for SB3 PPO.
#
# Architecture:
#   obs dict {x, edge_index, edge_attr}
#     → _obs_to_pyg_batch()           reconstruct PyG Batch from SB3 tensors
#     → GINEConv × num_layers          message passing (uses both node + edge feats)
#     → global_mean_pool               variable graph → fixed-size latent vector
#     → Linear(hidden_dim, latent_dim) projection
#     → actor head: Linear(latent_dim, NUM_ACTIONS)   → action logits
#     → critic head: Linear(latent_dim, 1)             → state value
#
# GINEConv vs GINConv:
#   GINConv  — aggregates only node features, ignores edge attributes.
#   GINEConv — transforms edge_attr to match node dim, adds it during
#              aggregation.  We use GINEConv because our edge features
#              (qubit role one-hots) carry meaningful structural information.
#
# SB3 batching note:
#   SB3's RolloutBuffer stores observations as stacked numpy arrays.  For
#   graphs with variable node/edge counts this stacking fails.  Current
#   workaround: set batch_size == n_steps in PPO so there is exactly one
#   minibatch per update — the full rollout is processed together via PyG
#   Batch, avoiding any cross-graph stacking.  A custom RolloutBuffer that
#   stores Data objects natively would remove this constraint.
# =============================================================================

from __future__ import annotations

__all__ = ["GNNFeaturesExtractor", "QuetsalGNNPolicy"]

from typing import Any

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import Schedule
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GINEConv, global_mean_pool

from quetsal.src.constants import EDGE_DIM, NODE_DIM


# ── Feature extractor ─────────────────────────────────────────────────────────


class GNNFeaturesExtractor(BaseFeaturesExtractor):
    """GINEConv encoder that maps a circuit DAG graph to a fixed-size vector.

    Plugs into SB3 as a features_extractor_class.  Accepts the Dict
    observation space produced by PassManagerEnv and outputs a 1-D feature
    vector of size `latent_dim` per graph.

    Parameters
    ----------
    observation_space : gym.spaces.Dict  (x, edge_index, edge_attr)
    hidden_dim        : width of each GINEConv MLP (default 64)
    num_layers        : number of GINEConv message-passing rounds (default 3)
    latent_dim        : size of the output feature vector (default 64)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        hidden_dim: int = 64,
        num_layers: int = 3,
        latent_dim: int = 64,
    ) -> None:
        # BaseFeaturesExtractor sets self._features_dim = latent_dim
        super().__init__(observation_space, features_dim=latent_dim)

        self.convs = nn.ModuleList()
        in_dim = NODE_DIM
        for _ in range(num_layers):
            # MLP inside each GINEConv: two linear layers with BN + ReLU
            mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            # edge_dim tells GINEConv to project edge_attr to match node dim
            self.convs.append(GINEConv(mlp, edge_dim=EDGE_DIM))
            in_dim = hidden_dim

        # Skip connection projection for layer 0 only:
        # input x is NODE_DIM=10, conv output is hidden_dim=64 — dimensions
        # must match before the residual addition.  Layers 1+ are hidden_dim→hidden_dim
        # so they add directly without projection.
        self.skip_proj = nn.Linear(NODE_DIM, hidden_dim, bias=False)

        # Final projection to latent space
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Parameters
        ----------
        observations : dict with keys "x", "edge_index", "edge_attr"
            Tensors as provided by SB3 from the rollout buffer.

        Returns
        -------
        torch.Tensor of shape [batch_size, latent_dim]
        """
        batch = _obs_to_pyg_batch(observations)

        x = batch.x.float()
        edge_index = batch.edge_index.long()
        edge_attr = batch.edge_attr.float()
        batch_vec = batch.batch  # node → graph index mapping

        # Message passing with skip connections (residual additions).
        # Layer 0: input is NODE_DIM=10, output is hidden_dim=64 — project
        #          the skip path via skip_proj before adding.
        # Layer 1+: both input and output are hidden_dim=64 — add directly.
        for i, conv in enumerate(self.convs):
            h = torch.relu(conv(x, edge_index, edge_attr))
            x = self.skip_proj(x) + h if i == 0 else x + h

        # Graph-level pooling: [total_nodes, hidden_dim] → [B, hidden_dim]
        x = global_mean_pool(x, batch_vec)

        return self.output_proj(x)  # [B, latent_dim]


# ── Obs → PyG Batch conversion ────────────────────────────────────────────────


def _obs_to_pyg_batch(observations: dict[str, torch.Tensor]) -> Batch:
    """Reconstruct a PyG Batch from SB3's padded fixed-size observations.

    Observations from PassManagerEnv are padded to (MAX_NODES, MAX_EDGES)
    with boolean node_mask / edge_mask marking real vs padding entries.

    SB3 passes observations in two shapes:
      - Single step (rollout collection): x shape [MAX_NODES, NODE_DIM]
      - Minibatch (policy update):        x shape [B, MAX_NODES, NODE_DIM]

    In both cases we use the masks to strip padding and build a PyG Batch.

    Returns
    -------
    PyG Batch object with .x, .edge_index, .edge_attr, .batch attributes.
    """
    x_raw         = observations["x"]
    edge_index_raw = observations["edge_index"]
    edge_attr_raw  = observations["edge_attr"]
    node_mask_raw  = observations["node_mask"]
    edge_mask_raw  = observations["edge_mask"]

    # Ensure tensors
    def _t(arr, dtype):
        return torch.as_tensor(arr, dtype=dtype) if not isinstance(arr, torch.Tensor) else arr.to(dtype)

    def _make_data(x_2d, ei_2d, ea_2d, nm, em):
        """Build a single PyG Data, inserting a dummy node if the graph is empty.

        An all-zero node_mask (from _zero_obs after DAG overflow) would produce
        a 0-node graph, causing global_mean_pool to return an empty tensor and
        crashing SB3's bootstrap path (predict_values(terminal_obs)[0]).
        A single zero-feature dummy node avoids this without affecting learning.
        """
        x_real = _t(x_2d, torch.float)[nm]
        em_bool = _t(em, torch.bool)
        ei_real = _t(ei_2d, torch.long)[:, em_bool]
        ea_real = _t(ea_2d, torch.float)[em_bool]
        if x_real.shape[0] == 0:
            x_real  = torch.zeros(1, x_real.shape[1] if x_real.dim() > 1 else NODE_DIM)
            ei_real = torch.zeros(2, 0, dtype=torch.long)
            ea_real = torch.zeros(0, ea_real.shape[1] if ea_real.dim() > 1 else EDGE_DIM)
        return Data(x=x_real, edge_index=ei_real, edge_attr=ea_real)

    # Single observation: x is 2-D [MAX_NODES, NODE_DIM]
    if x_raw.dim() == 2 if isinstance(x_raw, torch.Tensor) else x_raw.ndim == 2:
        nm = _t(node_mask_raw, torch.bool).squeeze()
        return Batch.from_data_list([_make_data(x_raw, edge_index_raw, edge_attr_raw, nm, edge_mask_raw)])

    # Minibatch: x is 3-D [B, MAX_NODES, NODE_DIM]
    b = x_raw.shape[0]
    data_list = []
    for i in range(b):
        nm = _t(node_mask_raw[i], torch.bool).squeeze()
        em = _t(edge_mask_raw[i], torch.bool).squeeze()
        x_real = _t(x_raw[i], torch.float)[nm]
        ei_raw = _t(edge_index_raw[i], torch.long)[:, em]
        ea_real = _t(edge_attr_raw[i], torch.float)[em]
        if x_real.shape[0] == 0:
            data_list.append(Data(
                x          = torch.zeros(1, NODE_DIM),
                edge_index = torch.zeros(2, 0, dtype=torch.long),
                edge_attr  = torch.zeros(0, EDGE_DIM),
            ))
        else:
            real_indices = nm.nonzero(as_tuple=True)[0]
            remap = torch.zeros(nm.shape[0], dtype=torch.long)
            remap[real_indices] = torch.arange(real_indices.shape[0])
            data_list.append(Data(x=x_real, edge_index=remap[ei_raw], edge_attr=ea_real))
    return Batch.from_data_list(data_list)


# ── ActorCriticPolicy ─────────────────────────────────────────────────────────


class QuetsalGNNPolicy(ActorCriticPolicy):
    """SB3 ActorCriticPolicy backed by the GINEConv feature extractor.

    Usage
    -----
    model = PPO(
        QuetsalGNNPolicy,
        env,
        policy_kwargs=dict(
            hidden_dim=64,
            num_layers=3,
            latent_dim=64,
        ),
        n_steps=1024,
        batch_size=1024,  # == n_steps: one minibatch per update (see SB3 batching note)
        ...
    )

    The actor and critic share the GNN encoder (shared trunk) and each
    have their own linear head on top of the latent vector.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        action_space: gym.spaces.Discrete,
        lr_schedule: Schedule,
        hidden_dim: int = 64,
        num_layers: int = 3,
        latent_dim: int = 64,
        **kwargs: Any,
    ) -> None:
        # Pass GNNFeaturesExtractor as the features extractor.
        # net_arch=[] tells SB3 not to add any extra MLP on top of our latent —
        # the actor/critic heads are plain Linear layers over latent_dim.
        kwargs["features_extractor_class"] = GNNFeaturesExtractor
        kwargs["features_extractor_kwargs"] = dict(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            latent_dim=latent_dim,
        )
        kwargs.setdefault("net_arch", [])             # no extra MLP trunk
        kwargs.setdefault("share_features_extractor", True)  # one shared GNN trunk
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)
        # SB3 calls _build() inside super().__init__(), which constructs
        # mlp_extractor + action_net + value_net using our features_extractor.
