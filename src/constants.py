"""Shared constants for Quetsal."""

from __future__ import annotations

__all__ = [
    "HERON_R2_BASIS",
    "ACTION_LABELS",
    "NUM_ACTIONS",
    "NUM_GATE_TYPES",
    "NODE_DIM",
    "EDGE_DIM",
    "GATE_INDEX",
    "CLIFFORD_GATES",
    "SKIP_GATES",
    "REWARD_PRIMARY",
    "DEPTH_PENALTY_WEIGHT",
    "TRUNCATION_PENALTY",
    "MAX_NODES",
    "MAX_EDGES",
    "MAX_STEPS_PER_EPISODE",
    "MIN_STEPS_BEFORE_STOP",
    "TRAINING_MODE",
    "TRAINING_MODE_ARGS",
]

# ---------------------------------------------------------------------------
# Target hardware basis  (IBM Heron r2)
# Backends: ibm_kingston, ibm_marrakesh, ibm_fez, ibm_torino
# ---------------------------------------------------------------------------
HERON_R2_BASIS: list[str] = ["cz", "id", "rx", "rz", "rzz", "sx", "x"]

# ---------------------------------------------------------------------------
# Action space — Qiskit optimization passes available to the RL agent
#
# Source: Qiskit 2.3 builtin_plugins.py → OptimizationPassManager.pass_manager()
# Includes all Rust-backed optimization passes — not just those used by the
# default levels.  The agent learns which passes are useful and in what order;
# restricting to only L1-L3 defaults would cap performance at those levels.
#
# Termination logic in default Qiskit (for reference):
#   L1, L2 — fixed-point check: stops when size AND depth stop decreasing
#   L3     — minimum-point check: tracks best-seen, allows temporary increases
#   Quetsal replaces this with DoNothing as learned termination signal.
#
# Excluded from agent action space (but some run automatically in the env):
#   ContractIdleWiresInControlFlow — wire cleanup; auto-run by env after each step
#   GatesInBasis + translation     — basis check + conditional re-translate;
#   VF2PostLayout / ApplyLayout    — layout re-optimisation (L3 post_loop only)
#   OptimizeCliffordT              — Clifford+T path only; Heron r2 is not Clifford+T
# ---------------------------------------------------------------------------
ACTION_LABELS: list[str] = [
    "Optimize1qGatesDecomposition",  # 0  L1,L2,L3 — 1q gate chain decomposition
    "InverseCancellation",  # 1  L1       — back-to-back inverse cancellation
    "CommutativeCancellation",  # 2  L2,L3    — commutation-based cancellation
    "ConsolidateAndSynthesize",  # 3  macro    — ConsolidateBlocks → UnitarySynthesis
    #              Qiskit never uses CB without US;
    #              combining removes the 2-step credit
    #              assignment problem entirely.
    "RemoveIdentityEquivalent",  # 4  L2,L3    — approx-aware identity removal
    "DoNothing",  # 5  terminate episode
]
NUM_ACTIONS: int = len(ACTION_LABELS)

# ---------------------------------------------------------------------------
# Graph encoder — node / edge dimensions
# ---------------------------------------------------------------------------
NUM_GATE_TYPES: int = 6  # one-hot length for gate vocabulary
NODE_DIM: int = 10  # total node feature dimension
EDGE_DIM: int = 6  # total edge feature dimension (src_role[3] + dst_role[3])

# Gate vocabulary — strictly Heron r2 native basis gates only.
# No ecr/cx aliases — those are Eagle-era gates absent on Heron r2 hardware.
# A properly transpiled ISA circuit on any target backend will only contain
# these six gate names in the optimisation-stage DAG.
GATE_INDEX: dict[str, int] = {
    "cz": 0,
    "rz": 1,
    "rx": 2,
    "sx": 3,
    "x": 4,
    "rzz": 5,
}

# Clifford gates among the 6:
#   cz, sx, x  → Clifford  (singly-controlled Pauli / standard generators)
#   rz, rx, rzz → NOT Clifford  (continuous rotation; Clifford group is finite)
CLIFFORD_GATES: frozenset[str] = frozenset({"cz", "sx", "x"})

# Gates skipped entirely when building the graph.
# These are absent at the optimisation stage:
#   measure / barrier — not touched by any optimisation pass
#   id                — removed by optimisation passes before Quetsal sees DAG
#   delay             — inserted by scheduling stage, which runs AFTER optimisation
#   reset             — mid-circuit reset; not an optimisation target
SKIP_GATES: frozenset[str] = frozenset(
    {
        "measure",
        "barrier",
        "id",
        "delay",
        "reset",
    }
)

# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------
REWARD_PRIMARY = "2q_reduction"  # (2q_before - 2q_after) / 2q_initial
DEPTH_PENALTY_WEIGHT: float = (
    0  # normalized depth change penalty; tune during ablations
)
TRUNCATION_PENALTY: float = 0  # penalty when episode hits MAX_STEPS without DoNothing

# ---------------------------------------------------------------------------
# Observation padding — fixed sizes for SB3 rollout buffer compatibility.
#
# SB3 pre-allocates numpy buffers of shape (n_steps, n_envs, *space.shape).
# Variable-size graph obs would fail this allocation, so we pad to fixed
# MAX_NODES / MAX_EDGES and add boolean masks to mark real vs padding.
# The GNN policy strips padding using the masks before message passing.
#
# Sizing rationale (worst case: 8-qubit Clifford-SU4-SU8, n_blocks=16):
#   Each SU(8) → ~30 basis gates after synthesis; 16 blocks × 30 ≈ 480 nodes.
#   Edges ≈ 2× nodes for typical sequential circuits.
#   1024/2048 gives ~2× safety margin.
# ---------------------------------------------------------------------------
MAX_NODES: int = 3000  # padded buffer size; generation filter uses 60% of this
MAX_EDGES: int = 6000

# ---------------------------------------------------------------------------
# Episode limits
# ---------------------------------------------------------------------------
MAX_STEPS_PER_EPISODE: int = 20
MIN_STEPS_BEFORE_STOP: int = 2  # DoNothing is ignored until this many steps have run

# ---------------------------------------------------------------------------
# Training mode
#   TRAINING_MODE = 0  QUICK  — smoke-test only; verifies full pipeline
#                               runs without errors in ~2-3 min on CPU
#   TRAINING_MODE = 1  FULL   — proper training run (default)
#
# train.py reads this and overrides its argparse defaults accordingly.
# Override from CLI with --mode 0 or --mode 1 to ignore this setting.
# ---------------------------------------------------------------------------
TRAINING_MODE: int = 1  # 0 = quick, 1 = full

# Args applied per mode (used by train.py)
TRAINING_MODE_ARGS: dict[int, dict] = {
    0: {  # QUICK — just verify nothing breaks
        "count_per_family": 3,
        "total_steps": 2048,
        "n_steps": 512,
        "n_epochs": 5,
        "checkpoint_freq": 2048,
    },
    1: {  # FULL — proper training
        "count_per_family": 100,
        "total_steps": 100_000,
        "n_steps": 512,
        "n_epochs": 10,
        "checkpoint_freq": 10_000,
    },
}
