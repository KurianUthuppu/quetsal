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
    "DEPTH_PENALTY_WEIGHT",
    "TERMINAL_BONUS",
    "STEP_PENALTY",
    "TRUNCATION_PENALTY",
    "MAX_NODES",
    "MAX_EDGES",
    "MAX_STEPS_PER_EPISODE",
    "MIN_STEPS_BEFORE_STOP",
    "TRAINING_MODE",
    "TRAINING_MODE_ARGS",
    "CURRICULUM_STAGES",
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
    # "RemoveIdentityEquivalent",  # 4  L2,L3    — approx-aware identity removal (disabled)
    "DoNothing",  # 4  terminate episode
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
DEPTH_PENALTY_WEIGHT: float = (
    0.03  # normalized depth change penalty; tune during ablations
)
TRUNCATION_PENALTY: float = (
    0.05  # penalty when episode hits MAX_STEPS without DoNothing
)
TERMINAL_BONUS: float = 0.1
STEP_PENALTY: float = 0.001  # small cost per non-DoNothing action; incentivises
# the agent to stop unless a pass genuinely helps

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
MIN_STEPS_BEFORE_STOP: int = 3  # DoNothing is ignored until this many steps have run

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
        "n_epochs": 3,
        "checkpoint_freq": 512,  # fire eval 4× so curriculum callback runs multiple times
        # curriculum smoke-test: lower promotion gates so all 3 stages run in 2048 steps
        "curriculum_smoke": {
            "eval_mean_reward_min": 0.0,  # always passes
            "donothing_per_ep_max": 9999,  # always passes
            "reward_delta_max": 9999,  # always passes
            "consecutive_evals": 1,  # promote after 1 passing eval, not 2
        },
    },
    1: {  # FULL — proper training
        "count_per_family": 100,
        "total_steps": 300_000,
        "n_steps": 1024,
        "n_epochs": 5,
        "checkpoint_freq": 10_000,
    },
}

# ---------------------------------------------------------------------------
# Curriculum learning — staged training schedule
#
# Stage 1 — Foundation: non-parametric families only (no random angles).
#   Agent learns structural optimisation (cancellation, consolidation) on
#   fixed-gate circuits before seeing parametric noise.
#   Promoted when eval_mean_reward > 0.25 AND donothing_per_ep < 1.5
#   for 2 consecutive eval checkpoints.
#
# Stage 2 — Parametric exposure: blend in QAOA/IQP/EfficientSU2 gradually.
#   Starts at 30% parametric mix, increases +10% per passing eval checkpoint
#   (max 60%). Entropy decays 0.05 → 0.02 over the stage.
#   Promoted when reward stabilises (δ < 0.02 over last 2 evals).
#
# Stage 3 — Balanced fine-tune: all 7 families, equal weight.
#   Low LR (1e-4), tight clip range (0.1), early-exit penalty for DoNothing
#   before MIN_STEPS_BEFORE_STOP to prevent lazy termination.
#
# Used only when train.py is invoked with --curriculum.
# ---------------------------------------------------------------------------
_ALL_FAMILIES = [
    "qv",
    "qaoa",
    "clifford_su4_su8",
    "clifford_su4",
    "iqp",
    "efficient_su2",
    "real_amplitudes",
]

CURRICULUM_STAGES: dict[int, dict] = {
    1: {
        "families": ["qv", "clifford_su4", "clifford_su4_su8"],
        "family_weights": {"qv": 0.40, "clifford_su4": 0.35, "clifford_su4_su8": 0.25},
        "max_steps": 150_000,  # foundation — non-parametric structural learning
        "ent_coef": 0.03,
        "promotion": {
            "eval_mean_reward_min": 0.25,  # unified name across all stages
            "donothing_per_ep_max": 1.5,
            "consecutive_evals": 2,
        },
    },
    2: {
        "families": [
            "qv",
            "clifford_su4",
            "clifford_su4_su8",
            "qaoa",
            "iqp",
            "efficient_su2",
        ],
        "nonparam_families": ["qv", "clifford_su4", "clifford_su4_su8"],
        "param_families": ["qaoa", "iqp", "efficient_su2"],
        "param_start_mix": 0.30,
        "param_blend_step": 0.10,
        "param_max_mix": 0.60,
        "max_steps": 100_000,  # parametric blend
        "ent_coef_start": 0.05,
        "ent_coef_end": 0.02,
        "step_penalty": 0.003,
        "promotion": {
            "eval_mean_reward_min": 0.10,  # same key as stage 1
            "reward_delta_max": 0.02,  # |reward[t] - reward[t-1]| < 0.02 (stability, not clip_range)
            "consecutive_evals": 2,
        },
    },
    3: {
        "families": _ALL_FAMILIES,
        "family_weights": {f: 1 / 7 for f in _ALL_FAMILIES},
        "max_steps": 50_000,  # balanced fine-tune  — total 300K
        "lr": 1e-4,
        "ent_coef": 0.01,
        "clip_range": 0.1,
        "donothing_early_penalty": 0.05,  # penalise DoNothing before MIN_STEPS_BEFORE_STOP
    },
}
