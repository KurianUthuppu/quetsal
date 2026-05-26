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
    "TERMINAL_BONUS_SCALE",
    "DEPTH_PRIMARY_FAMILIES",
    "FAMILY_DEPTH_CEILING",
    "FAMILY_REDUCTION_CEILING",
    "DEPTH_PRIMARY_STEP_DEPTH_WEIGHT",
    "DEPTH_PRIMARY_STEP_2Q_WEIGHT",
    "DEPTH_PRIMARY_TERMINAL_BONUS_SCALE",
    "STEP_PENALTY",
    "TRUNCATION_PENALTY",
    "MAX_NODES",
    "MAX_EDGES",
    "GLOBAL_DIM",
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
    "Optimize1qGatesDecomposition",  # 0  L1,L2,L3 — 1q gate chain decomposition.
    # "CommutativeInverseCancellation",  # DISABLED — rarely selected; ablation run without it.
    "ConsolidateAndSynthesize",  # 1  macro — ConsolidateBlocks → UnitarySynthesis.
    #              Qiskit never uses CB without US; combining removes the 2-step credit assignment problem.
    # "OptimizeCliffords",  # DISABLED — rarely selected; ablation run without it.
    # "Split2QUnitaries",              # REMOVED — genuinely subsumed by ConsolidateAndSynthesize.
    "ZXFullReduce",  # 2  pyzx.simplify.full_reduce() via QASM round-trip.
    #              ZX-calculus spider fusion + phase gadget reduction + Clifford simp.
    #              Highest 2q-reduction ceiling; most effective on Clifford-heavy circuits.
    # "RemoveIdentityEquivalent",  # DISABLED — rarely selected; ablation run without it.
    "DoNothing",  # 3  terminate episode
]
NUM_ACTIONS: int = len(ACTION_LABELS)

# ---------------------------------------------------------------------------
# Graph encoder — node / edge dimensions
# ---------------------------------------------------------------------------
NUM_GATE_TYPES: int = 6  # one-hot length for gate vocabulary
NODE_DIM: int = 10  # total node feature dimension
EDGE_DIM: int = 6  # total edge feature dimension (src_role[3] + dst_role[3])
GLOBAL_DIM: int = (
    4  # global feature vector: [step_frac, 2q_ratio, n_qubits_norm, depth_ratio]
)

# Gate vocabulary — strictly Heron r2 native basis gates only.
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
#   rz, rx, rzz → NOT Clifford  (continuous rotation)
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
TERMINAL_BONUS: float = 0.1  # Fixed base bonus when agent calls DoNothing
TERMINAL_BONUS_SCALE: float = 0  # Non-depth-primary terminal scale;
# Terminal reward = TERMINAL_BONUS + TERMINAL_BONUS_SCALE * min(final_reduction / ceiling, 1.0)
# Normalising by the per-family ceiling converts absolute gate reduction into a relative
# Ceilings derived from opt_level=3 benchmark runs; update after major circuit pool changes.
FAMILY_REDUCTION_CEILING: dict[str, float] = {
    "clifford_su4": 0.24,
    "clifford_su4_su8": 0.19,
    "qv": 0.28,
    "random_clifford": 0.23,
    "iqp": 0.02,
    "qaoa": 0.01,
    "efficient_su2": 0.01,
    "real_amplitudes": 0.01,
}

# Families where 2Q reduction is naturally near-zero and depth is the main
# useful optimisation target.  IQP is included because it behaves like the
# parametric ansatz families for reward purposes on the benchmark pool.
DEPTH_PRIMARY_FAMILIES: frozenset[str] = frozenset(
    {"iqp", "qaoa", "efficient_su2", "real_amplitudes"}
)

# Depth reduction ceilings for depth-primary families, used as terminal bonus
# denominator.  Values derived from opt_level=3 benchmark runs on the same pool.
FAMILY_DEPTH_CEILING: dict[str, float] = {
    "iqp": 0.12,
    "qaoa": 0.04,
    "efficient_su2": 0.49,
    "real_amplitudes": 0.46,
}

# Step reward weights for depth-primary families.
# Depth gets immediate credit so the agent can learn which pass caused the win;
# 2Q still gets a small weight for IQP-like edge cases.
DEPTH_PRIMARY_STEP_DEPTH_WEIGHT: float = 0.20
DEPTH_PRIMARY_STEP_2Q_WEIGHT: float = 0.10

# Depth-primary terminal shaping is separate from TERMINAL_BONUS_SCALE so
# non-parametric families keep the earlier constant-stop-bonus behaviour.
DEPTH_PRIMARY_TERMINAL_BONUS_SCALE: float = 0.05

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
# Sizing rationale (worst case: 10-qubit Clifford-SU4-SU8, n_blocks=21):
#   Each SU(8) → ~30 basis gates after synthesis; 21 blocks × 30 ≈ 630 nodes.
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
        "count_per_family": 500,
        "total_steps": 300_000,
        "n_steps": 512,
        "batch_size": 128,
        "n_epochs": 5,
        "checkpoint_freq": 10_000,
    },
}

# ---------------------------------------------------------------------------
# Curriculum learning — staged training schedule
#
# Stage 1 — Foundation: 4 non-parametric families (no random rotation angles).
#   Agent learns structural optimisation (cancellation, consolidation, ZX) on
#   fixed-gate circuits before encountering parametric noise.
#   Families: QV, Clifford-SU4, Clifford-SU4-SU8, Random Clifford.
#   Promoted when eval_mean_reward > 0.25 AND donothing_per_ep < 1.5
#   for 2 consecutive eval checkpoints.
#
# Stage 2 — Parametric exposure: introduces IQP (non-param) + QAOA/EfficientSU2 (param).
#   Parametric mix: 10% → 15% → 20% (2 blend steps of +5% each, one per passing eval).
#   Entropy decays 0.05 → 0.02 over the stage.
#   Parametric circuits yield near-zero 2q reduction — included only to
#   shape early-termination behaviour; capped at 20% to preserve reward signal.
#   Promoted when reward stabilises (|Δreward| < 0.02 for 2 consecutive evals)
#   AND eval_mean_reward ≥ 0.10 (safety floor: agent not broken).
#   Note: stage 2 uses reward_delta not donothing_per_ep because DoNothing
#   rate should INCREASE in stage 2 (agent learns to exit quickly on
#   un-reducible parametric circuits) — gating on it would be wrong.
#
# Stage 3 — Balanced fine-tune: all 8 families, equal weight within groups.
#   Non-param (5 families): 17% each = 85% total.
#   Param (3 families, incl. RealAmplitudes introduced here): 5% each = 15% total.
#   Low LR (1e-4), tight clip range (0.1), early-exit penalty for DoNothing
#   before MIN_STEPS_BEFORE_STOP to prevent lazy termination.
#
# Used only when train.py is invoked with --curriculum.
# ---------------------------------------------------------------------------
_ALL_FAMILIES = [
    "qv",
    "clifford_su4_su8",
    "clifford_su4",
    "iqp",
    "random_clifford",
    "qaoa",
    "efficient_su2",
    "real_amplitudes",
]

# Stage 3 weights: equal within groups — non-param 17% each (85% total),
# param 5% each (15% total).  Continuous with stage 2 which ends at 20% param.
_STAGE3_WEIGHTS = {
    "qv": 0.20,
    "clifford_su4_su8": 0.20,
    "clifford_su4": 0.20,
    "iqp": 0.05,
    "random_clifford": 0.20,
    "qaoa": 0.05,
    "efficient_su2": 0.05,
    "real_amplitudes": 0.05,
}

CURRICULUM_STAGES: dict[int, dict] = {
    1: {
        # 4 non-parametric families only.
        # Agent learns structural optimisation (cancellation, consolidation, ZX)
        # before encountering parametric noise or IQP topology.
        "families": [
            "qv",
            "clifford_su4",
            "clifford_su4_su8",
            "random_clifford",
        ],
        "family_weights": {
            "qv": 0.30,
            "clifford_su4": 0.25,
            "clifford_su4_su8": 0.25,
            "random_clifford": 0.20,
        },
        "max_steps": 200_000,
        "ent_coef": 0.05,
        "promotion": {
            "eval_mean_reward_min": 0.25,
            "donothing_per_ep_max": 1.5,
            "consecutive_evals": 2,
        },
    },
    2: {
        # Introduces IQP (non-param, new topology) + QAOA/EfficientSU2 (param).
        # Parametric mix: 10% → 15% → 20% over 2 blend steps (+5% per passing eval).
        # Parametric circuits show near-zero 2q reduction — included only to shape
        # early-termination behaviour; capped at 20% to avoid diluting training signal.
        "families": [
            "qv",
            "clifford_su4",
            "clifford_su4_su8",
            "iqp",
            "random_clifford",
            "qaoa",
            "efficient_su2",
        ],
        "nonparam_families": [
            "qv",
            "clifford_su4",
            "clifford_su4_su8",
            "random_clifford",
        ],
        "param_families": ["iqp", "qaoa", "efficient_su2"],
        "param_start_mix": 0.10,
        "param_blend_step": 0.05,
        "param_max_mix": 0.20,
        "max_steps": 200_000,
        "ent_coef_start": 0.04,
        "ent_coef_end": 0.03,
        "step_penalty": 0.003,
        "promotion": {
            "eval_mean_reward_min": 0.10,  # safety floor only — ensures agent isn't broken;
            #   not a quality bar. 0.10 fires only if non-param performance has degraded.
            #   Primary criterion is reward_delta_max below.
            "reward_delta_max": 0.02,  # convergence gate: |reward[t] - reward[t-1]| < 0.02
            #   for 2 consecutive 10K-step evals (< 10% relative change at reward ~0.18).
            #   Detects learning plateau on the current parametric blend — ready for stage 3.
            "consecutive_evals": 2,
        },
    },
    3: {
        "families": _ALL_FAMILIES,
        "family_weights": _STAGE3_WEIGHTS,
        "max_steps": 100_000,  # balanced fine-tune  — total 500K
        "lr": 1e-4,
        "ent_coef": 0.02,
        "clip_range": 0.1,
        "donothing_early_penalty": 0.05,  # penalise DoNothing before MIN_STEPS_BEFORE_STOP
    },
}
