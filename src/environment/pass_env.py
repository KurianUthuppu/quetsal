# =============================================================================
# quetsal/src/environment/pass_env.py
# Gymnasium environment for RL-based Qiskit transpiler pass sequencing.
#
# The agent selects optimization passes one at a time.  After each pass,
# the env auto-runs GatesInBasis + conditional BasisTranslator to ensure
# the DAG stays in the Heron r2 basis — mirroring Qiskit's builtin_plugins.py
# optimization loop (lines 593-611).
# =============================================================================

from __future__ import annotations

__all__ = ["PassManagerEnv"]

from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary as sel
from qiskit.transpiler.passes import BasisTranslator
from qiskit.transpiler.passes.utils.gates_basis import GatesInBasis
from qiskit.transpiler.passes.optimization import (
    Optimize1qGatesDecomposition,
    CommutativeInverseCancellation,
    ConsolidateBlocks,
    OptimizeCliffords,
    Split2QUnitaries,
    # RemoveIdentityEquivalent,  # disabled — not in current action space
    ContractIdleWiresInControlFlow,
)
from qiskit.transpiler.passes.synthesis.unitary_synthesis import UnitarySynthesis

from quetsal.src.environment.pyzx_pass import PyzxFullReduce
from quetsal.src.constants import (
    ACTION_LABELS,
    DEPTH_PENALTY_WEIGHT,
    STEP_PENALTY,
    TERMINAL_BONUS,
    TRUNCATION_PENALTY,
    EDGE_DIM,
    HERON_R2_BASIS,
    MAX_EDGES,
    MAX_NODES,
    MAX_STEPS_PER_EPISODE,
    MIN_STEPS_BEFORE_STOP,
    NODE_DIM,
    NUM_ACTIONS,
    SKIP_GATES,
)
from quetsal.src.encoder.dag_encoder import dag_to_pyg


# ── Helpers ───────────────────────────────────────────────────────────────────


class _DagOverflowError(RuntimeError):
    """Raised when a DAG exceeds MAX_NODES or MAX_EDGES mid-episode."""


def _count_2q(dag: DAGCircuit) -> int:
    """Count 2-qubit gates in a DAGCircuit (excluding SKIP_GATES)."""
    return sum(
        1
        for node in dag.topological_op_nodes()
        if node.op.name not in SKIP_GATES and len(node.qargs) >= 2
    )


# ── Environment ──────────────────────────────────────────────────────────────


class PassManagerEnv(gym.Env):
    """
    Gymnasium environment for Qiskit transpiler pass sequencing.

    Observation : dict with PyG-compatible tensors (x, edge_index, edge_attr)
    Action      : Discrete(NUM_ACTIONS) — index into ACTION_LABELS
    Reward      : normalized 2q gate reduction per step
    Termination : DoNothing action or MAX_STEPS_PER_EPISODE reached

    After each agent action, the env automatically runs:
      1. GatesInBasis check
      2. Conditional BasisTranslator (if non-basis gates were reintroduced)
      3. ContractIdleWiresInControlFlow (wire cleanup)
    This mirrors Qiskit's builtin_plugins.py optimization loop.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        circuits: list,
        max_steps: int = MAX_STEPS_PER_EPISODE,
        basis_gates: list[str] | None = None,
        donothing_early_penalty: float = 0.0,
    ):
        """
        Parameters
        ----------
        circuits    : list of Qiskit QuantumCircuit objects, already transpiled
                      through layout + routing + translation (i.e., ready for
                      the optimization stage).  The env samples one per episode.
        max_steps   : maximum optimization passes per episode before truncation.
        basis_gates             : target basis gates.  Defaults to HERON_R2_BASIS.
        donothing_early_penalty : stage 3 curriculum penalty applied when the agent
                                  selects DoNothing before MIN_STEPS_BEFORE_STOP.
                                  0.0 (default) disables the penalty.
        """
        super().__init__()

        self.circuits = circuits
        self.max_steps = max_steps
        self.basis_gates = basis_gates or HERON_R2_BASIS
        self.donothing_early_penalty = donothing_early_penalty

        # -- Action space: 7 discrete actions (6 passes + DoNothing) -----------
        self.action_space = spaces.Discrete(NUM_ACTIONS)

        # -- Observation space: padded fixed-size tensors ----------------------
        # SB3 pre-allocates numpy buffers of shape (n_steps, *space.shape).
        # Variable-size graphs would fail this allocation, so we pad to
        # MAX_NODES / MAX_EDGES and include boolean masks.  The GNN policy
        # uses node_mask / edge_mask to strip padding before message passing.
        self.observation_space = spaces.Dict(
            {
                "x": spaces.Box(
                    -np.inf, np.inf, shape=(MAX_NODES, NODE_DIM), dtype=np.float32
                ),
                "edge_index": spaces.Box(
                    0, MAX_NODES - 1, shape=(2, MAX_EDGES), dtype=np.int64
                ),
                "edge_attr": spaces.Box(
                    -np.inf, np.inf, shape=(MAX_EDGES, EDGE_DIM), dtype=np.float32
                ),
                "node_mask": spaces.Box(0, 1, shape=(MAX_NODES,), dtype=np.bool_),
                "edge_mask": spaces.Box(0, 1, shape=(MAX_EDGES,), dtype=np.bool_),
            }
        )

        # -- Build optimization passes (instantiated once, reused) -------------
        self._passes = self._build_passes()

        # -- Infrastructure passes (auto-run after every step) -----------------
        self._gates_in_basis = GatesInBasis(self.basis_gates)
        self._unitary_synthesis = UnitarySynthesis(
            self.basis_gates
        )  # handles unitary blocks from ConsolidateBlocks
        self._basis_translator = BasisTranslator(sel, self.basis_gates)
        self._contract_idle = ContractIdleWiresInControlFlow()

        # -- Episode state (set in reset) --------------------------------------
        self._dag: Optional[DAGCircuit] = None
        self._initial_2q: int = 0
        self._prev_2q: int = 0
        self._initial_depth: int = 0
        self._prev_depth: int = 0
        self._step_count: int = 0
        self._last_pass_map: dict[int, int] = {}
        self._current_family: str = "unknown"
        self._rng = np.random.default_rng()

    def _build_passes(self) -> list:
        """Instantiate the optimization passes (indices 0 to NUM_ACTIONS-2).

        Action 2 is the ConsolidateAndSynthesize macro — stored as a 2-tuple
        (ConsolidateBlocks, UnitarySynthesis) and run sequentially in step().
        Action 5 is ZXFullReduce (PyzxFullReduce) — falls back to unchanged
        DAG on any failure so it can never crash an episode.
        The last action (DoNothing, index 6) is handled as a special case in step().
        """
        return [
            Optimize1qGatesDecomposition(basis=self.basis_gates),  # 0
            CommutativeInverseCancellation(),                       # 1
            (
                ConsolidateBlocks(basis_gates=self.basis_gates),    # 2 macro
                UnitarySynthesis(self.basis_gates),
            ),
            OptimizeCliffords(),                                    # 3
            Split2QUnitaries(),                                     # 4
            PyzxFullReduce(),                                       # 5
            # DoNothing is action 6, handled as special case in step()
        ]

    def _run_basis_cleanup(self) -> None:
        """Auto-run GatesInBasis + conditional synthesis/translation + wire cleanup.

        ConsolidateBlocks produces opaque UnitaryGate blocks which BasisTranslator
        cannot handle (no equivalence rules for unitaries).  We run UnitarySynthesis
        first to decompose any unitary blocks, then BasisTranslator for any remaining
        non-basis gates from other passes.
        """
        self._gates_in_basis.run(self._dag)
        if not self._gates_in_basis.property_set.get("all_gates_in_basis", True):
            # UnitarySynthesis first: decomposes unitary blocks → basis gates
            self._dag = self._unitary_synthesis.run(self._dag)
            # BasisTranslator second: handles any remaining non-basis gates
            self._gates_in_basis.run(self._dag)
            if not self._gates_in_basis.property_set.get("all_gates_in_basis", True):
                self._dag = self._basis_translator.run(self._dag)

        # Wire cleanup
        self._dag = self._contract_idle.run(self._dag)

    def _encode_observation(self) -> dict[str, np.ndarray]:
        """Encode the current DAG as padded fixed-size numpy arrays.

        Real nodes/edges are written into the first N/E rows; the remainder
        is zero-padded.  node_mask and edge_mask are True for real entries.
        The GNN policy uses these masks to strip padding before message passing.
        """
        data = dag_to_pyg(
            self._dag,
            last_pass_map=self._last_pass_map,
            num_passes=NUM_ACTIONS,
        )

        n = data.x.shape[0]  # real node count
        e = data.edge_index.shape[1]  # real edge count

        if n > MAX_NODES or e > MAX_EDGES:
            raise _DagOverflowError(
                f"DAG too large for buffer: {n} nodes (max {MAX_NODES}), "
                f"{e} edges (max {MAX_EDGES})"
            )

        # Node features
        x_pad = np.zeros((MAX_NODES, NODE_DIM), dtype=np.float32)
        x_pad[:n] = data.x.numpy()

        # Edge index
        ei_pad = np.zeros((2, MAX_EDGES), dtype=np.int64)
        ei_pad[:, :e] = data.edge_index.numpy()

        # Edge attributes
        ea_pad = np.zeros((MAX_EDGES, EDGE_DIM), dtype=np.float32)
        ea_pad[:e] = data.edge_attr.numpy()

        # Masks — True = real, False = padding
        node_mask = np.zeros(MAX_NODES, dtype=np.bool_)
        node_mask[:n] = True
        edge_mask = np.zeros(MAX_EDGES, dtype=np.bool_)
        edge_mask[:e] = True

        return {
            "x": x_pad,
            "edge_index": ei_pad,
            "edge_attr": ea_pad,
            "node_mask": node_mask,
            "edge_mask": edge_mask,
        }

    def _zero_obs(self) -> dict[str, np.ndarray]:
        """Return an all-zeros observation (used when DAG overflows the buffer)."""
        return {
            "x": np.zeros((MAX_NODES, NODE_DIM), dtype=np.float32),
            "edge_index": np.zeros((2, MAX_EDGES), dtype=np.int64),
            "edge_attr": np.zeros((MAX_EDGES, EDGE_DIM), dtype=np.float32),
            "node_mask": np.zeros(MAX_NODES, dtype=np.bool_),
            "edge_mask": np.zeros(MAX_EDGES, dtype=np.bool_),
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """
        Reset the environment: sample a circuit, convert to DAG, encode.

        Returns (observation, info).
        """
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Sample a circuit from the pool
        idx = self._rng.integers(0, len(self.circuits))
        qc = self.circuits[idx]

        # Track circuit family for per-family pass logging
        self._current_family = (qc.metadata or {}).get("family", "unknown")

        # Convert to DAG
        self._dag = circuit_to_dag(qc)

        # Record initial 2q count and depth (denominators for reward normalisation)
        self._initial_2q = _count_2q(self._dag)
        self._prev_2q = self._initial_2q
        self._initial_depth = self._dag.depth()
        self._prev_depth = self._initial_depth
        self._step_count = 0
        self._last_pass_map = {}

        # Sanity-check dependent variables: circuits must have been filtered
        # by _transpile_to_opt_stage before reaching the env, so a zero-2q
        # circuit or a mismatched prev/initial is always a data pipeline bug.
        assert self._initial_2q > 0, (
            f"Circuit {idx} has no 2q gates after transpilation — "
            "should have been filtered by generate_training_circuits()"
        )
        assert (
            self._prev_2q == self._initial_2q
        ), f"prev_2q ({self._prev_2q}) != initial_2q ({self._initial_2q}) at reset"
        assert self._step_count == 0, "step_count not zeroed at reset"
        assert (
            self._initial_depth > 0
        ), f"Circuit {idx} has depth 0 at reset — unexpected empty DAG"

        obs = self._encode_observation()
        info = {
            "circuit_index": int(idx),
            "initial_2q": self._initial_2q,
            "initial_depth": self._dag.depth(),
        }
        return obs, info

    def step(
        self, action: int
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """
        Apply one optimization pass to the DAG.

        Parameters
        ----------
        action : int in [0, NUM_ACTIONS-1]
            0-5 = optimization pass, 6 = DoNothing (terminate).

        Returns
        -------
        observation : dict of numpy arrays (PyG-compatible)
        reward      : float — normalized 2q gate reduction this step
        terminated  : bool — True if DoNothing was selected
        truncated   : bool — True if max_steps reached
        info        : dict with diagnostic metrics
        """
        assert self.action_space.contains(action), f"Invalid action: {action}"

        self._step_count += 1
        terminated = False
        truncated = False

        # -- DoNothing → terminate episode -------------------------------------
        # Ignore DoNothing for the first MIN_STEPS_BEFORE_STOP steps so the
        # agent is forced to apply at least one real pass before it can stop.
        # This prevents the untrained policy from collapsing to ep_len=1.
        if action == NUM_ACTIONS - 1 and self._step_count >= MIN_STEPS_BEFORE_STOP:
            terminated = True
            # Terminal bonus: fixed reward for choosing to stop at the right time.
            # A cumulative-reduction bonus double-counts step rewards already
            # received and teaches the agent to do one pass then bail early.
            # A small fixed constant rewards timely termination without that bias.
            reward = TERMINAL_BONUS
            try:
                obs = self._encode_observation()
            except _DagOverflowError:
                obs = self._zero_obs()
            info = self._build_info(action, reward)
            return obs, reward, terminated, truncated, info

        # -- Apply the selected optimization pass ------------------------------
        # If DoNothing was suppressed (too early), treat it as a no-op step:
        # skip pass application and apply optional early-exit penalty (stage 3).
        if action == NUM_ACTIONS - 1:
            reward = -self.donothing_early_penalty  # stored positive, negated here (0.0 unless stage 3)
            self._prev_2q = _count_2q(self._dag)
            if self._step_count >= self.max_steps:
                truncated = True
            try:
                obs = self._encode_observation()
            except _DagOverflowError:
                obs = self._zero_obs()
                truncated = True
            info = self._build_info(action, reward)
            return obs, reward, terminated, truncated, info

        opt_pass = self._passes[action]
        if isinstance(opt_pass, tuple):
            # Macro-action: run each pass in sequence (ConsolidateBlocks → UnitarySynthesis)
            for p in opt_pass:
                self._dag = p.run(self._dag)
        else:
            self._dag = opt_pass.run(self._dag)

        # -- Auto-run basis cleanup (GatesInBasis + BasisTranslator) -----------
        self._run_basis_cleanup()

        # -- Update last_pass_map for all current nodes ------------------------
        # After DAG mutation, old node ids are invalid.  We mark ALL current
        # nodes with the pass that just ran, since we can't track which nodes
        # were specifically modified (Qiskit passes don't report this).
        self._last_pass_map = {
            id(node): action
            for node in self._dag.topological_op_nodes()
            if node.op.name not in SKIP_GATES
        }

        # -- Compute reward ----------------------------------------------------
        current_2q = _count_2q(self._dag)

        if self._initial_2q > 0:
            # Normalised step reward: 2q gates removed this step / initial count
            step_reduction = (self._prev_2q - current_2q) / self._initial_2q
        else:
            step_reduction = 0.0

        # Optional depth penalty: per-step depth change, normalized by initial depth.
        # Mirrors step_reduction: (prev - current) / initial_2q for 2q gates.
        # Positive depth_change = depth grew this step = penalty.
        depth_penalty = 0.0
        current_depth = self._dag.depth()
        if DEPTH_PENALTY_WEIGHT > 0.0 and self._initial_depth > 0:
            depth_change = (current_depth - self._prev_depth) / self._initial_depth
            depth_penalty = -DEPTH_PENALTY_WEIGHT * depth_change

        # Step penalty: small fixed cost per pass applied.
        # Incentivises the agent to stop unless the pass genuinely reduces 2q gates.
        reward = step_reduction + depth_penalty - STEP_PENALTY
        self._prev_2q = current_2q
        self._prev_depth = current_depth

        # -- Check truncation --------------------------------------------------
        if self._step_count >= self.max_steps:
            truncated = True
            # Truncation penalty: agent failed to call DoNothing within MAX_STEPS.
            # Penalises running all steps on already-optimal circuits (wasted compute).
            reward -= TRUNCATION_PENALTY

        # -- Encode observation ------------------------------------------------
        try:
            obs = self._encode_observation()
        except _DagOverflowError:
            # DAG inflated beyond buffer mid-episode (e.g. ConsolidateBlocks
            # intermediate expansion). Truncate cleanly with a zero observation.
            obs = self._zero_obs()
            truncated = True

        info = self._build_info(action, reward)
        return obs, reward, terminated, truncated, info

    def _build_info(self, action: int, reward: float) -> dict[str, Any]:
        """Build the info dict returned by step()."""
        current_2q = _count_2q(self._dag)
        return {
            "action_name": ACTION_LABELS[action],
            "family": self._current_family,
            "step": self._step_count,
            "current_2q": current_2q,
            "initial_2q": self._initial_2q,
            "total_reduction": (
                (self._initial_2q - current_2q) / self._initial_2q
                if self._initial_2q > 0
                else 0.0
            ),
            "depth": self._dag.depth(),
            "initial_depth": self._initial_depth,
            "total_depth_reduction": (
                (self._initial_depth - self._dag.depth()) / self._initial_depth
                if self._initial_depth > 0
                else 0.0
            ),
            "reward": reward,
        }
