# =============================================================================
# quetsal/src/environment/pyzx_pass.py
# Qiskit TransformationPass wrapping pyzx.simplify.full_reduce().
#
# ZX-calculus rewriting provides a qualitatively different class of
# optimization from the peephole passes (Optimize1q, CommutativeInverse,
# ConsolidateAndSynthesize).  full_reduce combines:
#   - Spider fusion (Clifford simplification)
#   - Phase gadget reduction
#   - Graph-like diagram simplification
#
# Pipeline per call:
#   1. BasisTranslator: Heron r2 → PyZX-safe basis (cx, cz, rz, rx, x, h…)
#   2. dag → QuantumCircuit → QASM2 string → pyzx.Circuit
#   3. pyzx.simplify.full_reduce()  +  g.normalize()
#   4. pyzx.extract_circuit() → to_basic_gates() → QASM2 → Qiskit DAG
#   5. Return new DAG.  env._run_basis_cleanup() handles retranslation to
#      Heron r2 (h, cx, cz → target basis) exactly as it does for every
#      other pass — no special handling needed.
#
# Safety: any exception (unsupported gate, extraction failure, QASM parse
# error) returns the *original* dag unchanged.  The episode continues; the
# agent receives zero reward for that action on that circuit, and learns
# not to apply ZXFullReduce where it doesn't help.
# =============================================================================

from __future__ import annotations

__all__ = ["PyzxFullReduce", "PYZX_AVAILABLE"]

import warnings

from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary as sel
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.qasm2 import dumps as _qasm2_dumps, loads as _qasm2_loads
from qiskit.transpiler import TransformationPass
from qiskit.transpiler.passes import BasisTranslator

# ---------------------------------------------------------------------------
# Resolve pyzx ONCE at module import time, not per-call.
#
# A missing/broken pyzx install must surface here — loudly and only once —
# rather than being silently swallowed on every ZXFullReduce action (which
# would make the action a permanent no-op the agent learns to avoid, with no
# log line to reveal why).  When the import fails, PYZX_AVAILABLE is False and
# _PYZX_IMPORT_ERROR carries the original exception for the startup warning.
# ---------------------------------------------------------------------------
try:
    import pyzx as _zx

    PYZX_AVAILABLE: bool = True
    _PYZX_IMPORT_ERROR: Exception | None = None
except Exception as _exc:  # ImportError, or a broken transitive dependency
    _zx = None
    PYZX_AVAILABLE = False
    _PYZX_IMPORT_ERROR = _exc

# Gate set that pyzx 0.10's QASM parser reliably handles.
# Notably absent: rzz (decomposes to cx+rz), sx (→ rx(π/2)).
# BasisTranslator expands both using Qiskit's SessionEquivalenceLibrary.
_PYZX_INPUT_BASIS: list[str] = [
    "cx",
    "cz",
    "rz",
    "rx",
    "x",
    "h",
    "s",
    "sdg",
    "t",
    "tdg",
]

# Skip ZX optimization on very large DAGs — full_reduce is O(n²) and slows
# down training steps significantly above this node count.
_MAX_NODES_FOR_ZX: int = 400


class PyzxFullReduce(TransformationPass):
    """ZX-calculus circuit simplification via pyzx.simplify.full_reduce().

    Most effective on Clifford-heavy and Clifford+T circuits; still applies
    phase gadget fusion and spider fusion on rotation-heavy circuits.

    Returns the original DAG unchanged on any *runtime* failure (unsupported
    gate, extraction failure, etc.) — guaranteed not to raise from run().
    A *missing* pyzx install is a different class of problem: it is reported
    once at construction (see __init__) instead of being silently absorbed
    on every action.

    Diagnostics: self.call_count / self.fallback_count expose how often this
    pass actually transformed the DAG vs. returned it unchanged.  A 100%
    fallback rate means ZX never did anything — surface it in training logs.
    """

    # Emit the "pyzx unavailable" warning only once per process, not per
    # PyzxFullReduce instance (one is built per env / per vectorised worker).
    _missing_warned: bool = False

    def __init__(self) -> None:
        super().__init__()
        self._pre_translator = BasisTranslator(sel, _PYZX_INPUT_BASIS)
        # Per-instance diagnostics — read by training callbacks if desired.
        self.call_count: int = 0
        self.fallback_count: int = 0

        if not PYZX_AVAILABLE and not PyzxFullReduce._missing_warned:
            PyzxFullReduce._missing_warned = True
            warnings.warn(
                "pyzx is not importable "
                f"({type(_PYZX_IMPORT_ERROR).__name__}: {_PYZX_IMPORT_ERROR}); "
                "the ZXFullReduce action is a permanent no-op and the agent "
                "will never learn to use it. Install it with `pip install pyzx` "
                "(declared in requirements.txt / pyproject.toml).",
                RuntimeWarning,
                stacklevel=2,
            )

    def run(self, dag):
        self.call_count += 1

        # Missing library: nothing we can do — return unchanged.  The startup
        # warning in __init__ already flagged this; don't spam per call.
        if not PYZX_AVAILABLE:
            self.fallback_count += 1
            return dag

        try:
            # Skip oversized DAGs — full_reduce is slow above ~400 nodes
            if dag.size() > _MAX_NODES_FOR_ZX:
                self.fallback_count += 1
                return dag

            # 1. Translate to PyZX-compatible basis (expands rzz, sx, etc.)
            dag_pre = self._pre_translator.run(dag)
            qc_pre = dag_to_circuit(dag_pre)
            qasm_str = _qasm2_dumps(qc_pre)

            # 2. QASM → PyZX circuit
            pzx_circ = _zx.qasm(qasm_str)
            n_qubits_in = pzx_circ.qubits

            # 3. ZX-calculus simplification
            g = pzx_circ.to_graph()
            _zx.simplify.full_reduce(g, quiet=True)
            g.normalize()

            # 4. Extract optimized circuit
            opt = _zx.extract_circuit(g.copy()).to_basic_gates()

            # Guard: extraction must preserve qubit count
            if opt.qubits != n_qubits_in:
                self.fallback_count += 1
                return dag

            # 5. PyZX circuit → QASM → Qiskit DAG
            opt_qasm = opt.to_qasm()
            qc_opt = _qasm2_loads(opt_qasm)
            return circuit_to_dag(qc_opt)

        except Exception:
            # Genuine per-circuit failure (unsupported gate, extraction/QASM
            # error). Fall back to the unchanged DAG — never crash an episode.
            self.fallback_count += 1
            return dag
