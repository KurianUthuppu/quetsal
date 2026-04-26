# =============================================================================
# quetsal/src/environment/circuits.py
# Training circuit generators for the Gymnasium environment.
#
# Each generator produces QuantumCircuits transpiled through
# init → layout → routing → translation (but NOT optimization).
# The RL agent IS the optimization stage — these circuits are its input.
#
# Circuit families (7 total):
#   - Quantum Volume (QV): layers of random SU(4) on shuffled qubit pairs
#   - QAOA MaxCut: via QAOAAnsatz on random graphs
#   - Clifford-SU4-SU8: mix of random Cliffords + SU(4) + SU(8) gates
#   - Clifford-SU4: mixed random Cliffords + random SU(4) gates
#   - IQP: commuting gate circuits via Qiskit's random_iqp
#   - EfficientSU2: hardware-efficient VQE ansatz (RY+RZ + CX layers)
#   - RealAmplitudes: real-valued VQE ansatz (RY + CX layers)
#
# Parametric circuits (QAOA, EfficientSU2, RealAmplitudes) are bound with
# random parameter values — the transpiler needs concrete angles to optimize.
#
# Coupling map: line topology (sufficient for 3-10 qubit training circuits).
# A real Heron r2 heavy-hex map would be used for final benchmarking.
# =============================================================================

from __future__ import annotations

__all__ = [
    "generate_qv_circuits",
    "generate_qaoa_circuits",
    "generate_clifford_su4_su8_circuits",
    "generate_clifford_su4_circuits",
    "generate_iqp_circuits",
    "generate_efficient_su2_circuits",
    "generate_real_amplitudes_circuits",
    "generate_pauli_gadget_circuits",
    "generate_random_clifford_circuits",
    "generate_training_circuits",
    "generate_weighted_circuits",
]

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import (
    QuantumVolume,
    UnitaryGate,
    efficient_su2,
    qaoa_ansatz,
    random_iqp,
    real_amplitudes,
)
from qiskit.quantum_info import SparsePauliOp, random_clifford, random_unitary
from qiskit.transpiler import CouplingMap, PassManager
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from quetsal.src.constants import HERON_R2_BASIS, MAX_NODES, SKIP_GATES


# ── Helpers ──────────────────────────────────────────────────────────────────


def _circuit_node_count(qc: QuantumCircuit) -> int:
    """Count optimisable op nodes in a circuit (excluding SKIP_GATES)."""
    return sum(1 for inst in qc.data if inst.operation.name not in SKIP_GATES)


def _transpile_to_opt_stage(
    circuits: list[QuantumCircuit],
    basis_gates: list[str],
    seed: int,
    family: str = "unknown",
) -> list[QuantumCircuit]:
    """Transpile circuits through init+layout+routing+translation only.

    Uses optimization_level=1 for good layout+routing but replaces the
    optimization and scheduling stages with empty PassManagers
    so the RL agent receives un-optimized circuits.

    Circuits that exceed MAX_NODES after transpilation are silently dropped —
    they would overflow the padded observation buffer.
    """
    results = []
    for qc in circuits:
        n = qc.num_qubits
        cm = CouplingMap.from_line(n)
        pm = generate_preset_pass_manager(
            optimization_level=1,
            basis_gates=basis_gates,
            coupling_map=cm,
            seed_transpiler=seed,
        )
        # Skip optimization and scheduling — the RL agent IS the optimizer
        pm.optimization = PassManager()
        pm.scheduling = PassManager()
        transpiled = pm.run(qc)
        # Drop circuits that would overflow the padded observation buffer.
        # Use 60% of MAX_NODES as the threshold: ConsolidateBlocks + UnitarySynthesis
        # in basis cleanup can roughly double a DAG's node count mid-episode before
        # re-synthesising it smaller, so we need ~2x headroom.
        two_q = sum(
            1
            for inst in transpiled.data
            if len(inst.qubits) >= 2 and inst.operation.name not in SKIP_GATES
        )
        if two_q == 0:
            continue  # no 2q gates → reward always 0, skip
        if _circuit_node_count(transpiled) <= int(MAX_NODES * 0.6):
            # Tag with family so the env and logger can report per-family stats
            if transpiled.metadata is None:
                transpiled.metadata = {}
            transpiled.metadata["family"] = family
            results.append(transpiled)
    return results


def _bind_random_params(qc: QuantumCircuit, rng: np.random.Generator) -> QuantumCircuit:
    """Bind random values to all parameters in a parametric circuit."""
    if not qc.parameters:
        return qc
    params = {p: float(rng.uniform(0, 2 * np.pi)) for p in qc.parameters}
    return qc.assign_parameters(params)


# ── Quantum Volume ───────────────────────────────────────────────────────────


def generate_qv_circuits(
    n_qubits_range: tuple[int, int] = (3, 10),
    count: int = 100,
    seed: int = 42,
    basis_gates: list[str] | None = None,
) -> list[QuantumCircuit]:
    """Generate Quantum Volume circuits transpiled to target basis.

    QV circuits consist of layers of random SU(4) gates on shuffled qubit
    pairs — a standard benchmark for quantum hardware and compilers.

    Parameters
    ----------
    n_qubits_range : (min, max) inclusive range for random qubit count.
    count          : number of circuits to generate.
    seed           : RNG seed for reproducibility.
    basis_gates    : target basis gates (default: HERON_R2_BASIS).
    """
    basis = basis_gates or HERON_R2_BASIS
    rng = np.random.default_rng(seed)

    raw = []
    for _ in range(count):
        n = int(rng.integers(n_qubits_range[0], n_qubits_range[1] + 1))
        qc = QuantumVolume(n, seed=int(rng.integers(0, 2**31)))
        qc = qc.decompose()
        raw.append(qc)

    return _transpile_to_opt_stage(raw, basis, seed, family="qv")


# ── QAOA MaxCut ──────────────────────────────────────────────────────────────


def _maxcut_cost_op(n: int, edges: list[tuple[int, int]]) -> SparsePauliOp:
    """Build MaxCut cost operator: sum_{(i,j)} Z_i Z_j."""
    terms = []
    for i, j in edges:
        z_str = ["I"] * n
        z_str[i] = "Z"
        z_str[j] = "Z"
        terms.append(("".join(z_str), 1.0))
    return SparsePauliOp.from_list(terms)


def generate_qaoa_circuits(
    n_qubits_range: tuple[int, int] = (3, 10),
    count: int = 100,
    seed: int = 42,
    p_layers: int = 2,
    basis_gates: list[str] | None = None,
) -> list[QuantumCircuit]:
    """Generate QAOA MaxCut circuits transpiled to target basis.

    Uses Qiskit's qaoa_ansatz with random MaxCut cost operators graphs
    with edge probability as 0.5.  Parameters are bound
    to random values before transpilation.

    Parameters
    ----------
    n_qubits_range : (min, max) inclusive range for random qubit count.
    count          : number of circuits to generate.
    seed           : RNG seed for reproducibility.
    p_layers       : number of QAOA layers (cost + mixer repetitions).
    basis_gates    : target basis gates (default: HERON_R2_BASIS).
    """
    basis = basis_gates or HERON_R2_BASIS
    rng = np.random.default_rng(seed)

    raw = []
    for _ in range(count):
        n = int(rng.integers(n_qubits_range[0], n_qubits_range[1] + 1))

        # Random Erdős–Rényi graph (edge probability 0.5)
        edges = [
            (i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < 0.5
        ]
        # Ensure at least one edge
        if not edges:
            i = int(rng.integers(0, n))
            j = int(rng.integers(0, n - 1))
            if j >= i:
                j += 1
            edges = [(min(i, j), max(i, j))]

        cost_op = _maxcut_cost_op(n, edges)
        qc = qaoa_ansatz(cost_op, reps=p_layers)
        qc = _bind_random_params(qc, rng)
        raw.append(qc)

    return _transpile_to_opt_stage(raw, basis, seed, family="qaoa")


# ── Clifford-SU4-SU8 ────────────────────────────────────────────────────────


def generate_clifford_su4_su8_circuits(
    n_qubits_range: tuple[int, int] = (3, 10),
    count: int = 100,
    seed: int = 42,
    basis_gates: list[str] | None = None,
) -> list[QuantumCircuit]:
    """Generate circuits mixing random Cliffords, SU(4), and SU(8) gates.

    Each gate block is chosen uniformly at random from three types:
      - Clifford (2q): analytically simplifiable, finite group
      - SU(4) (2q):   non-simplifiable random 2-qubit unitary
      - SU(8) (3q):   non-simplifiable random 3-qubit unitary → dense 2q output

    Parameters
    ----------
    n_qubits_range : (min, max) inclusive range for random qubit count (min 3).
    count          : number of circuits to generate.
    seed           : RNG seed for reproducibility.
    basis_gates    : target basis gates (default: HERON_R2_BASIS).
    """
    basis = basis_gates or HERON_R2_BASIS
    rng = np.random.default_rng(seed)

    raw = []
    for _ in range(count):
        # SU(8) requires at least 3 qubits — enforced by min below
        n = int(rng.integers(max(3, n_qubits_range[0]), n_qubits_range[1] + 1))
        # Fewer blocks than SU4-only: each SU(8) decomposes to many 2q gates
        n_blocks = int(rng.integers(n, 2 * n + 1))

        qc = QuantumCircuit(n)
        for _ in range(n_blocks):
            roll = rng.random()
            if roll < 1 / 3:
                # Random 2-qubit Clifford
                q1, q2 = rng.choice(n, 2, replace=False)
                cliff_qc = random_clifford(
                    2, seed=int(rng.integers(0, 2**31))
                ).to_circuit()
                qc.compose(cliff_qc, [int(q1), int(q2)], inplace=True)
            elif roll < 2 / 3:
                # Random SU(4) unitary
                q1, q2 = rng.choice(n, 2, replace=False)
                U = random_unitary(4, seed=int(rng.integers(0, 2**31)))
                qc.append(UnitaryGate(U), [int(q1), int(q2)])
            else:
                # Random SU(8) unitary (3-qubit)
                qs = rng.choice(n, 3, replace=False)
                U = random_unitary(8, seed=int(rng.integers(0, 2**31)))
                qc.append(UnitaryGate(U), [int(qs[0]), int(qs[1]), int(qs[2])])

        raw.append(qc)

    return _transpile_to_opt_stage(raw, basis, seed, family="clifford_su4_su8")


# ── Clifford-SU4 ────────────────────────────────────────────────────────────


def generate_clifford_su4_circuits(
    n_qubits_range: tuple[int, int] = (3, 10),
    count: int = 100,
    seed: int = 42,
    clifford_fraction: float = 0.5,
    basis_gates: list[str] | None = None,
) -> list[QuantumCircuit]:
    """Generate circuits mixing random Clifford subcircuits and random SU(4) gates.

    Parameters
    ----------
    n_qubits_range    : (min, max) inclusive range for random qubit count.
    count             : number of circuits to generate.
    seed              : RNG seed for reproducibility.
    clifford_fraction : probability that each gate is a Clifford (vs SU4).
    basis_gates       : target basis gates (default: HERON_R2_BASIS).
    """
    basis = basis_gates or HERON_R2_BASIS
    rng = np.random.default_rng(seed)

    raw = []
    for _ in range(count):
        n = int(rng.integers(n_qubits_range[0], n_qubits_range[1] + 1))
        n_gates = int(rng.integers(n, 3 * n + 1))

        qc = QuantumCircuit(n)
        for _ in range(n_gates):
            q1, q2 = rng.choice(n, 2, replace=False)
            if rng.random() < clifford_fraction:
                # Random 2-qubit Clifford → circuit → compose
                cliff_qc = random_clifford(
                    2, seed=int(rng.integers(0, 2**31))
                ).to_circuit()
                qc.compose(cliff_qc, [int(q1), int(q2)], inplace=True)
            else:
                # Random SU(4) unitary
                U = random_unitary(4, seed=int(rng.integers(0, 2**31)))
                qc.append(UnitaryGate(U), [int(q1), int(q2)])

        raw.append(qc)

    return _transpile_to_opt_stage(raw, basis, seed, family="clifford_su4")


# ── IQP (Instantaneous Quantum Polynomial) ──────────────────────────────────


def generate_iqp_circuits(
    n_qubits_range: tuple[int, int] = (3, 10),
    count: int = 100,
    seed: int = 42,
    basis_gates: list[str] | None = None,
) -> list[QuantumCircuit]:
    """Generate IQP circuits transpiled to target basis.

    Uses Qiskit's random_iqp generator.  IQP circuits have the structure
    H^n → Diagonal(T, CS gates) → H^n.

    Parameters
    ----------
    n_qubits_range : (min, max) inclusive range for random qubit count.
    count          : number of circuits to generate.
    seed           : RNG seed for reproducibility.
    basis_gates    : target basis gates (default: HERON_R2_BASIS).
    """
    basis = basis_gates or HERON_R2_BASIS
    rng = np.random.default_rng(seed)

    raw = []
    for _ in range(count):
        n = int(rng.integers(n_qubits_range[0], n_qubits_range[1] + 1))
        qc = random_iqp(n, seed=int(rng.integers(0, 2**31)))
        raw.append(qc)

    return _transpile_to_opt_stage(raw, basis, seed, family="iqp")


# ── EfficientSU2 (hardware-efficient ansatz) ─────────────────────────────────


def generate_efficient_su2_circuits(
    n_qubits_range: tuple[int, int] = (3, 10),
    count: int = 100,
    seed: int = 42,
    reps: int = 3,
    basis_gates: list[str] | None = None,
) -> list[QuantumCircuit]:
    """Generate EfficientSU2 ansatz circuits transpiled to target basis.

    Uses Qiskit's efficient_su2: alternating layers of RY+RZ single-qubit
    rotations and CX entanglement.  A standard hardware-efficient ansatz
    for VQE.  Parameters are bound to random values before transpilation.

    Parameters
    ----------
    n_qubits_range : (min, max) inclusive range for random qubit count.
    count          : number of circuits to generate.
    seed           : RNG seed for reproducibility.
    reps           : number of rotation+entanglement layer repetitions.
    basis_gates    : target basis gates (default: HERON_R2_BASIS).
    """
    basis = basis_gates or HERON_R2_BASIS
    rng = np.random.default_rng(seed)

    raw = []
    for _ in range(count):
        n = int(rng.integers(n_qubits_range[0], n_qubits_range[1] + 1))
        qc = efficient_su2(n, reps=reps)
        qc = _bind_random_params(qc, rng)
        raw.append(qc)

    return _transpile_to_opt_stage(raw, basis, seed, family="efficient_su2")


# ── RealAmplitudes (real-valued ansatz) ──────────────────────────────────────


def generate_real_amplitudes_circuits(
    n_qubits_range: tuple[int, int] = (3, 10),
    count: int = 100,
    seed: int = 42,
    reps: int = 3,
    basis_gates: list[str] | None = None,
) -> list[QuantumCircuit]:
    """Generate RealAmplitudes ansatz circuits transpiled to target basis.

    Uses Qiskit's real_amplitudes: alternating layers of RY rotations and
    CX entanglement.  Prepares states with only real amplitudes — commonly
    used for chemistry VQE.  Parameters bound to random values.

    Parameters
    ----------
    n_qubits_range : (min, max) inclusive range for random qubit count.
    count          : number of circuits to generate.
    seed           : RNG seed for reproducibility.
    reps           : number of rotation+entanglement layer repetitions.
    basis_gates    : target basis gates (default: HERON_R2_BASIS).
    """
    basis = basis_gates or HERON_R2_BASIS
    rng = np.random.default_rng(seed)

    raw = []
    for _ in range(count):
        n = int(rng.integers(n_qubits_range[0], n_qubits_range[1] + 1))
        qc = real_amplitudes(n, reps=reps)
        qc = _bind_random_params(qc, rng)
        raw.append(qc)

    return _transpile_to_opt_stage(raw, basis, seed, family="real_amplitudes")


# ── Pauli Gadget circuits ─────────────────────────────────────────────────────


def generate_pauli_gadget_circuits(
    n_qubits_range: tuple[int, int] = (3, 10),
    count: int = 100,
    seed: int = 42,
    n_layers_range: tuple[int, int] = (2, 6),
    basis_gates: list[str] | None = None,
) -> list[QuantumCircuit]:
    """Generate Pauli gadget circuits (Trotterized Hamiltonian structure).

    Alternating layers of random RZ rotations and
    CZ entanglement — mimics exp(iθ Z⊗Z) Trotter steps from Hamiltonian
    simulation.

    Parameters
    ----------
    n_qubits_range  : (min, max) inclusive range for random qubit count.
    count           : number of circuits to generate.
    seed            : RNG seed for reproducibility.
    n_layers_range  : (min, max) inclusive range for Trotter layer count.
    basis_gates     : target basis gates (default: HERON_R2_BASIS).
    """
    basis = basis_gates or HERON_R2_BASIS
    rng = np.random.default_rng(seed)

    raw = []
    for _ in range(count):
        n = int(rng.integers(n_qubits_range[0], n_qubits_range[1] + 1))
        n_layers = int(rng.integers(n_layers_range[0], n_layers_range[1] + 1))

        qc = QuantumCircuit(n)
        for layer in range(n_layers):
            # RZ layer: diagonal phase gadgets
            for q in range(n):
                theta = float(rng.uniform(0, 2 * np.pi))
                qc.rz(theta, q)
            # CZ entanglement layer: alternating even/odd pairs per layer
            offset = layer % 2
            for q in range(offset, n - 1, 2):
                qc.cz(q, q + 1)
        raw.append(qc)

    return _transpile_to_opt_stage(raw, basis, seed, family="pauli_gadget")


# ── Random Clifford circuits ──────────────────────────────────────────────────


def generate_random_clifford_circuits(
    n_qubits_range: tuple[int, int] = (3, 10),
    count: int = 100,
    seed: int = 42,
    basis_gates: list[str] | None = None,
) -> list[QuantumCircuit]:
    """Generate pure random Clifford circuits.

    Random n-qubit Cliffords via Qiskit's random_clifford().

    Parameters
    ----------
    n_qubits_range : (min, max) inclusive range for random qubit count.
    count          : number of circuits to generate.
    seed           : RNG seed for reproducibility.
    basis_gates    : target basis gates (default: HERON_R2_BASIS).
    """
    basis = basis_gates or HERON_R2_BASIS
    rng = np.random.default_rng(seed)

    raw = []
    for _ in range(count):
        n = int(rng.integers(n_qubits_range[0], n_qubits_range[1] + 1))
        cliff = random_clifford(n, seed=int(rng.integers(0, 2**31)))
        qc = cliff.to_circuit()
        raw.append(qc)

    return _transpile_to_opt_stage(raw, basis, seed, family="random_clifford")


# ── Convenience: mixed training pool ─────────────────────────────────────────


def generate_training_circuits(
    n_qubits_range: tuple[int, int] = (3, 10),
    count_per_family: int = 100,
    seed: int = 42,
    basis_gates: list[str] | None = None,
    families: list[str] | None = None,
) -> list[QuantumCircuit]:
    """Generate a mixed pool of training circuits from all 8 families (or a subset).

    Returns count_per_family circuits per included family, shuffled.
    Different seeds per family ensure no overlap.

    Parameters
    ----------
    families : if provided, only generate these families. Valid names:
               qv, qaoa, clifford_su4_su8, clifford_su4, iqp,
               efficient_su2, real_amplitudes, random_clifford.
               None (default) includes all 8 families.
    """
    basis = basis_gates or HERON_R2_BASIS
    _include = set(families) if families is not None else None
    _gen_seq = [
        ("qv", generate_qv_circuits, 0),
        ("qaoa", generate_qaoa_circuits, 1),
        ("clifford_su4_su8", generate_clifford_su4_su8_circuits, 2),
        ("clifford_su4", generate_clifford_su4_circuits, 3),
        ("iqp", generate_iqp_circuits, 4),
        ("efficient_su2", generate_efficient_su2_circuits, 5),
        ("real_amplitudes", generate_real_amplitudes_circuits, 6),
        ("random_clifford", generate_random_clifford_circuits, 7),
    ]
    circuits = []
    for family_name, gen_fn, offset in _gen_seq:
        if _include is None or family_name in _include:
            circuits.extend(
                gen_fn(
                    n_qubits_range, count_per_family, seed + offset, basis_gates=basis
                )
            )

    # Shuffle so the env doesn't see families in blocks
    rng = np.random.default_rng(seed)
    rng.shuffle(circuits)
    return circuits


# ── Weighted pool (used by curriculum learning) ───────────────────────────────

# Maps family name → generator function (common signature subset)
_FAMILY_GENERATORS: dict = {}  # populated after all generators are defined


def generate_weighted_circuits(
    families: list[str],
    family_weights: dict[str, float],
    total_count: int,
    n_qubits_range: tuple[int, int] = (3, 10),
    seed: int = 42,
    basis_gates: list[str] | None = None,
) -> list[QuantumCircuit]:
    """Generate a circuit pool with explicit per-family weights.

    Unlike generate_training_circuits() which uses equal counts per family,
    this function allocates circuits proportionally to the given weights.
    Used by CurriculumController to implement staged and blended training pools.

    Parameters
    ----------
    families      : ordered list of family names to include.
    family_weights: dict mapping family name → sampling weight (need not sum to 1;
                    they are normalised internally).
    total_count   : total circuits to generate (distributed proportionally).
    n_qubits_range: (min, max) inclusive qubit range.
    seed          : base RNG seed; each family gets seed + family_index.
    basis_gates   : target basis (default: HERON_R2_BASIS).
    """
    global _FAMILY_GENERATORS
    if not _FAMILY_GENERATORS:
        _FAMILY_GENERATORS = {
            "qv": generate_qv_circuits,
            "qaoa": generate_qaoa_circuits,
            "clifford_su4_su8": generate_clifford_su4_su8_circuits,
            "clifford_su4": generate_clifford_su4_circuits,
            "iqp": generate_iqp_circuits,
            "efficient_su2": generate_efficient_su2_circuits,
            "real_amplitudes": generate_real_amplitudes_circuits,
            "random_clifford": generate_random_clifford_circuits,
        }

    basis = basis_gates or HERON_R2_BASIS

    # Normalise weights
    total_w = sum(family_weights.get(f, 0.0) for f in families)
    if total_w == 0:
        raise ValueError(f"All family weights are zero for families={families}")

    circuits = []
    for i, family in enumerate(families):
        w = family_weights.get(family, 0.0)
        count = max(1, round(w / total_w * total_count))
        gen = _FAMILY_GENERATORS[family]
        circuits.extend(gen(n_qubits_range, count, seed + i, basis_gates=basis))

    rng = np.random.default_rng(seed)
    rng.shuffle(circuits)
    return circuits
