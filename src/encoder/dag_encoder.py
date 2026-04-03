# =============================================================================
# quetsal/graph/dag_encoder.py
# Qiskit 2.3  |  DAGCircuit → PyG Data encoder
#
# Target HW : ibm_kingston, ibm_marrakesh, ibm_fez, ibm_torino  (Heron r2)
# Paper ref : "RL for Adaptive Composition of Quantum Circuits" (Quantinuum)
#
# ┌─────────────────────────────────────────────────────────────────────┐
# │  NODE FEATURES  (dim = 10)                                          │
# │                                                                     │
# │  idx  feature           enc         rationale                       │
# │  0–5  gate_type         one-hot[6]  6 optimisable Heron r2 gates;   │
# │                                     no false ordinal relationship    │
# │  6    is_clifford        0 / 1      cz,sx,x → 1 ; rz,rx,rzz → 0    │
# │                                     not derivable from one-hot       │
# │  7    param_0  (θ/2π)   float 0–1  IBM radians ÷ 2π; 0 if no param │
# │  8    topo_pos (i/N-1)  float 0–1  unique per node; depth collapses  │
# │  9    last_pass (÷K+1)  float 0–1  paper-aligned; set by Gym step() │
# └─────────────────────────────────────────────────────────────────────┘
#
# ┌─────────────────────────────────────────────────────────────────────┐
# │  EDGE FEATURES  (dim = 6 = src_role[3] ++ dst_role[3])             │
# │  Qubit role one-hot  (paper Fig 2b):                                │
# │   0  target of 1Q gate                                              │
# │   1  1st qubit of 2Q gate  (e.g. CZ control)                       │
# │   2  2nd qubit of 2Q gate  (e.g. CZ target)                        │
# └─────────────────────────────────────────────────────────────────────┘
#
# ┌─────────────────────────────────────────────────────────────────────┐
# │  GATE VOCABULARY  (one-hot indices 0–5)                             │
# │   0  cz   2Q entangler        Clifford                              │
# │   1  rz   1Q Z-rotation       NOT Clifford  (virtual, 0 duration)  │
# │   2  rx   1Q X-rotation       NOT Clifford                         │
# │   3  sx   1Q √X               Clifford                             │
# │   4  x    1Q Pauli X          Clifford                             │
# │   5  rzz  2Q ZZ rotation      NOT Clifford  (Heron r2 native)      │
# │                                                                     │
# │  Excluded from graph (no optimisation pass touches these):          │
# │   measure, barrier, id, delay, reset                                │
# └─────────────────────────────────────────────────────────────────────┘
#
# =============================================================================

from __future__ import annotations

__all__ = ["dag_to_pyg", "verify_graph"]

import math
from collections import Counter
from typing import Optional

import torch
from torch_geometric.data import Data
from qiskit.dagcircuit import DAGOpNode

from quetsal.src.constants import (
    CLIFFORD_GATES,
    EDGE_DIM,
    GATE_INDEX,
    HERON_R2_BASIS,
    NODE_DIM,
    NUM_ACTIONS,
    NUM_GATE_TYPES,
    SKIP_GATES,
)


# ── Private helpers ───────────────────────────────────────────────────────────

def _gate_one_hot(name: str) -> list[float]:
    """
    Return a 6-element one-hot vector for gate name.
    Returns all-zeros for unknown gates — this should not occur on a correctly
    transpiled Heron r2 circuit and will be flagged by verify_graph().
    """
    vec = [0.0] * NUM_GATE_TYPES
    idx = GATE_INDEX.get(name)
    if idx is not None:
        vec[idx] = 1.0
    return vec


def _norm_param(p) -> float:
    """
    Normalise a rotation parameter to [0, 1] via (θ mod 2π) / 2π.
    IBM Qiskit gates use radians; Quantinuum paper used θ/π — we use θ/2π
    because IBM's natural range is [0, 2π], making θ/2π map cleanly to [0, 1].
    Returns 0.0 for symbolic/unbound ParameterExpression objects.
    """
    try:
        v = float(p)
        return (v % (2.0 * math.pi)) / (2.0 * math.pi)
    except (TypeError, ValueError):
        return 0.0


def _node_features(
    node: DAGOpNode,
    topo_idx: int,
    total_ops: int,
    last_pass_map: Optional[dict],
    num_passes: int,
) -> list[float]:
    """
    Build the 10-element feature vector for one DAGOpNode.

    Parameters
    ----------
    node          : DAGOpNode to encode
    topo_idx      : 0-based position in topological order over included nodes
    total_ops     : total included op node count in this DAG
    last_pass_map : {id(DAGOpNode) → int pass_index} or None.
                    WARNING: node id()s are invalidated after DAG mutation
                    (e.g. substitute_node_with_dag). The Gym env must rebuild
                    this map fresh after each pass application.
                    Pass None for initial encode → slot [9] = 0.0.
    num_passes    : K = action space size; used to normalise slot [9] to [0,1]
    """
    name = node.op.name

    # [0–5]  gate type one-hot
    feats: list[float] = _gate_one_hot(name)

    # [6]  is_clifford
    #      Explicit bit needed: Clifford membership is a semantic property.
    #      The GNN cannot infer it from the one-hot index alone without
    #      learning that mapping from data — making it explicit saves capacity.
    feats.append(1.0 if name in CLIFFORD_GATES else 0.0)

    # [7]  param_0 normalised to [0, 1]
    #      0.0 for non-parametric gates (cz, sx, x).
    #      Note: rz(0.0) after pass-induced cancellation also yields 0.0 here,
    #      but its one-hot col 1 stays hot — the GNN can still distinguish it
    #      from a non-parametric gate.
    params = node.op.params
    feats.append(_norm_param(params[0]) if len(params) > 0 else 0.0)

    # [8]  topo_pos = i / (N - 1),  clamped to [0, 1]
    #      Unique per node — two gates in the same depth layer get different
    #      topo_pos values, which depth_ratio cannot provide.
    #      Single-node circuit edge case: N=1 → topo_pos = 0.0 (safe).
    feats.append(topo_idx / max(total_ops - 1, 1))

    # [9]  last_pass_applied  → (pass_index + 1) / (K + 1)  → (0, 1]
    #      0.0 = untouched (initial circuit encode).
    #      Paper-aligned feature: gives the GNN memory of which pass most
    #      recently modified a node, reducing redundant re-application.
    #      Set exclusively by Gym env step() — encoder itself always passes
    #      the map in; never hardcoded here.
    if last_pass_map is not None and num_passes > 0:
        raw = last_pass_map.get(id(node), 0)
        feats.append(float(raw + 1) / float(num_passes + 1))
    else:
        feats.append(0.0)

    return feats   # length = NODE_DIM = 10


def _qubit_role(node: DAGOpNode, qubit_pos: int) -> list[int]:
    """
    3-element one-hot encoding which "slot" qubit qargs[qubit_pos] occupies
    at this node (paper Fig 2b).  Used to build edge features.

      0  sole target of a 1Q gate
      1  first  qubit of a 2Q gate  (e.g. CZ / RZZ qubit 0)
      2  second qubit of a 2Q gate  (e.g. CZ / RZZ qubit 1)

    The same 2Q gate emits different role vectors for its two outgoing edges,
    letting the GNN distinguish the two qubit wires even though they leave
    the same node.
    """
    n_q = len(node.qargs)
    if n_q == 1:
        role = 0
    elif n_q >= 2:
        role = 1 if qubit_pos == 0 else 2
    else:
        raise ValueError(f"Unexpected 0-qubit op node: {node.op.name}")
    vec = [0, 0, 0]
    vec[role] = 1
    return vec


# ── Public API ────────────────────────────────────────────────────────────────

def dag_to_pyg(
    dag,
    last_pass_map: Optional[dict] = None,
    num_passes: int = NUM_ACTIONS,
) -> Data:
    """
    Convert a Qiskit 2.3 DAGCircuit to a PyG Data object.

    Only the 6 optimisable Heron r2 gates (cz, rz, rx, sx, x, rzz) become
    graph nodes.  measure / barrier / id / delay / reset are skipped entirely.
    Edges represent directed qubit-wire data-flow dependencies between nodes.

    Parameters
    ----------
    dag           : DAGCircuit — obtained via circuit_to_dag(qc)
    last_pass_map : optional {id(DAGOpNode) → int pass_index}
                    Pass None on initial encode (Gym reset) → slot [9] = 0.
                    Pass the updated map from Gym env after each step().
                    Must be rebuilt after every DAG mutation — node id()s
                    are invalidated by substitute_node_with_dag().
    num_passes    : K = number of passes in the agent's action space.
                    Used to normalise slot [9] to [0, 1].
                    Defaults to NUM_ACTIONS from constants.

    Returns
    -------
    data.x            FloatTensor  [num_ops, 10]   node feature matrix
    data.edge_index   LongTensor   [2, num_edges]  COO edge index (src, dst)
    data.edge_attr    FloatTensor  [num_edges, 6]  edge feature matrix
    data.num_qubits   int          total qubits in circuit
    data.num_ops      int          number of graph nodes (included ops only)
    """
    # Filter to optimisable nodes only, preserving topological order
    all_op_nodes: list[DAGOpNode] = list(dag.topological_op_nodes())
    op_nodes = [n for n in all_op_nodes if n.op.name not in SKIP_GATES]
    total_ops = len(op_nodes)

    if total_ops == 0:
        return Data(
            x          = torch.zeros((0, NODE_DIM), dtype=torch.float),
            edge_index = torch.zeros((2, 0),        dtype=torch.long),
            edge_attr  = torch.zeros((0, EDGE_DIM), dtype=torch.float),
            num_qubits = dag.num_qubits(),
            num_ops    = 0,
        )

    node_to_idx: dict[DAGOpNode, int] = {n: i for i, n in enumerate(op_nodes)}
    included: set[DAGOpNode]          = set(op_nodes)

    # ── Node feature matrix ───────────────────────────────────────────────────
    x = torch.tensor(
        [
            _node_features(n, i, total_ops, last_pass_map, num_passes)
            for i, n in enumerate(op_nodes)
        ],
        dtype=torch.float,
    )

    # ── Edges ─────────────────────────────────────────────────────────────────
    # dag.edges() yields (src_node, dst_node, wire) triples.
    # wire is a Qubit object in Qiskit 2.x.
    # Only add an edge when both endpoints are included op nodes.
    src_list:       list[int]        = []
    dst_list:       list[int]        = []
    edge_attr_list: list[list[int]]  = []

    for src_node, dst_node, wire in dag.edges():
        if src_node not in included or dst_node not in included:
            continue

        src_idx = node_to_idx[src_node]
        dst_idx = node_to_idx[dst_node]

        # Find which qubit slot this wire occupies at each endpoint
        src_qargs = list(src_node.qargs)
        dst_qargs = list(dst_node.qargs)

        src_q_pos = src_qargs.index(wire) if wire in src_qargs else 0
        dst_q_pos = dst_qargs.index(wire) if wire in dst_qargs else 0

        # Edge feature = concat(src_role[3], dst_role[3]) → dim 6
        edge_feat = (
            _qubit_role(src_node, src_q_pos)
            + _qubit_role(dst_node, dst_q_pos)
        )
        src_list.append(src_idx)
        dst_list.append(dst_idx)
        edge_attr_list.append(edge_feat)

    if src_list:
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        edge_attr  = torch.tensor(edge_attr_list,       dtype=torch.float)
    else:
        edge_index = torch.zeros((2, 0),        dtype=torch.long)
        edge_attr  = torch.zeros((0, EDGE_DIM), dtype=torch.float)

    return Data(
        x          = x,
        edge_index = edge_index,
        edge_attr  = edge_attr,
        num_qubits = dag.num_qubits(),
        num_ops    = total_ops,
    )


def verify_graph(dag, data: Data, verbose: bool = True) -> dict[str, bool]:
    """
    Cross-check a PyG Data object against its source DAGCircuit.
    Returns a dict of named boolean checks; prints a report when verbose=True.

    Intended for debug mode only — not called during training.

    Example
    -------
    from qiskit.circuit.library import QuantumVolume
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit.converters import circuit_to_dag

    dag  = circuit_to_dag(pm.run(qc))
    data = dag_to_pyg(dag)
    verify_graph(dag, data)
    """
    all_op_nodes = list(dag.topological_op_nodes())
    op_nodes     = [n for n in all_op_nodes if n.op.name not in SKIP_GATES]
    included     = set(op_nodes)

    expected_nodes = len(op_nodes)
    expected_edges = sum(
        1 for s, d, _ in dag.edges()
        if s in included and d in included
    )

    # one_hot_valid: every included gate must be in vocab → row sum == 1.
    # All-zeros means an unknown gate slipped through — that is a bug.
    one_hot_sums = data.x[:, :NUM_GATE_TYPES].sum(dim=1)

    checks: dict[str, bool] = {
        "node_count"   : data.x.shape[0]          == expected_nodes,
        "edge_count"   : data.edge_index.shape[1] == expected_edges,
        "node_dim_10"  : data.x.shape[1]          == NODE_DIM,
        "edge_dim_6"   : (data.edge_attr.shape[1] == EDGE_DIM
                          if data.edge_attr.shape[0] > 0 else True),
        "no_self_loops": bool(
            (data.edge_index[0] != data.edge_index[1]).all()
            if data.edge_index.shape[1] > 0 else True
        ),
        "one_hot_valid": bool((one_hot_sums == 1.0).all()),
        "param_01"     : bool(
            data.x[:, 7].min() >= 0.0 and data.x[:, 7].max() <= 1.0
        ),
        "topo_01"      : bool(
            data.x[:, 8].min() >= 0.0 and data.x[:, 8].max() <= 1.0
        ),
        "lastpass_01"  : bool(
            data.x[:, 9].min() >= 0.0 and data.x[:, 9].max() <= 1.0
        ),
    }

    if verbose:
        W  = 64
        ok = lambda k: "✓" if checks[k] else "✗ FAIL"
        print(f"\n{'='*W}")
        print(f"  Quetzal — DAG → PyG  (Heron r2 | node={NODE_DIM} | edge={EDGE_DIM})")
        print(f"{'='*W}")
        print(f"  DAG included nodes : {expected_nodes:>4}   PyG nodes : {data.x.shape[0]:>4}  {ok('node_count')}")
        print(f"  DAG included edges : {expected_edges:>4}   PyG edges : {data.edge_index.shape[1]:>4}  {ok('edge_count')}")
        print(f"  Node dim = {NODE_DIM}       : {ok('node_dim_10')}")
        print(f"  Edge dim = {EDGE_DIM}        : {ok('edge_dim_6')}")
        print(f"  No self-loops      : {ok('no_self_loops')}")
        print(f"  One-hot all valid  : {ok('one_hot_valid')}")
        print(f"  param   ∈ [0, 1]   : {ok('param_01')}")
        print(f"  topo    ∈ [0, 1]   : {ok('topo_01')}")
        print(f"  lastpass∈ [0, 1]   : {ok('lastpass_01')}")

        _inv: dict[int, str] = {v: k for k, v in GATE_INDEX.items()}
        gate_ids = data.x[:, :NUM_GATE_TYPES].argmax(dim=1)
        print(f"\n  Gate distribution:")
        for gid in sorted(gate_ids.unique().tolist()):
            cnt  = (gate_ids == gid).sum().item()
            name = _inv.get(gid, "unknown")
            cliff = " [Clifford]" if name in CLIFFORD_GATES else ""
            print(f"    [{gid}] {name:<6}{cliff:<12}: {cnt:>4}")

        skipped_names = [n.op.name for n in all_op_nodes if n.op.name in SKIP_GATES]
        if skipped_names:
            print(f"\n  Skipped (not in graph):")
            for name, cnt in Counter(skipped_names).items():
                print(f"    {name:<12}: {cnt:>4}")

        print(f"\n  Sample features  (first 4 included ops):")
        print(f"  {'i':<3} {'gate':<6} {'one-hot':<10} "
              f"{'cliff':>6} {'p0':>7} {'topo':>7} {'lpass':>7}")
        print(f"  {'-'*56}")
        for i in range(min(4, data.x.shape[0])):
            row = data.x[i].tolist()
            nm  = op_nodes[i].op.name
            oh  = "".join(str(int(v)) for v in row[:6])
            print(f"  {i:<3} {nm:<6} [{oh}]  "
                  f"{row[6]:>6.1f} {row[7]:>7.4f} {row[8]:>7.4f} {row[9]:>7.4f}")

        all_ok = all(checks.values())
        print(f"\n  {'ALL CHECKS PASSED ✓' if all_ok else 'SOME CHECKS FAILED ✗'}")
        print(f"{'='*W}\n")

    return checks


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from qiskit.circuit.library import QuantumVolume
    from qiskit.converters import circuit_to_dag
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    qc  = QuantumVolume(4, seed=42)
    pm  = generate_preset_pass_manager(
              optimization_level=1, basis_gates=HERON_R2_BASIS)
    dag = circuit_to_dag(pm.run(qc))

    # Initial encode — no pass history
    data = dag_to_pyg(dag, last_pass_map=None, num_passes=NUM_ACTIONS)
    verify_graph(dag, data)

    print("Feature index reference:")
    print("  [0:6]  gate one-hot  cz=0, rz=1, rx=2, sx=3, x=4, rzz=5")
    print("  [6]    is_clifford   cz/sx/x → 1.0,  rz/rx/rzz → 0.0")
    print("  [7]    param_0       (θ mod 2π) / 2π  ;  0.0 if no param")
    print("  [8]    topo_pos      i / (N-1)  ;  unique per node")
    print("  [9]    last_pass     (pass_index+1)/(K+1) ;  0.0=untouched, set by Gym step()")
    print(f"\nDATA : {data}")
