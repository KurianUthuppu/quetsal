# Quetsal

### A GNN+PPO agent that adaptively sequences Qiskit optimization-stage passes — a learned replacement for the fixed pass pools behind `opt_level=1/2/3`.

> _Quetsal_ is named after the Quetzal — a bird known for navigating dense forest canopies with
> precision. Quetsal navigates the dense space of Qiskit transpilation passes, finding the optimal
> pass sequence through the optimization stage for each quantum circuit.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Qiskit](https://img.shields.io/badge/Qiskit-2.x-6929C4)](https://qiskit.org/)
![Status](https://img.shields.io/badge/status-v0.1--dev-orange)

---

## What is Quetsal?

Quetsal is a reinforcement learning agent that learns to sequence Qiskit transpilation passes adaptively per circuit, replacing the fixed pass pools behind `optimization_level=1/2/3`.

**What it is:**

- A `PassManagerStagePlugin`-compatible RL agent
- Takes a `DAGCircuit` graph encoding as input (via GINConv)
- Outputs a ranked sequence of Qiskit optimization passes
- Plugs into Qiskit's existing transpiler via the `pyproject.toml` entry point system

**What it is not:**

- Not a from-scratch transpiler or routing solver
- Not a ZX-calculus rewriter or gate-level optimizer
- Not a replacement for Qiskit's layout or routing stages

**Target metric:** Reduce 2-qubit gate count below `optimization_level=3` on QV, QAOA, and random SU4 benchmarks, 3–8 qubits, without increasing circuit depth.

---

## Architecture

```
DAGCircuit → GINConv Graph Encoder → PPO Policy (SB3) → Pass Selection Action
```

| Component     | Implementation                                                                                                                                                                          |
| ------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Graph encoder | 3-layer GINEConv (PyTorch Geometric); node features: gate one-hot[6], is_clifford, norm param, topo_pos, last_pass (dim=10); edge features: qubit role one-hot src+dst (dim=6)         |
| RL policy     | PPO (Stable-Baselines3); `QuetsalGNNPolicy` — shared GINEConv trunk + separate actor/critic linear heads                                                                               |
| Action space  | Discrete(6): `Optimize1qGatesDecomposition`, `InverseCancellation`, `CommutativeCancellation`, `ConsolidateBlocks+UnitarySynthesis` (macro), `RemoveIdentityEquivalent`, `DoNothing`   |
| Reward        | Per-step: `(prev_2q - current_2q) / initial_2q`; terminal: fixed bonus on `DoNothing`                                                                                                  |
| Target basis  | IBM Heron r2 — `cz, id, rx, rz, rzz, sx, x` (Kingston, Marrakesh, Fez, Torino)                                                                                                        |
| Integration   | `PassManagerStagePlugin` registered via entry point; drop-in replacement for the `optimization` stage                                                                                  |

---

## Quick Start

```bash
# Train (mode 1 = full 300k steps)
python -m quetsal.training.train --mode 1

# Benchmark against Qiskit baselines
python -m quetsal.benchmarks.eval \
  --model runs/quetsal/best_model/<timestamp>/best_model.zip \
  --n-circuits 50 --save-csv

# Baseline only (no model required)
python -m quetsal.benchmarks.eval --baseline-only
```

---

## Benchmarks

Baseline comparisons run against Qiskit's optimization stage at `optimization_level=1/2/3` from the same pre-optimized starting circuit (post layout+routing, pre-optimization).

Circuit families (7 total):

- Quantum Volume (QV), 3–8 qubits
- QAOA MaxCut, 3–8 qubits
- Clifford-SU4-SU8 (mixed), 3–8 qubits
- Clifford-SU4 (mixed), 3–8 qubits
- IQP (commuting diagonal), 3–8 qubits
- EfficientSU2 (VQE ansatz), 3–8 qubits
- RealAmplitudes (VQE ansatz), 3–8 qubits

Validation target: IBM Heron r2 backends (Kingston, Marrakesh, Fez, Torino).

---

## Related Work

- Mills et al. (Quantinuum, Jan 2026): _"Reinforcement Learning for Adaptive Composition of Quantum Circuit Optimisation Passes"_ — arXiv:2601.21629. Uses GINConv + PPO over PyTKET passes on the Quantinuum native gateset.
- Quetschlich et al. (TU Munich): _MQT Predictor_ — selects among preset compilation flows across Qiskit/TKET.
- IBM Research (2024): RL for Clifford synthesis and SABRE routing — individual sub-task RL.
- ZX+GNN papers: gate-level rewriting via graph rewrite rules — operates below the pass abstraction layer.

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
