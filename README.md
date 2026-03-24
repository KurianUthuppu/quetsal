# Quetsal

### ### A GNN+PPO agent that adaptively sequences Qiskit optimization-stage passes — a learned replacement for the fixed pass pools behind `opt_level=1/2/3`.

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

| Component     | Implementation                                                                                                                                       |
| ------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| Graph encoder | 3-layer GINConv (PyTorch Geometric) node features: Ex.: gate type, qubit index, gate error rate etc:-                                                |
| RL policy     | PPO (Stable-Baselines3); custom `ActorCriticPolicy` with GNN feature extractor                                                                       |
| Action space  | Discrete — subset of Qiskit optimization passes (e.g. `CXCancellation`, `CommutativeCancellation`, `ConsolidateBlocks`, `OptimizeSwapBeforeMeasure`) |
| Reward        | `Δ_2q_gates / initial_2q_gates` — normalized two-qubit gate count reduction                                                                          |
| Integration   | `PassManagerStagePlugin` registered via entry point; drop-in replacement for the `optimization` stage                                                |

---

## Benchmarks

Baseline comparisons run against `generate_preset_pass_manager` at `optimization_level=1/2/3`.

Circuit families:

- Quantum Volume (QV) circuits, 3–8 qubits
- QAOA MaxCut circuits, 3–8 qubits
- Random SU4 circuits, 3–8 qubits

Validation target: IBM Kingston backend (156-qubit Heron r2).

---

## Related Work

- Mills et al. (Quantinuum, Jan 2026): _"Reinforcement Learning for Adaptive Composition of Quantum Circuit Optimisation Passes"_ — arXiv:2601.21629. Uses GINConv + PPO over PyTKET passes on the Quantinuum native gateset.
- Quetschlich et al. (TU Munich): _MQT Predictor_ — selects among preset compilation flows across Qiskit/TKET.
- IBM Research (2024): RL for Clifford synthesis and SABRE routing — individual sub-task RL.
- ZX+GNN papers: gate-level rewriting via graph rewrite rules — operates below the pass abstraction layer.

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
