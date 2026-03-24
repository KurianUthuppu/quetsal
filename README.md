# Quetsal

### A GNN+PPO agent that selects Qiskit transpilation passes — an intelligence layer above `opt_level=1/2`, not a from-scratch transpiler.

> _Quetsal_ is named after the Quetzal — a bird known for navigating dense forest canopies with
> precision. Quetsal navigates the dense space of Qiskit transpilation passes, finding the optimal
> circuit through optimization for each quantum circuit.

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

**Target metric:** Reduce 2-qubit gate count below `optimization_level=2` on QV, QAOA, and random SU4 benchmarks, 3–8 qubits, without increasing circuit depth.

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

## Roadmap

| Milestone                         | Target     | Scope                                                                                    |
| --------------------------------- | ---------- | ---------------------------------------------------------------------------------------- |
| **M1 — Environment + encoding**   | April 2026 | `DAGCircuit → PyG graph`, Gymnasium env (reset/step/reward), benchmark circuit suite     |
| **M2 — PPO agent end-to-end**     | May 2026   | GINConv encoder wired to SB3 PPO, first training run, simulator results vs `opt_level=2` |
| **M3 — Real hardware validation** | June 2026  | IBM Kingston runs, reward tuning, plugin integration via `PassManagerStagePlugin`        |
| **M4 — Release + arXiv**          | July 2026  | PyPI package, arXiv quant-ph submission, `v1.0` tag                                      |

`v0.1-dev` — target: end of April 2026

---

## Related Work

- Mills et al. (Quantinuum, Jan 2026): _"Reinforcement Learning for Adaptive Composition of Quantum Circuit Optimisation Passes"_ — arXiv:2601.21629. Uses GINConv + PPO over PyTKET passes on the Quantinuum native gateset.
- Quetschlich et al. (TU Munich): _MQT Predictor_ — selects among preset compilation flows across Qiskit/TKET.
- IBM Research (2024): RL for Clifford synthesis and SABRE routing — individual sub-task RL.
- ZX+GNN papers: gate-level rewriting via graph rewrite rules — operates below the pass abstraction layer.

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
