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
- Not a standalone ZX-calculus rewriter — ZX-calculus (`ZXFullReduce` via PyZX) is one selectable action among seven; the RL agent decides when and whether to invoke it
- Not a replacement for Qiskit's layout or routing stages

**Target metric:** Reduce 2-qubit gate count below `optimization_level=3` across diverse circuit families — Quantum Volume, QAOA, Clifford-SU4, Clifford-SU4-SU8, IQP, EfficientSU2, and RealAmplitudes, without increasing circuit depth.

---

## Architecture

```
DAGCircuit → GINConv Graph Encoder → PPO Policy (SB3) → Pass Selection Action
```

| Component     | Implementation                                                                                                                                                                                        |
| ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Graph encoder | 4-layer GINEConv (PyTorch Geometric); node features: gate one-hot[6], is_clifford, norm param, topo_pos, last_pass (dim=10); edge features: qubit role one-hot src+dst (dim=6)                        |
| RL policy     | PPO (Stable-Baselines3); `QuetsalGNNPolicy` — shared GINEConv trunk + separate actor/critic linear heads                                                                                              |
| Action space  | Discrete(7): `Optimize1qGatesDecomposition`, `CommutativeInverseCancellation`, `ConsolidateAndSynthesize` (macro: CB→US), `OptimizeCliffords`, `Split2QUnitaries`, `ZXFullReduce` (pyzx), `DoNothing` |
| Reward        | Per-step: `(prev_2q − current_2q) / initial_2q − depth_penalty − step_penalty`; terminal bonus on `DoNothing`                                                                                         |
| Target basis  | IBM Heron r2 — `cz, id, rx, rz, rzz, sx, x` (Kingston, Marrakesh, Fez, Torino)                                                                                                                        |
| Integration   | `PassManagerStagePlugin` registered via `[project.entry-points."qiskit.transpiler.optimization"]`; drop-in replacement for the `optimization` stage                                                   |

---

## Quick Start

```bash
# Train (mode 1 = full 500k steps)
python -m quetsal.training.train --mode 1

# Benchmark against Qiskit baselines
python -m quetsal.benchmarks.eval \
  --model runs/quetsal/best_model/<timestamp>/best_model.zip \
  --n-circuits 100 --save-csv

# Baseline only (no model required)
python -m quetsal.benchmarks.eval --baseline-only
```

Use via plugin (after `pip install -e quetsal/`):

```python
from quetsal.src.plugin.quetsal_plugin import QuetsalPlugin
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

plugin = QuetsalPlugin(model_path="runs/quetsal/best_model/20260429_042101/best_model.zip")
pm = generate_preset_pass_manager(optimization_level=1, basis_gates=..., coupling_map=cm)
pm.optimization = plugin.pass_manager(pass_manager_config=None)
optimized_circuit = pm.run(raw_circuit)
```

---

## Current Results

Benchmark: fine-tuned Quetsal run evaluated on **716 circuits**
(100 Clifford-SU4 + 57 Clifford-SU4-SU8 + 59 QV + 100 RandomClifford + 100 IQP + 100 EfficientSU2 + 100 QAOA + 100 RealAmplitudes),
3-10 qubits, Heron r2 basis. All optimizers receive the **same pre-transpiled input** (post layout+routing+translation, pre-optimization).

### Non-parametric circuits — 2q gate reduction (%)

| Optimizer   | Clifford-SU4 | Clifford-SU4-SU8 |       QV | RandomClifford | **Mean (316)** |
| :---------- | -----------: | ---------------: | -------: | -------------: | -------------: |
| opt_level=1 |          0.0 |              0.0 |      0.0 |            0.0 |            0.0 |
| opt_level=2 |         13.9 |             10.9 |     14.8 |            6.5 |           11.2 |
| opt_level=3 |         13.9 |             10.9 |     14.8 |            6.5 |           11.2 |
| **Quetsal** |     **23.9** |         **15.7** | **27.4** |       **21.8** |       **22.4** |

**~2x opt_level=2/3** (+11.2pp overall). Best gains: RandomClifford +15.3pp, QV +12.6pp, Clifford-SU4 +10.1pp.

### Non-parametric circuits — depth reduction (%)

| Optimizer   | Clifford-SU4 | Clifford-SU4-SU8 |       QV | RandomClifford | Mean |
| :---------- | -----------: | ---------------: | -------: | -------------: | ---: |
| opt_level=1 |         64.1 |             52.8 |     65.6 |           27.7 | 50.8 |
| opt_level=2 |         68.8 |             58.3 |     70.1 |           35.9 | 56.7 |
| opt_level=3 |         68.8 |             58.3 |     70.1 |           35.9 | 56.7 |
| **Quetsal** |     **71.2** |         **57.4** | **75.0** |       **39.4** | **59.4** |

Quetsal reduces depth further overall (59.4% vs 56.7%), with the largest gains on QV (+4.9pp), RandomClifford (+3.5pp), and Clifford-SU4 (+2.4pp). Clifford-SU4-SU8 remains close to Qiskit's opt_level=2/3 depth while improving 2q reduction.

### Parametric circuits — 2q gate reduction (%)

Parametric families have near-zero 2q gate reduction across Qiskit and Quetsal; the meaningful comparison is depth.

| Optimizer   |  IQP | EfficientSU2 | QAOA | RealAmplitudes | Mean |
| :---------- | ---: | -----------: | ---: | -------------: | ---: |
| opt_level=1 |  0.0 |          0.0 |  0.0 |            0.0 |  0.0 |
| opt_level=2 |  0.0 |          0.0 |  0.0 |            0.0 |  0.0 |
| opt_level=3 |  0.0 |          0.0 |  0.0 |            0.0 |  0.0 |
| **Quetsal** |  0.0 |          0.0 |  0.0 |            0.0 |  0.0 |

### Parametric circuits — depth reduction (%)

| Optimizer   |  IQP | EfficientSU2 | QAOA | RealAmplitudes | Mean |
| :---------- | ---: | -----------: | ---: | -------------: | ---: |
| opt_level=1 | 11.7 |         29.0 |  2.0 |           23.2 | 16.5 |
| opt_level=2 | 11.7 |         48.5 |  2.0 |           45.7 | 27.0 |
| opt_level=3 | 11.7 |         48.5 |  2.0 |           45.7 | 27.0 |
| **Quetsal** | 11.7 |         29.0 |  2.0 |           23.2 | 16.5 |

Quetsal currently matches opt_level=1 depth behaviour on EfficientSU2 and RealAmplitudes, while opt_level=2/3 still achieve substantially deeper reductions on those two ansatz families. This is the main remaining optimization target.

### Overall (716 circuits, all families)

| Metric             | Value  |
| :----------------- | -----: |
| Mean 2q reduction  |   9.9% |
| Mean depth change  | -35.4% |

The 9.9% overall 2q figure is diluted by the four parametric families (400 circuits, ~0% reduction each). On the 316 non-parametric circuits the mean is **22.4%**.

---

## Related Work

- Mills et al. (Quantinuum, Jan 2026): _"Reinforcement Learning for Adaptive Composition of Quantum Circuit Optimisation Passes"_ — arXiv:2601.21629. Uses GINConv + PPO over PyTKET passes on the Quantinuum native gateset.
- Quetschlich et al. (TU Munich): _MQT Predictor_ — selects among preset compilation flows across Qiskit/TKET.
- IBM Research (2024): RL for Clifford synthesis and SABRE routing — individual sub-task RL.
- ZX+GNN papers: gate-level rewriting via graph rewrite rules — operates below the pass abstraction layer.

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
