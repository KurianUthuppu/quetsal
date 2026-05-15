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

Benchmark: 500K training steps across all 8 circuit families (4 non-parametric + 4 parametric), evaluated on **716 circuits**
(100 Clifford-SU4 + 57 Clifford-SU4-SU8 + 59 QV + 100 RandomClifford + 100 IQP + 100 EfficientSU2 + 100 QAOA + 100 RealAmplitudes),
3–10 qubits, Heron r2 basis. All optimizers receive the **same pre-transpiled input** (post layout+routing+translation, pre-optimization).

### Non-parametric circuits — 2q gate reduction (%)

| Optimizer   | Clifford-SU4 | Clifford-SU4-SU8 |       QV | RandomClifford | **Mean (316)** |
| :---------- | -----------: | ---------------: | -------: | -------------: | -------------: |
| opt_level=1 |          0.0 |              0.0 |      0.0 |            0.0 |            0.0 |
| opt_level=2 |         13.9 |             10.9 |     14.8 |            6.5 |           11.2 |
| opt_level=3 |         13.9 |             10.9 |     14.8 |            6.5 |           11.2 |
| **Quetsal** |     **21.8** |         **15.5** | **28.1** |       **20.3** |       **21.3** |

**~2× opt_level=2/3** (+10.1pp overall). Best gains: QV +13.3pp, RandomClifford +13.8pp, Clifford-SU4 +7.9pp.

### Non-parametric circuits — depth reduction (%)

| Optimizer   | Clifford-SU4 | Clifford-SU4-SU8 |       QV | RandomClifford | Mean |
| :---------- | -----------: | ---------------: | -------: | -------------: | ---: |
| opt_level=1 |         64.1 |             52.8 |     65.6 |           27.7 | 50.8 |
| opt_level=2 |         68.8 |             58.3 |     70.1 |           35.9 | 56.7 |
| opt_level=3 |         68.8 |             58.3 |     70.1 |           35.9 | 56.7 |
| **Quetsal** |     **70.1** |         **57.9** | **75.4** |       **41.8** | **59.9** |

Quetsal reduces depth further overall (59.9% vs 56.7%), with the largest gains on QV (+5.3pp) and RandomClifford (+5.9pp).

### Parametric circuits — Quetsal results

Parametric families have near-zero 2q gate reduction — the agent correctly learns to terminate early on circuits where structural cancellation is not possible. Depth reduction for EfficientSU2 and RealAmplitudes comes from 1q gate chain optimisation.

| Family         | Count | 2q reduction | Depth change | Avg elapsed (s) |
| :------------- | ----: | -----------: | -----------: | --------------: |
| IQP            |   100 |         0.0% |        −3.5% |           0.072 |
| EfficientSU2   |   100 |         0.0% |       −29.0% |           0.026 |
| QAOA           |   100 |         0.0% |        +0.0% |           0.030 |
| RealAmplitudes |   100 |         0.0% |       −23.2% |           0.023 |

### Overall (716 circuits, all families)

| Metric             | Value  |
| :----------------- | -----: |
| Mean 2q reduction  |   9.4% |
| Mean depth change  | −34.2% |

The 9.4% overall 2q figure is diluted by the four parametric families (400 circuits, ~0% reduction each). On the 316 non-parametric circuits the mean is **21.3%**.

> **Inference cost:** ~0.13–0.33 s/circuit for non-parametric families; ~0.02–0.07 s/circuit for parametric (agent exits after minimal steps).

---

## Related Work

- Mills et al. (Quantinuum, Jan 2026): _"Reinforcement Learning for Adaptive Composition of Quantum Circuit Optimisation Passes"_ — arXiv:2601.21629. Uses GINConv + PPO over PyTKET passes on the Quantinuum native gateset.
- Quetschlich et al. (TU Munich): _MQT Predictor_ — selects among preset compilation flows across Qiskit/TKET.
- IBM Research (2024): RL for Clifford synthesis and SABRE routing — individual sub-task RL.
- ZX+GNN papers: gate-level rewriting via graph rewrite rules — operates below the pass abstraction layer.

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
