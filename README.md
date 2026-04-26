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
# Train (mode 1 = full 300k steps)
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

plugin = QuetsalPlugin(model_path="runs/quetsal/final_20260417_215040.zip")
pm = generate_preset_pass_manager(optimization_level=1, basis_gates=..., coupling_map=cm)
pm.optimization = plugin.pass_manager(pass_manager_config=None)
optimized_circuit = pm.run(raw_circuit)
```

---

## Current Results

Benchmark: model `20260422` (300k steps, non-parametric families), evaluated on 460 circuits
(100 Clifford-SU4 + 77 Clifford-SU4-SU8 + 100 PauliGadget + 83 QV + 100 RandomClifford),
3–8 qubits, Heron r2 basis.
All optimizers receive the **same pre-transpiled input** (post layout+routing+translation, pre-optimization).

### Mean 2q gate reduction (%)

| Optimizer   | Clifford-SU4 | Clifford-SU4-SU8 | PauliGadget |       QV | RandomClifford | **Total (460)** |
| :---------- | -----------: | ---------------: | ----------: | -------: | -------------: | --------------: |
| opt_level=1 |          0.0 |              0.0 |         0.0 |      0.0 |            0.0 |             0.0 |
| opt_level=2 |         17.2 |             11.7 |        63.3 |     16.4 |            5.1 |            23.5 |
| opt_level=3 |         17.3 |             11.7 |        63.3 |     16.4 |            5.1 |            23.5 |
| **Quetsal** |     **21.9** |         **16.5** |    **63.3** | **22.5** |       **21.5** |        **30.1** |

**+6.6pp over opt_level=3** across all families (best: +16.4pp on RandomClifford, +6.1pp on QV).
PauliGadget at ceiling — opt_level=2 already achieves full reduction; no headroom for RL.

### Mean depth change (%) _(negative = better)_

| Optimizer   | Clifford-SU4 | Clifford-SU4-SU8 | PauliGadget |        QV | RandomClifford |     Total |
| :---------- | -----------: | ---------------: | ----------: | --------: | -------------: | --------: |
| opt_level=3 |        −70.8 |            −59.0 |       −67.2 |     −70.3 |          −37.2 |     −60.6 |
| **Quetsal** |    **−71.2** |            −58.4 |       −57.7 | **−72.6** |      **−44.6** | **−60.6** |

Depth impact is neutral overall (−60.6% both). Quetsal improves depth on QV (−2.3pp) and
RandomClifford (−7.4pp); PauliGadget depth increases slightly (+9.5pp) but 2q reduction is unchanged.

---

## Related Work

- Mills et al. (Quantinuum, Jan 2026): _"Reinforcement Learning for Adaptive Composition of Quantum Circuit Optimisation Passes"_ — arXiv:2601.21629. Uses GINConv + PPO over PyTKET passes on the Quantinuum native gateset.
- Quetschlich et al. (TU Munich): _MQT Predictor_ — selects among preset compilation flows across Qiskit/TKET.
- IBM Research (2024): RL for Clifford synthesis and SABRE routing — individual sub-task RL.
- ZX+GNN papers: gate-level rewriting via graph rewrite rules — operates below the pass abstraction layer.

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
