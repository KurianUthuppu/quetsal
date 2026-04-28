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

Benchmark: 300K steps, 4-action ablation, 4 non-parametric families, evaluated on 316 circuits
(100 Clifford-SU4 + 57 Clifford-SU4-SU8 + 59 QV + 100 RandomClifford),
3–10 qubits, Heron r2 basis.
All optimizers receive the **same pre-transpiled input** (post layout+routing+translation, pre-optimization).

### Mean 2q gate reduction (%)

| Optimizer   | Clifford-SU4 | Clifford-SU4-SU8 |       QV | RandomClifford | **Total (316)** |
| :---------- | -----------: | ---------------: | -------: | -------------: | --------------: |
| opt_level=0 |          0.0 |              0.0 |      0.0 |            0.0 |             0.0 |
| opt_level=1 |         13.9 |             10.9 |     14.8 |            6.5 |            11.2 |
| opt_level=2 |         13.9 |             10.9 |     14.8 |            6.5 |            11.2 |
| **Quetsal** |     **22.9** |         **15.3** | **27.5** |       **22.2** |        **22.2** |

**~2× opt_level=1/2** (+11.0pp overall). Best gains: RandomClifford +15.7pp, QV +12.6pp, Clifford-SU4 +9.1pp.

### Mean depth change (%) _(negative = better)_

| Optimizer   | Clifford-SU4 | Clifford-SU4-SU8 |        QV | RandomClifford | Total |
| :---------- | -----------: | ---------------: | --------: | -------------: | ----: |
| opt_level=2 |        −68.8 |            −58.3 |     −70.1 |          −35.9 | −56.7 |
| **Quetsal** |    **−71.0** |            −57.3 | **−74.1** |          −34.5 | −57.6 |

Depth impact is neutral overall (−57.6% vs −56.7%). Quetsal improves depth on Clifford-SU4 (−2.2pp)
and QV (−4.0pp); RandomClifford depth increases slightly (+1.4pp).

---

## Related Work

- Mills et al. (Quantinuum, Jan 2026): _"Reinforcement Learning for Adaptive Composition of Quantum Circuit Optimisation Passes"_ — arXiv:2601.21629. Uses GINConv + PPO over PyTKET passes on the Quantinuum native gateset.
- Quetschlich et al. (TU Munich): _MQT Predictor_ — selects among preset compilation flows across Qiskit/TKET.
- IBM Research (2024): RL for Clifford synthesis and SABRE routing — individual sub-task RL.
- ZX+GNN papers: gate-level rewriting via graph rewrite rules — operates below the pass abstraction layer.

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
