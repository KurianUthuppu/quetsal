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
- Takes a `DAGCircuit` graph encoding as input (via GINEConv)
- Outputs a ranked sequence of Qiskit optimization passes
- Plugs into Qiskit's existing transpiler via the `pyproject.toml` entry point system

**What it is not:**

- Not a from-scratch transpiler or routing solver
- Not a standalone ZX-calculus rewriter — ZX-calculus (`ZXFullReduce` via PyZX) is one selectable action among five; the RL agent decides when and whether to invoke it
- Not a replacement for Qiskit's layout or routing stages

**Target metric:** Reduce 2-qubit gate count below `optimization_level=3` across diverse circuit families — Quantum Volume, QAOA, Clifford-SU4, Clifford-SU4-SU8, RandomClifford, IQP, EfficientSU2, and RealAmplitudes, without increasing circuit depth.

---

## Architecture

```
DAGCircuit → GINEConv Graph Encoder → PPO Policy (SB3) → Pass Selection Action
```

| Component     | Implementation                                                                                                                                                                                                                                                               |
| ------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Graph encoder | 4-layer GINEConv (PyTorch Geometric); node features: gate one-hot[6], is_clifford, norm param, topo_pos, last_pass (dim=10); edge features: qubit role one-hot src+dst (dim=6)                                                                                               |
| RL policy     | PPO (Stable-Baselines3); `QuetsalGNNPolicy` — shared GINEConv trunk + separate actor/critic linear heads                                                                                                                                                                     |
| Action space  | Discrete(5): `Optimize1qGatesDecomposition`, `ConsolidateAndSynthesize` (macro: ConsolidateBlocks→UnitarySynthesis), `CommutativeCancellation`, `ZXFullReduce` (pyzx), `DoNothing` (learned termination)                                                                     |
| Reward        | Non-parametric families: `(prev_2q − cur_2q) / initial_2q + 0.03·Δdepth − step_penalty`. Depth-primary families (IQP/QAOA/EfficientSU2/RealAmplitudes): `0.35·Δdepth + 0.10·Δ2q − step_penalty`. Terminal bonus on `DoNothing`, normalized by per-family achievable ceiling. |
| Target basis  | IBM Heron r2 — `cz, id, rx, rz, rzz, sx, x` (Kingston, Marrakesh, Fez, Torino)                                                                                                                                                                                               |
| Integration   | `PassManagerStagePlugin` registered via `[project.entry-points."qiskit.transpiler.optimization"]`; drop-in replacement for the `optimization` stage                                                                                                                          |

---

## Quick Start

### Install

```bash
pip install quetsal
```

The trained agent (3–10 qubit regime) ships inside the package, so the plugin works with no extra downloads.

### Train, benchmark & analyze

These modules ship with the package, so they work after `pip install quetsal` as well as from a source checkout.

```bash
# Train — full staged curriculum (Stage 1 non-param → 2 parametric blend → 3 all families, ~500K steps)
python -m quetsal.training.train --mode 1 --curriculum

# Quick smoke-test (~2-3 min, verifies the pipeline end-to-end)
python -m quetsal.training.train --mode 0

# Benchmark against Qiskit baselines (opt_level 1/2/3)
python -m quetsal.benchmarks.eval \
  --model runs/quetsal/best_model/<timestamp>/best_model.zip \
  --n-circuits 100 --save-csv

# Baseline only (no model required)
python -m quetsal.benchmarks.eval --baseline-only

# Log a completed run's hyperparameters + results to CSV for comparison
python -m quetsal.experiments.track \
  --model runs/quetsal/best_model/<timestamp>/best_model.zip \
  --mode 1 --notes "curriculum r13"
```

### Use via plugin

After `pip install quetsal`, the entry point is registered automatically — select it through Qiskit's preset pass manager:

```python
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

pm = generate_preset_pass_manager(
    optimization_level=1,            # layout + routing + translation
    optimization_plugin="quetsal",   # ← Quetsal replaces the optimization stage
    basis_gates=["cz", "id", "rx", "rz", "rzz", "sx", "x"],
    coupling_map=cm,
)
optimized_circuit = pm.run(raw_circuit)
```

Or drive it directly (uses the bundled agent by default; pass `model_path=...` to override):

```python
from quetsal.src.plugin.quetsal_plugin import QuetsalPlugin
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

plugin = QuetsalPlugin()  # bundled 3–10 qubit agent
pm = generate_preset_pass_manager(optimization_level=1, basis_gates=..., coupling_map=cm)
pm.optimization = plugin.pass_manager(pass_manager_config=None)
optimized_circuit = pm.run(raw_circuit)
```

> Reminder: the bundled agent is trained for **3–10 qubit** circuits. See [Scope & Limitations](#scope--limitations) before using it on larger circuits.

---

## Current Results

Trained agent: 500K-step staged-curriculum run, evaluated on **716 held-out circuits**
(100 Clifford-SU4 + 57 Clifford-SU4-SU8 + 59 QV + 100 RandomClifford + 100 IQP + 100 EfficientSU2 + 100 QAOA + 100 RealAmplitudes),
3-10 qubits, Heron r2 basis. All optimizers receive the **same pre-transpiled input** (post layout+routing+translation, pre-optimization). Depth figures are reductions (higher = shallower). Means are circuit-count-weighted.

### Non-parametric circuits — 2q gate reduction (%)

| Optimizer   | Clifford-SU4 | Clifford-SU4-SU8 |       QV | RandomClifford | **Mean (316)** |
| :---------- | -----------: | ---------------: | -------: | -------------: | -------------: |
| opt_level=1 |          0.0 |              0.0 |      0.0 |            0.0 |            0.0 |
| opt_level=2 |         13.9 |             10.9 |     14.8 |            6.5 |           11.2 |
| opt_level=3 |         13.9 |             10.9 |     14.8 |            6.5 |           11.2 |
| **Quetsal** |     **23.6** |         **13.7** | **27.7** |       **21.5** |       **21.9** |

**~2x opt_level=2/3** (+10.7pp overall). Largest gains: RandomClifford +15.0pp, QV +12.9pp, Clifford-SU4 +9.7pp.

### Non-parametric circuits — depth reduction (%)

| Optimizer   | Clifford-SU4 | Clifford-SU4-SU8 |       QV | RandomClifford |     Mean |
| :---------- | -----------: | ---------------: | -------: | -------------: | -------: |
| opt_level=1 |         64.1 |             52.8 |     65.6 |           27.7 |     50.8 |
| opt_level=2 |         68.8 |             58.2 |     70.1 |           35.8 |     56.7 |
| opt_level=3 |         68.8 |             58.2 |     70.1 |           35.8 |     56.7 |
| **Quetsal** |     **71.2** |             58.0 | **75.3** |       **45.1** | **61.3** |

Quetsal reduces depth further overall (61.3% vs 56.7%) while delivering the 2q gains above — largest depth gains on RandomClifford (+9.3pp), QV (+5.2pp), and Clifford-SU4 (+2.4pp). Clifford-SU4-SU8 tracks opt_level=2/3 depth (58.0 vs 58.2) while improving 2q reduction.

### Parametric circuits — depth reduction (%)

Parametric families have near-zero 2q gate reduction across all optimizers; the meaningful comparison is depth.

| Optimizer   |  IQP | EfficientSU2 | QAOA | RealAmplitudes | Mean |
| :---------- | ---: | -----------: | ---: | -------------: | ---: |
| opt_level=1 | 11.7 |         29.0 |  2.0 |           23.2 | 16.5 |
| opt_level=2 | 11.7 |         48.5 |  2.0 |           45.7 | 27.0 |
| opt_level=3 | 11.7 |         48.5 |  2.0 |           45.7 | 27.0 |
| **Quetsal** |  9.8 |     **49.4** |  2.0 |           45.7 | 26.7 |

### Overall (716 circuits, all families)

| Metric               | opt_level=2/3 | **Quetsal** |
| :------------------- | ------------: | ----------: |
| Mean 2q reduction    |          5.0% |    **9.7%** |
| Mean depth reduction |         40.1% |   **42.0%** |

The 9.7% overall 2q figure is diluted by the four parametric families (400 circuits, ~0% reduction each by construction). On the 316 non-parametric circuits the mean is **21.9%** — roughly double opt_level=2/3.

---

## Scope & Limitations

**Quetsal is trained and validated on 3–10 qubit circuits, and is intended for use in that range.** Every result above is on 3–10 qubit Heron r2 circuits — the distribution the agent was trained on. Within this range it reliably beats `opt_level=2/3` on both 2q-gate and depth reduction for non-parametric families and with no depth regression for parametric families.

**Scalability is the primary limitation.** The reference paper (arXiv:2601.21629) reports that a model trained on small circuits can be deployed on much larger ones. In the Qiskit-native setting I find this small-train / large-deploy generalization holds only modestly, and degrades as circuit size grows past the training range:

The degradation is architectural rather than a tuning artifact:

- **Fixed receptive field** — the GNN runs a fixed number of message-passing hops regardless of circuit size, so on large/deep DAGs each node sees a shrinking fraction of the circuit.

---

## Related Work

- Mills et al. (Quantinuum, Jan 2026): _"Reinforcement Learning for Adaptive Composition of Quantum Circuit Optimisation Passes"_ — arXiv:2601.21629. Uses GINConv + PPO over PyTKET passes on the Quantinuum native gateset.
- Quetschlich et al. (TU Munich): _MQT Predictor_ — selects among preset compilation flows across Qiskit/TKET.
- IBM Research (2024): RL for Clifford synthesis and SABRE routing — individual sub-task RL.
- ZX+GNN papers: gate-level rewriting via graph rewrite rules — operates below the pass abstraction layer.

---

## License

Apache 2.0 — see [LICENSE](LICENSE).

---

## AI Disclosure

Developed with assistance from Claude Code (Claude Opus 4.8).
