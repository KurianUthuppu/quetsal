# =============================================================================
# quetsal/benchmarks/eval.py
# Baseline benchmark: Qiskit optimization_level=1/2/3 vs Quetsal agent.
#
# Usage (Qiskit baseline only, no agent required):
#   python -m quetsal.benchmarks.eval --baseline-only
#
# Usage (full comparison, requires a trained model):
#   python -m quetsal.benchmarks.eval --model runs/quetsal/best_model/<ts>/best_model.zip
#
# Metrics reported per circuit:
#   2q_before   : 2q gate count after layout/routing (before optimization)
#   2q_after    : 2q gate count after the optimizer under test
#   reduction   : (2q_before - 2q_after) / 2q_before  [0..1]
#   depth_before / depth_after
#   steps_taken : number of passes applied (agent only)
# =============================================================================

from __future__ import annotations

__all__ = [
    "CircuitResult",
    "BenchmarkResult",
    "run_qiskit_baseline",
    "run_agent_baseline",
    "print_summary",
    "save_per_circuit_csv",
]

import argparse
import csv
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from qiskit import QuantumCircuit
from qiskit.transpiler import CouplingMap, PassManager
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from quetsal.src.constants import HERON_R2_BASIS, MAX_STEPS_PER_EPISODE, SKIP_GATES


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class CircuitResult:
    name: str
    family: str
    n_qubits: int
    two_q_before: int
    two_q_after: int
    depth_before: int
    depth_after: int
    steps_taken: int        # passes applied; -1 for Qiskit baselines
    elapsed_s: float

    @property
    def reduction(self) -> float:
        if self.two_q_before == 0:
            return 0.0
        return (self.two_q_before - self.two_q_after) / self.two_q_before

    @property
    def depth_change(self) -> float:
        if self.depth_before == 0:
            return 0.0
        return (self.depth_after - self.depth_before) / self.depth_before


@dataclass
class BenchmarkResult:
    label: str
    results: list[CircuitResult] = field(default_factory=list)

    @property
    def mean_reduction(self) -> float:
        valid = [r.reduction for r in self.results if r.two_q_before > 0]
        return sum(valid) / len(valid) if valid else 0.0

    @property
    def mean_depth_change(self) -> float:
        valid = [r.depth_change for r in self.results if r.depth_before > 0]
        return sum(valid) / len(valid) if valid else 0.0

    @property
    def mean_elapsed_s(self) -> float:
        return sum(r.elapsed_s for r in self.results) / len(self.results) if self.results else 0.0


# ── Helpers ───────────────────────────────────────────────────────────────────

def _count_2q(qc) -> int:
    """Count 2-qubit gates in a QuantumCircuit (excluding SKIP_GATES)."""
    return sum(
        1 for inst in qc.data
        if inst.operation.name not in SKIP_GATES and len(inst.qubits) >= 2
    )


def _run_opt_level(qc, opt_level: int, seed: int = 42) -> tuple["QuantumCircuit", float]:
    """Run only the optimization stage at a given opt_level on an already-transpiled circuit.

    Circuits from generate_training_circuits() are already through layout+routing.
    Skipping init/layout/routing/translation via empty PassManagers ensures the
    property_set is correctly threaded through pm.run() — unlike pm.optimization.run()
    which creates an isolated property_set and breaks passes that depend on it.
    """
    cm = CouplingMap.from_line(qc.num_qubits)
    pm = generate_preset_pass_manager(
        optimization_level=opt_level,
        basis_gates=HERON_R2_BASIS,
        coupling_map=cm,
        seed_transpiler=seed,
    )
    pm.init = PassManager()
    pm.layout = PassManager()
    pm.routing = PassManager()
    pm.translation = PassManager()
    pm.scheduling = PassManager()
    t0 = time.time()
    out = pm.run(qc)
    return out, time.time() - t0


_NONPARAM_FAMILIES = {
    "QV", "Clifford-SU4", "Clifford-SU4-SU8",
    "IQP", "RandomClifford",
}

_ALL_FAMILY_KEYS = [
    "QV", "QAOA", "Clifford-SU4-SU8", "Clifford-SU4",
    "IQP", "EfficientSU2", "RealAmplitudes", "RandomClifford",
]


def _generate_tagged_circuits(
    n_qubits_range: tuple[int, int],
    count_per_family: int,
    seed: int,
    families: set[str] | None = None,
) -> list[tuple[QuantumCircuit, str]]:
    """Generate circuits from the requested families, each tagged with its family name.

    ``families`` is a set of family keys (e.g. ``{"QV", "Clifford-SU4"}``).
    Pass ``None`` (default) to include all 8 families.
    """
    from quetsal.src.environment.circuits import (
        generate_clifford_su4_circuits,
        generate_clifford_su4_su8_circuits,
        generate_efficient_su2_circuits,
        generate_iqp_circuits,
        generate_qaoa_circuits,
        generate_qv_circuits,
        generate_random_clifford_circuits,
        generate_real_amplitudes_circuits,
    )
    import numpy as np

    basis = HERON_R2_BASIS
    all_families = [
        ("QV",               generate_qv_circuits(n_qubits_range, count_per_family, seed,     basis)),
        ("QAOA",             generate_qaoa_circuits(n_qubits_range, count_per_family, seed+1,  basis_gates=basis)),
        ("Clifford-SU4-SU8", generate_clifford_su4_su8_circuits(n_qubits_range, count_per_family, seed+2, basis)),
        ("Clifford-SU4",     generate_clifford_su4_circuits(n_qubits_range, count_per_family, seed+3, basis_gates=basis)),
        ("IQP",              generate_iqp_circuits(n_qubits_range, count_per_family, seed+4,  basis)),
        ("EfficientSU2",     generate_efficient_su2_circuits(n_qubits_range, count_per_family, seed+5, basis_gates=basis)),
        ("RealAmplitudes",   generate_real_amplitudes_circuits(n_qubits_range, count_per_family, seed+6, basis_gates=basis)),
        ("RandomClifford",   generate_random_clifford_circuits(n_qubits_range, count_per_family, seed+7, basis_gates=basis)),
    ]

    tagged: list[tuple[QuantumCircuit, str]] = []
    for family_name, circuits in all_families:
        if families is not None and family_name not in families:
            continue
        for qc in circuits:
            tagged.append((qc, family_name))

    rng = np.random.default_rng(seed)
    rng.shuffle(tagged)
    return tagged


# ── Baseline runner ───────────────────────────────────────────────────────────

def run_qiskit_baseline(
    tagged_circuits: list[tuple[QuantumCircuit, str]],
    opt_levels: list[int] = (1, 2, 3),
    seed: int = 42,
) -> list[BenchmarkResult]:
    """Benchmark Qiskit optimization_level=1/2/3 on a list of tagged QuantumCircuits.

    Circuits from generate_training_circuits() are already through layout+routing.
    Only the optimization stage is run at each opt_level from that same starting point.
    """
    benchmarks = {lvl: BenchmarkResult(label=f"opt_level={lvl}") for lvl in opt_levels}

    for i, (qc, family) in enumerate(tagged_circuits):
        two_q_before = _count_2q(qc)
        depth_before = qc.depth()

        for lvl in opt_levels:
            optimized, elapsed = _run_opt_level(qc, opt_level=lvl, seed=seed)
            two_q_after = _count_2q(optimized)
            depth_after = optimized.depth()
            benchmarks[lvl].results.append(CircuitResult(
                name=f"circuit_{i}",
                family=family,
                n_qubits=qc.num_qubits,
                two_q_before=two_q_before,
                two_q_after=two_q_after,
                depth_before=depth_before,
                depth_after=depth_after,
                steps_taken=-1,
                elapsed_s=elapsed,
            ))

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(tagged_circuits)}] done")

    return list(benchmarks.values())


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_summary(results: list[BenchmarkResult]) -> None:
    print()
    print(f"{'Optimizer':<20} {'Mean 2q reduction':>18} {'Mean depth change':>18} {'N circuits':>12} {'Mean time (s)':>14}")
    print("-" * 88)
    for r in results:
        print(
            f"{r.label:<20} "
            f"{r.mean_reduction * 100:>17.1f}% "
            f"{r.mean_depth_change * 100:>17.1f}% "
            f"{len(r.results):>12} "
            f"{r.mean_elapsed_s:>13.3f}s"
        )
    print()


def save_per_circuit_csv(
    results: list[BenchmarkResult],
    out_dir: str = "benchmarks/results",
) -> None:
    """Save all optimizers into a single CSV with an 'optimizer' column."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(out_dir) / f"benchmark_{timestamp}.csv"

    columns = [
        "optimizer", "circuit", "family", "n_qubits",
        "two_q_before", "two_q_after", "two_q_reduction_pct",
        "depth_before", "depth_after", "depth_change_pct",
        "steps_taken", "elapsed_s",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for br in results:
            for r in br.results:
                writer.writerow({
                    "optimizer":           br.label,
                    "circuit":             r.name,
                    "family":              r.family,
                    "n_qubits":            r.n_qubits,
                    "two_q_before":        r.two_q_before,
                    "two_q_after":         r.two_q_after,
                    "two_q_reduction_pct": round(r.reduction * 100, 2),
                    "depth_before":        r.depth_before,
                    "depth_after":         r.depth_after,
                    "depth_change_pct":    round(r.depth_change * 100, 2),
                    "steps_taken":         "NA" if r.steps_taken == -1 else r.steps_taken,
                    "elapsed_s":           round(r.elapsed_s, 4),
                })
    print(f"[benchmark] Per-circuit CSV -> {path}")


# ── Agent runner ─────────────────────────────────────────────────────────────

def run_agent_baseline(
    model_path: str,
    tagged_circuits: list[tuple[QuantumCircuit, str]],
) -> BenchmarkResult:
    """Evaluate a trained Quetsal agent on tagged circuits."""
    from quetsal.src.agent.ppo_agent import load_agent
    from quetsal.src.environment.pass_env import PassManagerEnv
    from qiskit.converters import dag_to_circuit

    first_qc = tagged_circuits[0][0]
    dummy_env = PassManagerEnv(circuits=[first_qc], max_steps=MAX_STEPS_PER_EPISODE)
    model = load_agent(model_path, env=dummy_env)
    result = BenchmarkResult(label="Quetsal agent")

    for i, (qc, family) in enumerate(tagged_circuits):
        two_q_before = _count_2q(qc)
        depth_before = qc.depth()

        env = PassManagerEnv(circuits=[qc], max_steps=MAX_STEPS_PER_EPISODE)
        obs, _ = env.reset()
        t0 = time.time()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(int(action))
            done = terminated or truncated

        final_qc = dag_to_circuit(env._dag)
        two_q_after = _count_2q(final_qc)
        depth_after = final_qc.depth()
        result.results.append(CircuitResult(
            name=f"circuit_{i}",
            family=family,
            n_qubits=qc.num_qubits,
            two_q_before=two_q_before,
            two_q_after=two_q_after,
            depth_before=depth_before,
            depth_after=depth_after,
            steps_taken=env._step_count,
            elapsed_s=time.time() - t0,
        ))

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(tagged_circuits)}] done")

    return result


# ── Entry point ───────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Quetsal benchmark vs Qiskit baselines")
    p.add_argument(
        "--baseline-only", action="store_true",
        help="Only run Qiskit opt_level baselines (no agent required)",
    )
    p.add_argument(
        "--model", type=str, default=None,
        help="Path to trained Quetsal model zip (required without --baseline-only)",
    )
    p.add_argument("--n-circuits", type=int, default=50,
                   help="Number of circuits per family to benchmark")
    p.add_argument("--min-qubits", type=int, default=3)
    p.add_argument("--max-qubits", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--opt-levels", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument(
        "--families", type=str, nargs="+", default=None,
        metavar="FAMILY",
        help=(
            "Families to include. Use 'nonparam' as a shorthand for "
            "QV Clifford-SU4 Clifford-SU4-SU8 IQP RandomClifford. "
            f"Available: {', '.join(_ALL_FAMILY_KEYS)}"
        ),
    )
    p.add_argument("--save-csv", action="store_true",
                   help="Save per-circuit results to benchmarks/results/")
    return p.parse_args()


def main():
    args = _parse_args()

    # Resolve --families shorthand
    families: set[str] | None = None
    if args.families is not None:
        expanded: list[str] = []
        for f in args.families:
            if f.lower() == "nonparam":
                expanded.extend(_NONPARAM_FAMILIES)
            else:
                expanded.append(f)
        families = set(expanded)
        unknown = families - set(_ALL_FAMILY_KEYS)
        if unknown:
            print(f"[benchmark] ERROR: unknown families: {unknown}. Valid: {_ALL_FAMILY_KEYS}")
            return

    family_label = ", ".join(sorted(families)) if families else "all 8"
    print(f"[benchmark] Generating {args.n_circuits} circuits per family ({family_label})...")
    tagged_circuits = _generate_tagged_circuits(
        n_qubits_range=(args.min_qubits, args.max_qubits),
        count_per_family=args.n_circuits,
        seed=args.seed,
        families=families,
    )
    print(f"[benchmark] {len(tagged_circuits)} circuits ready")

    print(f"[benchmark] Running Qiskit baselines (opt_level={args.opt_levels})...")
    baseline_results = run_qiskit_baseline(
        tagged_circuits, opt_levels=args.opt_levels, seed=args.seed
    )

    if not args.baseline_only:
        if args.model is None:
            print("[benchmark] ERROR: --model required unless --baseline-only is set")
            return
        print(f"[benchmark] Running Quetsal agent: {args.model}")
        agent_result = run_agent_baseline(args.model, tagged_circuits)
        baseline_results.append(agent_result)

    print_summary(baseline_results)

    if args.save_csv:
        save_per_circuit_csv(baseline_results)


if __name__ == "__main__":
    main()
