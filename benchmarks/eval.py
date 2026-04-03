# =============================================================================
# quetsal/benchmarks/eval.py
# Baseline benchmark: Qiskit optimization_level=1/2/3 vs Quetsal agent.
#
# Usage (Qiskit baseline only, no agent required):
#   python -m quetsal.benchmarks.eval --baseline-only
#
# Usage (full comparison, requires a trained model):
#   python -m quetsal.benchmarks.eval --model runs/quetsal/quetsal_final.zip
#
# Metrics reported per circuit:
#   2q_before   : 2q gate count after layout/routing (before optimization)
#   2q_after    : 2q gate count after the optimizer under test
#   reduction   : (2q_before - 2q_after) / 2q_before  [0..1]
#   depth_before / depth_after
# =============================================================================

from __future__ import annotations

__all__ = [
    "CircuitResult",
    "BenchmarkResult",
    "run_qiskit_baseline",
    "run_agent_baseline",
    "print_summary",
]

import argparse
import time
from dataclasses import dataclass, field

from qiskit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit.transpiler import CouplingMap
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.transpiler import PassManager

from quetsal.src.constants import HERON_R2_BASIS, MAX_STEPS_PER_EPISODE, SKIP_GATES
from quetsal.src.environment.circuits import generate_training_circuits


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class CircuitResult:
    name: str
    n_qubits: int
    two_q_before: int
    two_q_after: int
    depth_before: int
    depth_after: int
    elapsed_s: float

    @property
    def reduction(self) -> float:
        if self.two_q_before == 0:
            return 0.0
        return (self.two_q_before - self.two_q_after) / self.two_q_before


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
        valid = [
            (r.depth_after - r.depth_before) / r.depth_before
            for r in self.results if r.depth_before > 0
        ]
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
    """Run Qiskit transpile at a given optimization_level. Returns (circuit, elapsed)."""
    cm = CouplingMap.from_line(qc.num_qubits)
    t0 = time.time()
    out = transpile(
        qc,
        basis_gates=HERON_R2_BASIS,
        coupling_map=cm,
        optimization_level=opt_level,
        seed_transpiler=seed,
    )
    return out, time.time() - t0


def _get_pre_opt_circuit(qc, seed: int = 42) -> "QuantumCircuit":
    """Transpile through layout/routing only (no optimization) — same as pass_env."""
    cm = CouplingMap.from_line(qc.num_qubits)
    pm = generate_preset_pass_manager(
        optimization_level=1,
        basis_gates=HERON_R2_BASIS,
        coupling_map=cm,
        seed_transpiler=seed,
    )
    pm.optimization = PassManager()
    pm.scheduling = PassManager()
    return pm.run(qc)


# ── Baseline runner ───────────────────────────────────────────────────────────

def run_qiskit_baseline(
    circuits,
    opt_levels: list[int] = (1, 2, 3),
    seed: int = 42,
) -> list[BenchmarkResult]:
    """Benchmark Qiskit optimization_level=1/2/3 on a list of QuantumCircuits.

    Each circuit is first transpiled through layout+routing only (the same
    pre-optimization state the RL agent receives), then optimized at each
    opt_level from that same starting point.
    """
    benchmarks = {lvl: BenchmarkResult(label=f"opt_level={lvl}") for lvl in opt_levels}

    for i, qc in enumerate(circuits):
        # Get the unoptimized post-routing circuit (agent's starting point)
        pre = _get_pre_opt_circuit(qc, seed=seed)
        two_q_before = _count_2q(pre)
        depth_before = pre.depth()

        for lvl in opt_levels:
            optimized, elapsed = _run_opt_level(pre, opt_level=lvl, seed=seed)
            two_q_after = _count_2q(optimized)
            depth_after = optimized.depth()
            benchmarks[lvl].results.append(CircuitResult(
                name=f"circuit_{i}",
                n_qubits=qc.num_qubits,
                two_q_before=two_q_before,
                two_q_after=two_q_after,
                depth_before=depth_before,
                depth_after=depth_after,
                elapsed_s=elapsed,
            ))

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(circuits)}] done")

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


# ── Agent runner ─────────────────────────────────────────────────────────────

def run_agent_baseline(model_path: str, circuits, seed: int = 42) -> BenchmarkResult:
    """Evaluate a trained Quetsal agent on circuits."""
    from quetsal.src.agent.ppo_agent import load_agent
    from quetsal.src.environment.pass_env import PassManagerEnv
    from qiskit.converters import dag_to_circuit

    # load_agent needs an env to bind observation/action spaces.
    # Use a throwaway env built from the first circuit.
    pre0 = _get_pre_opt_circuit(circuits[0], seed=seed)
    dummy_env = PassManagerEnv(circuits=[pre0], max_steps=MAX_STEPS_PER_EPISODE)
    model = load_agent(model_path, env=dummy_env)
    result = BenchmarkResult(label="Quetsal agent")

    for i, qc in enumerate(circuits):
        pre = _get_pre_opt_circuit(qc, seed=seed)
        two_q_before = _count_2q(pre)
        depth_before = pre.depth()

        # Run the agent on this single circuit
        env = PassManagerEnv(circuits=[pre], max_steps=MAX_STEPS_PER_EPISODE)
        obs, _ = env.reset()
        t0 = time.time()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(int(action))
            done = terminated or truncated

        # Read final circuit from the env's DAG
        final_qc = dag_to_circuit(env._dag)
        two_q_after = _count_2q(final_qc)
        depth_after = final_qc.depth()
        result.results.append(CircuitResult(
            name=f"circuit_{i}",
            n_qubits=qc.num_qubits,
            two_q_before=two_q_before,
            two_q_after=two_q_after,
            depth_before=depth_before,
            depth_after=depth_after,
            elapsed_s=time.time() - t0,
        ))

        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(circuits)}] done")

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
    p.add_argument("--max-qubits", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--opt-levels", type=int, nargs="+", default=[1, 2, 3])
    return p.parse_args()


def main():
    args = _parse_args()

    print(f"[benchmark] Generating {args.n_circuits} circuits per family...")
    circuits = generate_training_circuits(
        n_qubits_range=(args.min_qubits, args.max_qubits),
        count_per_family=args.n_circuits,
        seed=args.seed,
        basis_gates=HERON_R2_BASIS,
    )
    print(f"[benchmark] {len(circuits)} circuits ready")

    print(f"[benchmark] Running Qiskit baselines (opt_level={args.opt_levels})...")
    baseline_results = run_qiskit_baseline(
        circuits, opt_levels=args.opt_levels, seed=args.seed
    )

    if not args.baseline_only:
        if args.model is None:
            print("[benchmark] ERROR: --model required unless --baseline-only is set")
            return
        print(f"[benchmark] Running Quetsal agent: {args.model}")
        agent_result = run_agent_baseline(args.model, circuits)
        baseline_results.append(agent_result)

    print_summary(baseline_results)


if __name__ == "__main__":
    main()
