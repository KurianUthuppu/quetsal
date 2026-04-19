# =============================================================================
# quetsal/training/circuit_cache.py
# Master pool: generate once, sample forever.
#
# Workflow
# --------
# 1. First run (pool file does not exist):
#      build_master_pool() generates max_per_family circuits for each of the
#      7 families, runs them through the full transpile+filter pipeline, and
#      saves the result as a dict[family -> list[QuantumCircuit]] to disk.
#
# 2. Every subsequent run (pool file exists):
#      load_master_pool() deserialises the dict in seconds.
#      sample_pool() / sample_weighted_pool() randomly draw the required
#      count from the stored list — instantaneous, reproducible via seed.
#
# This means changing --count-per-family, --n-steps, or any other training
# hyperparameter never triggers regeneration; only changing pool-level params
# (qubit range, seed, pool size) invalidates the pool.
#
# Pool structure: dict[str, list[QuantumCircuit]]
#   {"qv": [...], "qaoa": [...], "clifford_su4_su8": [...], ...}
# Each circuit has metadata["family"] already set by the generator.
# =============================================================================

from __future__ import annotations

__all__ = [
    "build_master_pool",
    "extend_master_pool",
    "save_master_pool",
    "load_master_pool",
    "sample_pool",
    "sample_weighted_pool",
]

import pickle
import time
from pathlib import Path

import numpy as np

_ALL_FAMILIES = [
    "qv",
    "qaoa",
    "clifford_su4_su8",
    "clifford_su4",
    "iqp",
    "efficient_su2",
    "real_amplitudes",
]


# ── Build ─────────────────────────────────────────────────────────────────────


def build_master_pool(
    max_per_family: int,
    min_qubits: int,
    max_qubits: int,
    seed: int,
    basis_gates: list[str],
) -> dict[str, list]:
    """Generate and return a master circuit pool for all 7 families.

    Calls each family's generator with ``count=max_per_family``.  Circuits
    that fail the node-count or 2q-gate filter are silently dropped, so the
    actual count per family may be slightly below max_per_family.

    Prints per-family progress — this is the slow one-time call (≈20-30 min
    at 1000/family on CPU).

    Parameters
    ----------
    max_per_family : circuits to request per family (some may be filtered).
    min_qubits     : minimum qubit count for generated circuits.
    max_qubits     : maximum qubit count for generated circuits.
    seed           : base RNG seed; each family receives seed + family_index.
    basis_gates    : target basis gates (e.g. HERON_R2_BASIS).

    Returns
    -------
    dict mapping family name -> list of transpiled QuantumCircuit objects.
    """
    from quetsal.src.environment.circuits import (
        generate_clifford_su4_circuits,
        generate_clifford_su4_su8_circuits,
        generate_efficient_su2_circuits,
        generate_iqp_circuits,
        generate_qaoa_circuits,
        generate_qv_circuits,
        generate_real_amplitudes_circuits,
    )

    _generators = {
        "qv":                  generate_qv_circuits,
        "qaoa":                generate_qaoa_circuits,
        "clifford_su4_su8":    generate_clifford_su4_su8_circuits,
        "clifford_su4":        generate_clifford_su4_circuits,
        "iqp":                 generate_iqp_circuits,
        "efficient_su2":       generate_efficient_su2_circuits,
        "real_amplitudes":     generate_real_amplitudes_circuits,
    }

    pool: dict[str, list] = {}
    t_start = time.time()
    print(f"[master_pool] Building pool: {max_per_family}/family requested, "
          f"qubits={min_qubits}-{max_qubits}, seed={seed}")

    for i, (family, gen) in enumerate(_generators.items()):
        t0 = time.time()
        print(f"[master_pool] ({i+1}/7) {family}...", end=" ", flush=True)
        circuits = gen(
            n_qubits_range=(min_qubits, max_qubits),
            count=max_per_family,
            seed=seed + i,
            basis_gates=basis_gates,
        )
        pool[family] = circuits
        print(f"{len(circuits)} circuits in {time.time()-t0:.0f}s")

    total = sum(len(v) for v in pool.values())
    print(f"[master_pool] Done: {total:,} circuits total in {time.time()-t_start:.0f}s")
    return pool


# ── Extend ───────────────────────────────────────────────────────────────────


def extend_master_pool(
    pool: dict[str, list],
    target_per_family: int,
    min_qubits: int,
    max_qubits: int,
    base_seed: int,
    basis_gates: list[str],
) -> tuple[dict[str, list], bool]:
    """Top up any family that has fewer than target_per_family circuits.

    Uses seed = base_seed + 10_000 + family_index for the extra generation
    batch so there is no overlap with the circuits already in the pool.
    Generates 3× the shortfall to account for the node-count filter, then
    keeps only as many as needed.

    Parameters
    ----------
    pool             : existing master pool dict (modified in-place).
    target_per_family: minimum circuits per family after extension.
    min_qubits       : qubit range lower bound.
    max_qubits       : qubit range upper bound.
    base_seed        : original build seed (offset applied internally).
    basis_gates      : target basis gates.

    Returns
    -------
    (pool, was_extended) — pool is the updated dict;
    was_extended is True if any family was topped up.
    """
    from quetsal.src.environment.circuits import (
        generate_clifford_su4_circuits,
        generate_clifford_su4_su8_circuits,
        generate_efficient_su2_circuits,
        generate_iqp_circuits,
        generate_qaoa_circuits,
        generate_qv_circuits,
        generate_real_amplitudes_circuits,
    )

    _generators = {
        "qv":               generate_qv_circuits,
        "qaoa":             generate_qaoa_circuits,
        "clifford_su4_su8": generate_clifford_su4_su8_circuits,
        "clifford_su4":     generate_clifford_su4_circuits,
        "iqp":              generate_iqp_circuits,
        "efficient_su2":    generate_efficient_su2_circuits,
        "real_amplitudes":  generate_real_amplitudes_circuits,
    }

    was_extended = False
    t_start = time.time()

    for i, (family, gen) in enumerate(_generators.items()):
        current = len(pool.get(family, []))
        if current >= target_per_family:
            continue

        needed = target_per_family - current
        ext_seed = base_seed + 10_000 + i
        print(
            f"[master_pool] Extending {family}: {current} -> {target_per_family} "
            f"(need {needed}, seed={ext_seed})...",
            end=" ", flush=True,
        )
        t0 = time.time()
        # Generate 3× budget to account for filtering; trim to what's needed
        candidates = gen(
            n_qubits_range=(min_qubits, max_qubits),
            count=needed * 3,
            seed=ext_seed,
            basis_gates=basis_gates,
        )
        added = candidates[:needed]
        pool[family] = pool.get(family, []) + added
        actual = len(pool[family])
        print(f"got {len(added)}, total now {actual} ({time.time()-t0:.0f}s)")
        if len(added) < needed:
            print(
                f"[master_pool] WARNING: {family} still has only {actual} circuits "
                f"(target {target_per_family}). Increase --master-pool-size or "
                f"relax qubit-range filter."
            )
        was_extended = True

    if was_extended:
        total = sum(len(v) for v in pool.values())
        print(f"[master_pool] Extension done: {total:,} circuits total in {time.time()-t_start:.0f}s")

    return pool, was_extended


# ── Persist ───────────────────────────────────────────────────────────────────


def save_master_pool(pool: dict[str, list], path: str | Path) -> None:
    """Pickle the master pool dict to disk.

    Parameters
    ----------
    pool : dict[family -> list[QuantumCircuit]] returned by build_master_pool.
    path : file path (e.g. ``runs/circuits/master.pkl``).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(pool, f, protocol=5)
    total = sum(len(v) for v in pool.values())
    print(f"[master_pool] Saved {total:,} circuits -> {path}")


def load_master_pool(path: str | Path) -> dict[str, list] | None:
    """Load a master pool from disk, or return None if the file does not exist.

    Parameters
    ----------
    path : path previously passed to save_master_pool().

    Returns
    -------
    dict[family -> list[QuantumCircuit]], or None if the file is missing.
    """
    path = Path(path)
    if not path.exists():
        return None
    with open(path, "rb") as f:
        pool = pickle.load(f)
    total = sum(len(v) for v in pool.values())
    per = {f: len(v) for f, v in pool.items()}
    print(f"[master_pool] Loaded {total:,} circuits <- {path}")
    print(f"[master_pool]   per family: { {f: per[f] for f in _ALL_FAMILIES if f in per} }")
    return pool


# ── Sample ────────────────────────────────────────────────────────────────────


def sample_pool(
    pool: dict[str, list],
    families: list[str],
    count_per_family: int,
    *,
    rng_seed: int | None = None,
) -> list:
    """Draw count_per_family circuits from each family in the master pool.

    Samples without replacement (capped at available count if the pool is
    smaller than requested).  The combined list is shuffled.

    Parameters
    ----------
    pool             : master pool dict from load_master_pool().
    families         : which families to sample from (subset of pool keys).
    count_per_family : how many circuits per family to draw.
    rng_seed         : seed for the sampling RNG (None = non-deterministic).

    Returns
    -------
    Shuffled list of QuantumCircuit objects.
    """
    rng = np.random.default_rng(rng_seed)
    result = []
    for family in families:
        available = pool.get(family, [])
        if not available:
            raise KeyError(
                f"Family '{family}' not found in master pool. "
                f"Available families: {list(pool.keys())}"
            )
        n = min(count_per_family, len(available))
        if n < count_per_family:
            print(
                f"[master_pool] WARNING: {family} has only {len(available)} circuits "
                f"(<{count_per_family} requested) — using all available."
            )
        indices = rng.choice(len(available), n, replace=False)
        result.extend(available[i] for i in indices)

    rng.shuffle(result)
    return result


def sample_weighted_pool(
    pool: dict[str, list],
    families: list[str],
    family_weights: dict[str, float],
    total_count: int,
    *,
    rng_seed: int | None = None,
) -> list:
    """Draw circuits from the master pool with explicit per-family weights.

    Allocates ``total_count`` circuits proportionally to ``family_weights``.
    Mirrors the logic of generate_weighted_circuits() but reads from the pool
    instead of generating.  Used by CurriculumController for stage 2 blending.

    Parameters
    ----------
    pool           : master pool dict from load_master_pool().
    families       : ordered list of family names to include.
    family_weights : dict[family -> weight] (need not sum to 1; normalised internally).
    total_count    : total circuits to draw (distributed proportionally).
    rng_seed       : seed for the sampling RNG.

    Returns
    -------
    Shuffled list of QuantumCircuit objects.
    """
    rng = np.random.default_rng(rng_seed)
    total_w = sum(family_weights.get(f, 0.0) for f in families)
    if total_w == 0:
        raise ValueError(f"All family weights are zero for families={families}")

    result = []
    for family in families:
        w = family_weights.get(family, 0.0)
        count = max(1, round(w / total_w * total_count))
        available = pool.get(family, [])
        if not available:
            raise KeyError(
                f"Family '{family}' not found in master pool. "
                f"Available families: {list(pool.keys())}"
            )
        n = min(count, len(available))
        if n < count:
            print(
                f"[master_pool] WARNING: {family} has only {len(available)} circuits "
                f"(<{count} requested) — using all available."
            )
        indices = rng.choice(len(available), n, replace=False)
        result.extend(available[i] for i in indices)

    rng.shuffle(result)
    return result
