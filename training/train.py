# =============================================================================
# quetsal/training/train.py
# Training entry point for the Quetsal PPO agent.
#
# Usage:
#   python -m quetsal.training.train                         # uses TRAINING_MODE from constants.py
#   python -m quetsal.training.train --mode 0               # quick smoke-test (~2-3 min)
#   python -m quetsal.training.train --mode 1               # full training run
#   python -m quetsal.training.train --total-steps 200000   # override individual arg
#
# Flow:
#   1. Generate training circuits (all 7 families)
#   2. Instantiate PassManagerEnv
#   3. Create PPO agent with QuetsalGNNPolicy
#   4. Run .learn() with checkpoint + logging callbacks
#   5. Save final model
# =============================================================================

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path


# ── Tee: mirror stdout to a log file ─────────────────────────────────────────


class _Tee:
    """Write to both the real stdout and a log file simultaneously.

    SB3 uses print() for its rollout tables and progress output, so we
    capture everything by replacing sys.stdout rather than using logging.
    """

    def __init__(self, log_path: Path) -> None:
        self._stdout = sys.stdout
        self._file = open(log_path, "w", encoding="utf-8", buffering=1)

    def write(self, data: str) -> int:
        self._stdout.write(data)
        self._file.write(data)
        return len(data)

    def flush(self) -> None:
        self._stdout.flush()
        self._file.flush()

    def close(self) -> None:
        sys.stdout = self._stdout
        self._file.close()

    # Proxy any other attribute lookups to the real stdout
    def __getattr__(self, name: str):
        return getattr(self._stdout, name)


from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
)
from stable_baselines3.common.monitor import Monitor

from quetsal.src.agent.ppo_agent import make_ppo_agent, save_agent
from quetsal.training.callbacks import (
    CurriculumCallback,
    EarlyStoppingCallback,
    LoggingEvalCallback,
    PassLogWrapper,
    PassLoggerCallback,
)
from quetsal.training.curriculum import CurriculumController
from quetsal.src.constants import (
    CURRICULUM_STAGES,
    HERON_R2_BASIS,
    MAX_STEPS_PER_EPISODE,
    TRAINING_MODE,
    TRAINING_MODE_ARGS,
)
from quetsal.src.environment.circuits import generate_training_circuits
from quetsal.src.environment.pass_env import PassManagerEnv
from quetsal.training.circuit_cache import (
    build_master_pool,
    extend_master_pool,
    load_master_pool,
    sample_pool,
    sample_weighted_pool,
    save_master_pool,
)


# ── CLI args ──────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Quetsal PPO agent")

    # Mode — resolves defaults for training-scale args before other flags
    p.add_argument(
        "--mode",
        type=int,
        default=TRAINING_MODE,
        choices=[0, 1],
        help="0=quick smoke-test (~2-3 min), 1=full training (default from constants.py)",
    )

    # Data
    p.add_argument(
        "--count-per-family",
        type=int,
        default=None,
        help="Training circuits per family (7 families total)",
    )
    p.add_argument("--min-qubits", type=int, default=3)
    p.add_argument("--max-qubits", type=int, default=10)
    p.add_argument("--circuit-seed", type=int, default=42)

    # Training
    p.add_argument(
        "--total-steps", type=int, default=None, help="Total env steps for PPO.learn()"
    )
    p.add_argument(
        "--n-steps",
        type=int,
        default=None,
        help="Rollout length per update",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="PPO minibatch size. Defaults to n_steps (single minibatch). "
             "Set smaller (e.g. n_steps//2) for multiple minibatches per epoch.",
    )
    p.add_argument("--n-epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--clip-range", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.05)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--vf-coef", type=float, default=0.75)
    p.add_argument("--max-grad-norm", type=float, default=0.5)

    # GNN
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--num-layers", type=int, default=3)
    p.add_argument("--latent-dim", type=int, default=64)

    # I/O
    p.add_argument(
        "--checkpoint-dir",
        type=str,
        default="runs/quetsal",
        help="Directory for checkpoints and final model",
    )
    p.add_argument(
        "--checkpoint-freq",
        type=int,
        default=None,
        help="Save checkpoint every N env steps",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--verbose", type=int, default=1)
    p.add_argument(
        "--curriculum",
        action="store_true",
        default=False,
        help="Enable staged curriculum learning (stage 1→2→3 with promotion gates)",
    )
    p.add_argument(
        "--max-stage",
        type=int,
        default=3,
        choices=[1, 2, 3],
        help="Highest curriculum stage to enter (default 3 = all stages). "
             "Use --max-stage 1 to train on non-parametric families only.",
    )
    p.add_argument(
        "--early-stopping-patience",
        type=int,
        default=0,
        help="Stop training if eval_mean_reward does not improve for this many "
             "consecutive eval checkpoints. 0 = disabled (default).",
    )
    p.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=0.005,
        help="Minimum reward improvement to count as progress for early stopping "
             "(default 0.005).",
    )
    p.add_argument(
        "--notes",
        type=str,
        default="",
        help="Free-text annotation logged to experiment_log.csv",
    )
    p.add_argument(
        "--master-pool-path",
        type=str,
        default=None,
        help="Path to the master circuit pool .pkl file. "
             "If the file exists, circuits are sampled from it (no generation). "
             "If the file does not exist, the pool is generated once and saved. "
             "All training runs with the same path reuse the same pool regardless "
             "of --count-per-family or other training hyperparameters. "
             "Example: --master-pool-path runs/circuits/master.pkl",
    )
    p.add_argument(
        "--master-pool-size",
        type=int,
        default=1000,
        help="Circuits per family to generate when building the master pool "
             "(only used if --master-pool-path is set and the file does not yet "
             "exist). Default 1000. Some families may produce fewer after filtering.",
    )
    p.add_argument(
        "--build-pool-only",
        action="store_true",
        default=False,
        help="Build (or verify) the master pool at --master-pool-path and exit "
             "without training. Requires --master-pool-path.",
    )

    args = p.parse_args()

    if args.build_pool_only and not args.master_pool_path:
        p.error("--build-pool-only requires --master-pool-path")

    # Apply mode defaults for args that were not explicitly set (still None).
    # Skip non-scalar entries (e.g. curriculum_smoke) — those are consumed
    # directly from TRAINING_MODE_ARGS, not surfaced as argparse attributes.
    mode_defaults = TRAINING_MODE_ARGS[args.mode]
    for key, val in mode_defaults.items():
        if isinstance(val, dict):
            continue  # not a CLI arg
        attr = key.replace("-", "_")
        if getattr(args, attr, None) is None:
            setattr(args, attr, val)

    return args


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    args = _parse_args()
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── 0. Set up logging (stdout + log file) ─────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logs_dir = ckpt_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / f"train_{timestamp}.log"
    tee = _Tee(log_path)
    sys.stdout = tee
    print(f"[quetsal] Log -> {log_path}")

    # Emit all resolved hyperparameters as a single JSON line so that
    # track.py can read actual values instead of reconstructing from defaults.
    _hparams = {
        "n_steps":        args.n_steps,
        "batch_size":     args.batch_size if args.batch_size is not None else args.n_steps,
        "n_epochs":       args.n_epochs,
        "total_steps":    args.total_steps,
        "checkpoint_freq": args.checkpoint_freq,
        "count_per_family": args.count_per_family,
        "lr":             args.lr,
        "gamma":          args.gamma,
        "clip_range":     args.clip_range,
        "ent_coef":       args.ent_coef,
        "gae_lambda":     args.gae_lambda,
        "vf_coef":        args.vf_coef,
        "max_grad_norm":  args.max_grad_norm,
        "hidden_dim":     args.hidden_dim,
        "num_layers":     args.num_layers,
        "latent_dim":     args.latent_dim,
    }
    print(f"[quetsal] hparams: {json.dumps(_hparams)}")

    # ── 1. Load or build master pool, then sample training circuits ───────────
    _master_pool: dict | None = None

    if args.master_pool_path:
        _pool_path = Path(args.master_pool_path)
        _master_pool = load_master_pool(_pool_path)
        if _master_pool is None:
            print(
                f"[quetsal] Master pool not found at {_pool_path}. "
                f"Building with {args.master_pool_size}/family "
                f"(qubits={args.min_qubits}-{args.max_qubits}, seed={args.circuit_seed})..."
            )
            _master_pool = build_master_pool(
                max_per_family=args.master_pool_size,
                min_qubits=args.min_qubits,
                max_qubits=args.max_qubits,
                seed=args.circuit_seed,
                basis_gates=HERON_R2_BASIS,
            )
            save_master_pool(_master_pool, _pool_path)
        else:
            # Pool loaded — extend any family below the target size
            _min_count = min(len(v) for v in _master_pool.values())
            if _min_count < args.master_pool_size:
                print(
                    f"[quetsal] Pool has min {_min_count}/family "
                    f"(target {args.master_pool_size}). Extending..."
                )
                _master_pool, _extended = extend_master_pool(
                    pool=_master_pool,
                    target_per_family=args.master_pool_size,
                    min_qubits=args.min_qubits,
                    max_qubits=args.max_qubits,
                    base_seed=args.circuit_seed,
                    basis_gates=HERON_R2_BASIS,
                )
                if _extended:
                    save_master_pool(_master_pool, _pool_path)

        if args.build_pool_only:
            total = sum(len(v) for v in _master_pool.values())
            print(f"[quetsal] Pool ready ({total:,} circuits). Exiting (--build-pool-only).")
            tee.close()
            return

    # Sample training circuits from master pool (if available) or generate directly
    _s1 = CURRICULUM_STAGES[1]
    _s1_total = args.count_per_family * len(_s1["families"])  # used for eval sizing below

    if _master_pool is not None:
        if args.curriculum:
            circuits = sample_weighted_pool(
                _master_pool,
                families=_s1["families"],
                family_weights=_s1["family_weights"],
                total_count=_s1_total,
                rng_seed=args.circuit_seed,
            )
        else:
            _ALL_FAMILIES = list(_master_pool.keys())
            circuits = sample_pool(
                _master_pool,
                families=_ALL_FAMILIES,
                count_per_family=args.count_per_family,
                rng_seed=args.circuit_seed,
            )
        print(f"[quetsal] {len(circuits)} training circuits sampled from master pool")
    else:
        print(
            f"[quetsal] Generating training circuits "
            f"({'stage 1 curriculum — non-parametric only' if args.curriculum else f'{args.count_per_family} per family'})..."
        )
        t0 = time.time()
        if args.curriculum:
            from quetsal.src.environment.circuits import generate_weighted_circuits
            circuits = generate_weighted_circuits(
                families=_s1["families"],
                family_weights=_s1["family_weights"],
                total_count=_s1_total,
                n_qubits_range=(args.min_qubits, args.max_qubits),
                seed=args.circuit_seed,
                basis_gates=HERON_R2_BASIS,
            )
        else:
            circuits = generate_training_circuits(
                n_qubits_range=(args.min_qubits, args.max_qubits),
                count_per_family=args.count_per_family,
                seed=args.circuit_seed,
                basis_gates=HERON_R2_BASIS,
            )
        print(f"[quetsal] {len(circuits)} circuits ready in {time.time() - t0:.1f}s")

    # ── 2. Instantiate environment ────────────────────────────────────────────
    env = PassManagerEnv(
        circuits=circuits,
        max_steps=MAX_STEPS_PER_EPISODE,
    )

    # ── 3. Build PPO agent ────────────────────────────────────────────────────
    # Curriculum stage 1 overrides ent_coef to the stage-specific value
    _ent_coef = CURRICULUM_STAGES[1]["ent_coef"] if args.curriculum else args.ent_coef

    model = make_ppo_agent(
        env=env,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        learning_rate=args.lr,
        clip_range=args.clip_range,
        ent_coef=_ent_coef,
        gae_lambda=args.gae_lambda,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        latent_dim=args.latent_dim,
        seed=args.seed,
        verbose=args.verbose,
        device=args.device,
    )
    print(f"[quetsal] Policy: {model.policy}")
    _total = sum(p.numel() for p in model.policy.parameters())
    _unique = sum(p.numel() for p in set(model.policy.parameters()))
    _pi_is_shared = (
        model.policy.pi_features_extractor is model.policy.features_extractor
    )
    _vf_is_shared = (
        model.policy.vf_features_extractor is model.policy.features_extractor
    )
    print(
        f"[quetsal] Policy params — total: {_total:,} | unique: {_unique:,} | shared: {_pi_is_shared and _vf_is_shared}"
    )
    print(f"[quetsal] Trunk shared — pi: {_pi_is_shared}, vf: {_vf_is_shared}")

    # ── 4. Callbacks ──────────────────────────────────────────────────────────
    checkpoint_cb = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=str(ckpt_dir / "checkpoints"),
        name_prefix="quetsal",
        verbose=1,
    )

    # Eval env — uses stage 1 families when curriculum, else full pool
    _EVAL_SEED_OFFSET = 999
    _eval_count = max(2, int(args.count_per_family * 0.20))

    if _master_pool is not None:
        if args.curriculum:
            _s1_eval_total = max(len(_s1["families"]) * 2, int(_s1_total * 0.20))
            eval_circuits = sample_weighted_pool(
                _master_pool,
                families=_s1["families"],
                family_weights=_s1["family_weights"],
                total_count=_s1_eval_total,
                rng_seed=args.circuit_seed + _EVAL_SEED_OFFSET,
            )
        else:
            _ALL_FAMILIES = list(_master_pool.keys())
            eval_circuits = sample_pool(
                _master_pool,
                families=_ALL_FAMILIES,
                count_per_family=_eval_count,
                rng_seed=args.circuit_seed + _EVAL_SEED_OFFSET,
            )
    else:
        if args.curriculum:
            from quetsal.src.environment.circuits import generate_weighted_circuits
            _s1_eval_total = max(len(_s1["families"]) * 2, int(_s1_total * 0.20))
            eval_circuits = generate_weighted_circuits(
                families=_s1["families"],
                family_weights=_s1["family_weights"],
                total_count=_s1_eval_total,
                n_qubits_range=(args.min_qubits, args.max_qubits),
                seed=args.circuit_seed + _EVAL_SEED_OFFSET,
                basis_gates=HERON_R2_BASIS,
            )
        else:
            eval_circuits = generate_training_circuits(
                n_qubits_range=(args.min_qubits, args.max_qubits),
                count_per_family=_eval_count,
                seed=args.circuit_seed + _EVAL_SEED_OFFSET,
                basis_gates=HERON_R2_BASIS,
            )
    print(f"[quetsal] Eval pool: {len(eval_circuits)} circuits ({_eval_count}/family requested, {len(eval_circuits)//7} avg after filtering)")

    _inner_eval_env = PassManagerEnv(circuits=eval_circuits, max_steps=MAX_STEPS_PER_EPISODE)
    _pass_log_wrapper = PassLogWrapper(_inner_eval_env)
    eval_env = Monitor(_pass_log_wrapper)

    _pass_log_path = Path("experiments") / f"pass_log_{timestamp}.csv"

    best_model_dir = ckpt_dir / "best_model" / timestamp
    eval_cb = LoggingEvalCallback(
        eval_env=eval_env,
        best_model_save_path=str(best_model_dir),
        log_path=str(ckpt_dir / "logs" / f"eval_{timestamp}"),
        eval_freq=args.checkpoint_freq,
        n_eval_episodes=len(eval_circuits),
        deterministic=True,
        verbose=1,
        pass_log_path=_pass_log_path,
        pass_log_verbose=args.verbose,
    )

    pass_logger_cb = PassLoggerCallback(
        log_freq=args.checkpoint_freq,
        save_path=str(_pass_log_path),
        verbose=args.verbose,
    )

    cb_list = [checkpoint_cb, eval_cb, pass_logger_cb]

    if args.early_stopping_patience > 0:
        _es_curriculum_cb = None  # filled in below if --curriculum is also set
        early_stopping_cb = EarlyStoppingCallback(
            eval_cb=eval_cb,
            eval_freq=args.checkpoint_freq,
            patience=args.early_stopping_patience,
            min_delta=args.early_stopping_min_delta,
            curriculum_cb=_es_curriculum_cb,  # updated below if curriculum enabled
            verbose=args.verbose,
        )
        cb_list.append(early_stopping_cb)
        print(
            f"[quetsal] Early stopping enabled — patience={args.early_stopping_patience} evals, "
            f"min_delta={args.early_stopping_min_delta}"
        )
    else:
        early_stopping_cb = None

    if args.curriculum:
        curriculum_ctrl = CurriculumController(
            model=model,
            train_env=env,
            eval_env_inner=_inner_eval_env,
            n_qubits_range=(args.min_qubits, args.max_qubits),
            count_per_family=args.count_per_family,
            circuit_seed=args.circuit_seed,
            ckpt_dir=ckpt_dir,
            verbose=args.verbose,
            smoke=(args.mode == 0),  # lower promotion gates for quick smoke-test
            max_stage=args.max_stage,
            master_pool=_master_pool,  # None if --master-pool-path not set
        )
        curriculum_cb = CurriculumCallback(
            curriculum=curriculum_ctrl,
            eval_cb=eval_cb,
            eval_log_wrapper=_pass_log_wrapper,
            eval_freq=args.checkpoint_freq,
            verbose=args.verbose,
        )
        cb_list.append(curriculum_cb)
        _stage_cap = f" (capped at Stage {args.max_stage})" if args.max_stage < 3 else ""
        print(f"[quetsal] Curriculum learning enabled — starting at Stage 1{_stage_cap}")

        # Link curriculum_cb into early stopping so the counter resets on stage promotion
        if early_stopping_cb is not None:
            early_stopping_cb.curriculum_cb = curriculum_cb

    callbacks = CallbackList(cb_list)

    # ── 5. Train ──────────────────────────────────────────────────────────────
    print(f"[quetsal] Training for {args.total_steps:,} steps -> {ckpt_dir}")
    t1 = time.time()
    model.learn(
        total_timesteps=args.total_steps,
        callback=callbacks,
        progress_bar=True,
    )
    elapsed = time.time() - t1
    print(f"[quetsal] Training complete in {elapsed:.1f}s")
    final_path = ckpt_dir / f"final_{timestamp}"
    save_agent(model, final_path)
    print(f"[quetsal] Final model -> {final_path}.zip")
    print(f"[quetsal] Best model  -> {best_model_dir / 'best_model.zip'}")

    print(
        f"\n[quetsal] To log this run to experiment_log.csv, run:\n"
        f"  python -m quetsal.experiments.track "
        f"--model {best_model_dir / 'best_model.zip'} "
        f"--train-log {log_path} "
        f'--mode {args.mode} --notes "<your notes here>"'
    )

    tee.close()


if __name__ == "__main__":
    main()
