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
from quetsal.training.callbacks import LoggingEvalCallback, PassLogWrapper, PassLoggerCallback
from quetsal.src.constants import (
    HERON_R2_BASIS,
    MAX_STEPS_PER_EPISODE,
    TRAINING_MODE,
    TRAINING_MODE_ARGS,
)
from quetsal.src.environment.circuits import generate_training_circuits
from quetsal.src.environment.pass_env import PassManagerEnv


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
        help="Rollout length (batch_size is set equal to this)",
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
        "--notes",
        type=str,
        default="",
        help="Free-text annotation logged to experiment_log.csv",
    )

    args = p.parse_args()

    # Apply mode defaults for args that were not explicitly set (still None)
    mode_defaults = TRAINING_MODE_ARGS[args.mode]
    for key, val in mode_defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, val)

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

    # ── 1. Generate circuits ──────────────────────────────────────────────────
    print(
        f"[quetsal] Generating training circuits ({args.count_per_family} per family)..."
    )
    t0 = time.time()
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
    model = make_ppo_agent(
        env=env,
        n_steps=args.n_steps,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        learning_rate=args.lr,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
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

    # Eval env uses the same circuit pool but a fixed seed subset
    # so evaluation episodes are reproducible across checkpoints
    _EVAL_SEED_OFFSET = 999  # ensures eval circuits differ from the training pool
    _eval_count = max(2, int(args.count_per_family * 0.20))  # 20% of training, min 2/family
    eval_circuits = generate_training_circuits(
        n_qubits_range=(args.min_qubits, args.max_qubits),
        count_per_family=_eval_count,
        seed=args.circuit_seed + _EVAL_SEED_OFFSET,
        basis_gates=HERON_R2_BASIS,
    )
    print(f"[quetsal] Eval pool: {len(eval_circuits)} circuits ({_eval_count}/family requested, {len(eval_circuits)//7} avg after filtering)")
    eval_env = Monitor(
        PassLogWrapper(
            PassManagerEnv(circuits=eval_circuits, max_steps=MAX_STEPS_PER_EPISODE)
        )
    )

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

    callbacks = CallbackList([checkpoint_cb, eval_cb, pass_logger_cb])

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
