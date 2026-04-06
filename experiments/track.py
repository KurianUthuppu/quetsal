# =============================================================================
# quetsal/experiments/track.py
# Experiment tracker — run manually after training to log one CSV row.
#
# Usage:
#   python -m quetsal.experiments.track \
#       --model runs/quetsal/best_model_<timestamp>/best_model.zip \
#       --train-log runs/quetsal/logs/train_<timestamp>.log \
#       --mode 1 \
#       --notes "ent_coef 0.01->0.05, n_steps 512->2048"
#
# train.py prints the exact invocation at the end of each run.
# Reads all lever values from constants.py and train.py defaults.
# Reads training metrics from the training log file (most recent .log in
# runs/quetsal/logs/) so they are captured after SB3 has fully flushed output.
# =============================================================================

from __future__ import annotations

import argparse
import csv
import re
from datetime import datetime
from pathlib import Path
from typing import Any

# ── Column definitions ────────────────────────────────────────────────────────

_COLUMNS = [
    # Meta
    "timestamp", "mode", "notes", "model_path",
    # Environment constants
    "DEPTH_PENALTY_WEIGHT", "TRUNCATION_PENALTY", "TERMINAL_BONUS", "STEP_PENALTY",
    "MAX_STEPS_PER_EPISODE", "MIN_STEPS_BEFORE_STOP",
    "MAX_NODES", "MAX_EDGES",
    # Training scale
    "n_steps", "n_epochs", "total_steps", "checkpoint_freq", "count_per_family",
    # PPO hyperparameters
    "lr", "gamma", "clip_range", "ent_coef", "gae_lambda",
    "vf_coef", "max_grad_norm",
    # GNN architecture
    "hidden_dim", "num_layers", "latent_dim",
    # Rollout metrics (final rollout block in training log)
    "rollout_ep_len_mean", "rollout_ep_rew_mean",
    "rollout_fps", "rollout_total_timesteps",
    # Eval metrics (final eval block in training log)
    "eval_mean_ep_length", "eval_mean_reward",
    # Train metrics (final train block in training log)
    "train_approx_kl", "train_clip_fraction", "train_clip_range",
    "train_entropy_loss", "train_explained_variance", "train_learning_rate",
    "train_loss", "train_n_updates", "train_policy_gradient_loss", "train_value_loss",
]

_LOG_FILE = "experiments/experiment_log.csv"
_RUNS_DIR = "runs/quetsal"
_LOGS_SUBDIR = "logs"


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Log a completed training run to CSV")
    p.add_argument("--model", type=str, required=True,
                   help="Path to best_model.zip (printed by train.py at end of run)")
    p.add_argument("--mode", type=int, default=1, choices=[0, 1])
    p.add_argument("--notes", type=str, default="")
    p.add_argument("--log-file", type=str, default=_LOG_FILE)
    p.add_argument(
        "--train-log", type=str, default=None,
        help="Path to training .log file. If omitted, uses most recent in runs/quetsal/",
    )
    return p.parse_args()


# ── Lever snapshot ────────────────────────────────────────────────────────────

def _collect_levers(mode: int) -> dict:
    from quetsal.src import constants as C
    import sys, importlib.util

    mode_args = C.TRAINING_MODE_ARGS[mode]

    row = {
        "DEPTH_PENALTY_WEIGHT":  C.DEPTH_PENALTY_WEIGHT,
        "TRUNCATION_PENALTY":    C.TRUNCATION_PENALTY,
        "TERMINAL_BONUS":        C.TERMINAL_BONUS,
        "STEP_PENALTY":          C.STEP_PENALTY,
        "MAX_STEPS_PER_EPISODE": C.MAX_STEPS_PER_EPISODE,
        "MIN_STEPS_BEFORE_STOP": C.MIN_STEPS_BEFORE_STOP,
        "MAX_NODES":             C.MAX_NODES,
        "MAX_EDGES":             C.MAX_EDGES,
        "n_steps":               mode_args["n_steps"],
        "n_epochs":              mode_args["n_epochs"],
        "total_steps":           mode_args["total_steps"],
        "checkpoint_freq":       mode_args["checkpoint_freq"],
        "count_per_family":      mode_args["count_per_family"],
    }

    # Read train.py defaults via its argparse
    train_path = Path(__file__).parent.parent / "training" / "train.py"
    orig_argv = sys.argv[:]
    sys.argv = ["train"]
    try:
        spec = importlib.util.spec_from_file_location("_train_tmp", train_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        ta = mod._parse_args()
        row.update({
            "lr":            ta.lr,
            "gamma":         ta.gamma,
            "clip_range":    ta.clip_range,
            "ent_coef":      ta.ent_coef,
            "gae_lambda":    ta.gae_lambda,
            "vf_coef":       ta.vf_coef,
            "max_grad_norm": ta.max_grad_norm,
            "hidden_dim":    ta.hidden_dim,
            "num_layers":    ta.num_layers,
            "latent_dim":    ta.latent_dim,
        })
    except Exception:
        pass
    finally:
        sys.argv = orig_argv

    return row


# ── Log file parser ───────────────────────────────────────────────────────────

def _find_latest_log() -> Path | None:
    logs = sorted(
        (Path(_RUNS_DIR) / _LOGS_SUBDIR).glob("train_*.log"),
        key=lambda p: p.stat().st_mtime,
    )
    return logs[-1] if logs else None


def _parse_log(log_path: Path) -> dict:
    """Extract the LAST occurrence of each metric from the training log."""
    text = log_path.read_text(encoding="utf-8", errors="replace")

    def _last(pattern: str) -> str:
        """Return the last match of a key|value SB3 table row."""
        matches = re.findall(pattern, text)
        return matches[-1].strip() if matches else ""

    return {
        # rollout block
        "rollout_ep_len_mean":  _last(r"\|\s+ep_len_mean\s+\|\s+([\d.]+)"),
        "rollout_ep_rew_mean":  _last(r"\|\s+ep_rew_mean\s+\|\s+([\d.eE+\-]+)"),
        "rollout_fps":          _last(r"\|\s+fps\s+\|\s+([\d.]+)"),
        "rollout_total_timesteps": _last(r"\|\s+total_timesteps\s+\|\s+([\d,]+)").replace(",", ""),
        # eval block
        "eval_mean_ep_length":  _last(r"\|\s+mean_ep_length\s+\|\s+([\d.]+)"),
        "eval_mean_reward":     _last(r"\|\s+mean_reward\s+\|\s+([\d.eE+\-]+)"),
        # train block
        "train_approx_kl":           _last(r"\|\s+approx_kl\s+\|\s+([\d.eE+\-]+)"),
        "train_clip_fraction":        _last(r"\|\s+clip_fraction\s+\|\s+([\d.eE+\-]+)"),
        "train_clip_range":           _last(r"\|\s+clip_range\s+\|\s+([\d.eE+\-]+)"),
        "train_entropy_loss":         _last(r"\|\s+entropy_loss\s+\|\s+(-?[\d.eE+\-]+)"),
        "train_explained_variance":   _last(r"\|\s+explained_variance\s+\|\s+(-?[\d.eE+\-]+)"),
        "train_learning_rate":        _last(r"\|\s+learning_rate\s+\|\s+([\d.eE+\-]+)"),
        "train_loss":                 _last(r"\|\s+loss\s+\|\s+(-?[\d.eE+\-]+)"),
        "train_n_updates":            _last(r"\|\s+n_updates\s+\|\s+([\d.]+)"),
        "train_policy_gradient_loss": _last(r"\|\s+policy_gradient_loss\s+\|\s+(-?[\d.eE+\-]+)"),
        "train_value_loss":           _last(r"\|\s+value_loss\s+\|\s+([\d.eE+\-]+)"),
    }


# ── CSV append ────────────────────────────────────────────────────────────────

def _append_csv(row: dict, log_file: str) -> None:
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not log_path.exists()
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_COLUMNS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = _parse_args()

    # Find training log
    log_path = Path(args.train_log) if args.train_log else _find_latest_log()
    if log_path is None or not log_path.exists():
        print(f"[track] ERROR: no training log found in {_RUNS_DIR}/")
        print("[track] Pass --train-log <path> explicitly.")
        return

    print(f"[track] Reading metrics from: {log_path}")
    metrics = _parse_log(log_path)

    print("[track] Collecting lever values...")
    levers = _collect_levers(args.mode)

    row: dict[str, Any] = {
        "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mode":       args.mode,
        "notes":      args.notes.strip(),
        "model_path": args.model,
    }
    row.update(levers)
    row.update(metrics)

    _append_csv(row, args.log_file)
    print(f"[track] Row appended -> {args.log_file}")

    # Echo key metrics to console
    print(f"\n  rollout  ep_len={metrics['rollout_ep_len_mean']}  ep_rew={metrics['rollout_ep_rew_mean']}")
    print(f"  eval     mean_reward={metrics['eval_mean_reward']}")
    print(f"  train    explained_variance={metrics['train_explained_variance']}  entropy={metrics['train_entropy_loss']}\n")


if __name__ == "__main__":
    main()
