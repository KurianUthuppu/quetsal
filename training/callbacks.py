# =============================================================================
# quetsal/training/callbacks.py
# SB3 callbacks and env wrapper for per-family pass logging.
#
# PassLogWrapper        — gym.Wrapper around eval env; intercepts step() to
#                         accumulate per-family pass counts directly.
# LoggingEvalCallback   — EvalCallback subclass; flushes PassLogWrapper after
#                         each eval round (no unsupported 'callback' kwarg).
# PassLoggerCallback    — BaseCallback for training rollouts; flushes every
#                         log_freq env steps.
#
# CSV columns per family:
#   episodes            — completed episodes seen for this family
#   <pass>_total        — raw cumulative pass count
#   <pass>_per_ep       — pass count / episodes (avg per circuit replay)
# =============================================================================

from __future__ import annotations

__all__ = ["PassLogWrapper", "LoggingEvalCallback", "PassLoggerCallback"]

import csv
from collections import defaultdict, Counter
from pathlib import Path

import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

from quetsal.src.constants import ACTION_LABELS


# ── Shared helpers ────────────────────────────────────────────────────────────

_FAMILIES_ORDER = [
    "qv",
    "qaoa",
    "clifford_su4_su8",
    "clifford_su4",
    "iqp",
    "efficient_su2",
    "real_amplitudes",
    "unknown",
]

# CSV: episodes + two columns per action (total + per_ep)
_CSV_COLS = (
    ["step", "split", "family", "episodes"]
    + [f"{a}_total" for a in ACTION_LABELS]
    + [f"{a}_per_ep" for a in ACTION_LABELS]
)


def _ordered_families(counts: dict[str, Counter]) -> list[str]:
    seen = set(counts.keys())
    families = [f for f in _FAMILIES_ORDER if f in seen]
    families += sorted(seen - set(_FAMILIES_ORDER))
    return families


def _print_table(
    counts: dict[str, Counter],
    episodes: dict[str, int],
    step: int,
    split: str,
) -> None:
    families = _ordered_families(counts)
    if not families:
        return

    col_w, fam_w, ep_w = 7, 20, 8
    # Two sub-columns per action: total (int) + per_ep (float)
    action_header = "".join(
        f"{a[:6]:>{col_w}} {'avg':>{col_w}}" for a in ACTION_LABELS
    )
    header = f"{'family':<{fam_w}}{'episodes':>{ep_w}}  {action_header}"
    sep = "-" * len(header)

    print(f"\n[PassLogger/{split}] step={step:,}")
    print(sep)
    print(header)
    print(sep)

    for fam in families:
        n = max(episodes.get(fam, 1), 1)
        row = f"{fam:<{fam_w}}{episodes.get(fam, 0):>{ep_w}}  "
        row += "".join(
            f"{counts[fam].get(a, 0):>{col_w}} {counts[fam].get(a, 0)/n:>{col_w}.2f}"
            for a in ACTION_LABELS
        )
        print(row)

    print(sep)
    total_n = sum(episodes.get(f, 0) for f in families)
    denom = max(total_n, 1)
    total_row = f"{'TOTAL':<{fam_w}}{total_n:>{ep_w}}  "
    total_row += "".join(
        f"{sum(counts[f].get(a, 0) for f in families):>{col_w}} "
        f"{sum(counts[f].get(a, 0) for f in families)/denom:>{col_w}.2f}"
        for a in ACTION_LABELS
    )
    print(total_row)
    print(sep + "\n")


def _write_csv(
    counts: dict[str, Counter],
    episodes: dict[str, int],
    step: int,
    split: str,
    save_path: Path,
) -> None:
    families = _ordered_families(counts)
    if not families:
        return

    save_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not save_path.exists()
    with open(save_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_COLS, extrasaction="ignore")
        if write_header:
            writer.writeheader()

        for fam in families:
            n = max(episodes.get(fam, 1), 1)
            row: dict = {"step": step, "split": split, "family": fam, "episodes": episodes.get(fam, 0)}
            for a in ACTION_LABELS:
                raw = counts[fam].get(a, 0)
                row[f"{a}_total"] = raw
                row[f"{a}_per_ep"] = f"{raw / n:.4f}"
            writer.writerow(row)

        # TOTAL row
        total_n = sum(episodes.get(f, 0) for f in families)
        denom = max(total_n, 1)
        total_row: dict = {"step": step, "split": split, "family": "TOTAL", "episodes": total_n}
        for a in ACTION_LABELS:
            raw = sum(counts[f].get(a, 0) for f in families)
            total_row[f"{a}_total"] = raw
            total_row[f"{a}_per_ep"] = f"{raw / denom:.4f}"
        writer.writerow(total_row)


# ── Eval env wrapper ──────────────────────────────────────────────────────────


class PassLogWrapper(gym.Wrapper):
    """Gym wrapper that accumulates per-family pass counts and episode counts."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self._counts: dict[str, Counter] = defaultdict(Counter)
        self._episodes: dict[str, int] = defaultdict(int)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        family = info.get("family", "unknown")
        action_name = info.get("action_name")
        if action_name:
            self._counts[family][action_name] += 1
        if terminated or truncated:
            self._episodes[family] += 1
        return obs, reward, terminated, truncated, info

    def flush(self, step: int, save_path: Path, verbose: int) -> None:
        if verbose >= 1:
            _print_table(self._counts, self._episodes, step, "eval")
        _write_csv(self._counts, self._episodes, step, "eval", save_path)
        self._counts = defaultdict(Counter)
        self._episodes = defaultdict(int)


# ── Eval callback subclass ────────────────────────────────────────────────────


class LoggingEvalCallback(EvalCallback):
    """EvalCallback that flushes a PassLogWrapper after each eval round."""

    def __init__(
        self,
        *args,
        pass_log_path: str | Path,
        pass_log_verbose: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._pass_log_path = Path(pass_log_path)
        self._pass_log_verbose = pass_log_verbose

    def _on_step(self) -> bool:
        will_eval = self.eval_freq > 0 and self.n_calls % self.eval_freq == 0
        result = super()._on_step()
        if will_eval:
            inner = self.eval_env
            try:
                env = inner.envs[0]
            except AttributeError:
                env = inner
            while env is not None:
                if isinstance(env, PassLogWrapper):
                    env.flush(self.num_timesteps, self._pass_log_path, self._pass_log_verbose)
                    break
                env = getattr(env, "env", None)
        return result


# ── Training callback ─────────────────────────────────────────────────────────


class PassLoggerCallback(BaseCallback):
    """Log per-family pass counts (absolute + per episode) during training."""

    def __init__(self, log_freq: int, save_path: str | Path, verbose: int = 1) -> None:
        super().__init__(verbose=verbose)
        self.log_freq = log_freq
        self.save_path = Path(save_path)
        self._counts: dict[str, Counter] = defaultdict(Counter)
        self._episodes: dict[str, int] = defaultdict(int)
        self._last_flush: int = 0

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", [])
        for i, info in enumerate(self.locals.get("infos", [])):
            family = info.get("family", "unknown")
            action_name = info.get("action_name")
            if action_name:
                self._counts[family][action_name] += 1
            if i < len(dones) and dones[i]:
                self._episodes[family] += 1

        if self.num_timesteps - self._last_flush >= self.log_freq:
            if self.verbose >= 1:
                _print_table(self._counts, self._episodes, self.num_timesteps, "train")
            _write_csv(self._counts, self._episodes, self.num_timesteps, "train", self.save_path)
            self._last_flush = self.num_timesteps

        return True
