# =============================================================================
# quetsal/training/callbacks.py
# SB3 callbacks and gym wrapper for training observability and curriculum control.
#
# PassLogWrapper        — gym.Wrapper; intercepts step() to accumulate per-family
#                         pass counts and episode counts; flushed by eval/train CBs.
# LoggingEvalCallback   — EvalCallback subclass; flushes PassLogWrapper after each
#                         eval round; explicit best-model save.
# PassLoggerCallback    — BaseCallback; logs per-family pass counts to CSV every
#                         log_freq training steps.
# CurriculumCallback    — BaseCallback; fires at each eval checkpoint to drive stage
#                         promotion, stage 2 blend updates, and entropy decay.
# EarlyStoppingCallback — BaseCallback; stops training when eval reward stagnates
#                         for `patience` evals; counter reset on stage promotion.
# =============================================================================

from __future__ import annotations

__all__ = [
    "PassLogWrapper",
    "LoggingEvalCallback",
    "PassLoggerCallback",
    "CurriculumCallback",
    "EarlyStoppingCallback",
]

import csv
from collections import defaultdict, Counter
from pathlib import Path

import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

from quetsal.src.constants import ACTION_LABELS, CURRICULUM_STAGES

# ── Shared helpers ────────────────────────────────────────────────────────────

_FAMILIES_ORDER = [
    # Stage 1 — non-param foundation
    "qv",
    "clifford_su4",
    "clifford_su4_su8",
    "random_clifford",
    # Stage 2 — adds IQP (non-param) + parametric blend
    "iqp",
    "qaoa",
    "efficient_su2",
    # Stage 3 — adds parametric fine-tune
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
    action_header = "".join(f"{a[:6]:>{col_w}} {'avg':>{col_w}}" for a in ACTION_LABELS)
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
            row: dict = {
                "step": step,
                "split": split,
                "family": fam,
                "episodes": episodes.get(fam, 0),
            }
            for a in ACTION_LABELS:
                raw = counts[fam].get(a, 0)
                row[f"{a}_total"] = raw
                row[f"{a}_per_ep"] = f"{raw / n:.4f}"
            writer.writerow(row)

        # TOTAL row
        total_n = sum(episodes.get(f, 0) for f in families)
        denom = max(total_n, 1)
        total_row: dict = {
            "step": step,
            "split": split,
            "family": "TOTAL",
            "episodes": total_n,
        }
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
        # DoNothing counts after flush() has already zeroed the live counters.
        # Snapshot preserved after each flush so CurriculumCallback can read
        self._last_counts: dict[str, Counter] = defaultdict(Counter)
        self._last_episodes: dict[str, int] = defaultdict(int)

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
        self._last_counts = dict(self._counts)
        self._last_episodes = dict(self._episodes)
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
        self._explicit_best_reward: float = float("-inf")

    def _on_step(self) -> bool:
        will_eval = self.eval_freq > 0 and self.n_calls % self.eval_freq == 0
        result = super()._on_step()
        if will_eval:
            # Explicit best-model save: SB3's internal save (os.path.join +
            # zipfile write) can silently fail to overwrite on Windows when the
            # existing zip file handle isn't fully released before the next
            # write.  We track our own best and re-save directly via pathlib to
            # guarantee the file is always the true best checkpoint.
            _reward = getattr(self, "last_mean_reward", float("-inf"))
            _save_dir = getattr(self, "best_model_save_path", None)
            if _reward > self._explicit_best_reward and _save_dir:
                self._explicit_best_reward = _reward
                _dest = Path(_save_dir) / "best_model"
                self.model.save(str(_dest))
                if self.verbose >= 1:
                    print(
                        f"[quetsal] Best model updated "
                        f"(eval_reward={_reward:.4f}) -> {_dest}.zip"
                    )

            inner = self.eval_env
            try:
                env = inner.envs[0]
            except AttributeError:
                env = inner
            while env is not None:
                if isinstance(env, PassLogWrapper):
                    env.flush(
                        self.num_timesteps, self._pass_log_path, self._pass_log_verbose
                    )
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
            _write_csv(
                self._counts,
                self._episodes,
                self.num_timesteps,
                "train",
                self.save_path,
            )
            self._last_flush = self.num_timesteps

        return True


# ── Curriculum callback ───────────────────────────────────────────────────────


class CurriculumCallback(BaseCallback):
    """Fires after each eval round and calls CurriculumController.check_and_promote().

    Reads eval_mean_reward from LoggingEvalCallback.last_mean_reward and
    DoNothing-per-episode from the PassLogWrapper attached to the eval env.
    Also drives stage 2 blend updates and entropy decay each eval checkpoint.

    Parameters
    ----------
    curriculum      : CurriculumController instance.
    eval_cb         : the LoggingEvalCallback so we can read last_mean_reward.
    eval_log_wrapper: the PassLogWrapper around the eval env (for DoNothing count).
    eval_freq       : must match EvalCallback.eval_freq to detect eval timing.
    verbose         : 0 = silent, 1 = print stage info.
    """

    def __init__(
        self,
        curriculum,  # CurriculumController
        eval_cb,  # LoggingEvalCallback
        eval_log_wrapper: "PassLogWrapper",
        eval_freq: int,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose=verbose)
        self.curriculum = curriculum
        self.eval_cb = eval_cb
        self.eval_log_wrapper = eval_log_wrapper
        self.eval_freq = eval_freq
        self._stage_start_steps: int = 0

    def _on_step(self) -> bool:
        # Mirror EvalCallback's condition — fire just after eval ran
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            eval_mean_reward = getattr(self.eval_cb, "last_mean_reward", None)
            if eval_mean_reward is None:
                return True  # eval hasn't run yet

            # DoNothing per episode: read from the snapshot saved before flush()
            # zeroed the live counters (eval_cb fires before curriculum_cb).
            donothing_label = "DoNothing"
            total_episodes = sum(self.eval_log_wrapper._last_episodes.values())
            donothing_total = sum(
                self.eval_log_wrapper._last_counts[f].get(donothing_label, 0)
                for f in self.eval_log_wrapper._last_counts
            )
            donothing_per_ep = donothing_total / max(total_episodes, 1)

            # Stage 2-specific: update blend and entropy decay every eval
            if self.curriculum.stage == 2:
                self.curriculum.update_stage2_blend(eval_mean_reward)
                steps_in_stage = self.num_timesteps - self._stage_start_steps
                max_steps = sum(
                    CURRICULUM_STAGES[2].get("max_steps", 400_000)
                    for _ in [1]  # single lookup
                )
                self.curriculum.update_stage2_entropy(steps_in_stage, max_steps)

            # Check promotion
            promoted = self.curriculum.check_and_promote(
                eval_mean_reward, donothing_per_ep
            )
            if promoted:
                self._stage_start_steps = self.num_timesteps

        return True


# ── Early stopping callback ───────────────────────────────────────────────────


class EarlyStoppingCallback(BaseCallback):
    """Stop training when eval_mean_reward has not improved for ``patience`` evals.

    Fires at every eval checkpoint (must use the same ``eval_freq`` as
    LoggingEvalCallback).  Prints a warning at ``warn_after`` consecutive
    evals without improvement, then stops training at ``patience``.

    On curriculum runs: counter is reset on every stage promotion so that
    temporary reward drops at stage transitions do not trigger early stopping.

    Parameters
    ----------
    eval_cb     : LoggingEvalCallback — source of last_mean_reward.
    eval_freq   : must match EvalCallback.eval_freq.
    patience    : evals without improvement before stopping (default 5).
    min_delta   : minimum reward improvement to reset the counter (default 0.005).
    warn_after  : print warning after this many bad evals.  Defaults to patience // 2.
    curriculum_cb : optional CurriculumCallback — resets counter on stage promotion.
    verbose     : 0 = silent, 1 = print warnings and stop message.
    """

    def __init__(
        self,
        eval_cb,  # LoggingEvalCallback
        eval_freq: int,
        patience: int = 5,
        min_delta: float = 0.005,
        warn_after: int | None = None,
        curriculum_cb=None,  # CurriculumCallback — optional
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose=verbose)
        self.eval_cb = eval_cb
        self.eval_freq = eval_freq
        self.patience = patience
        self.min_delta = min_delta
        self.warn_after = (
            warn_after if warn_after is not None else max(1, patience // 2)
        )
        self.curriculum_cb = curriculum_cb

        self._best_reward: float = float("-inf")
        self._no_improve_count: int = 0
        self._last_curriculum_stage: int = 1

    def reset_counter(self) -> None:
        """Manually reset the no-improvement counter (called on stage promotion)."""
        self._no_improve_count = 0
        self._best_reward = float("-inf")  # reset best so new stage calibrates fresh

    def _on_step(self) -> bool:
        if self.eval_freq <= 0 or self.n_calls % self.eval_freq != 0:
            return True

        reward = getattr(self.eval_cb, "last_mean_reward", None)
        if reward is None:
            return True  # eval hasn't fired yet

        # Reset counter if curriculum just promoted to a new stage
        if self.curriculum_cb is not None:
            current_stage = getattr(self.curriculum_cb.curriculum, "stage", 1)
            if current_stage != self._last_curriculum_stage:
                self._last_curriculum_stage = current_stage
                self.reset_counter()
                if self.verbose >= 1:
                    print(
                        f"[EarlyStopping] Stage promoted to {current_stage} — "
                        f"resetting no-improvement counter"
                    )
                return True

        if reward > self._best_reward + self.min_delta:
            self._best_reward = reward
            self._no_improve_count = 0
        else:
            self._no_improve_count += 1

            if self.verbose >= 1 and self._no_improve_count == self.warn_after:
                print(
                    f"\n[EarlyStopping] WARNING: no improvement for "
                    f"{self._no_improve_count}/{self.patience} evals "
                    f"(best={self._best_reward:.4f}, current={reward:.4f}, "
                    f"min_delta={self.min_delta}). "
                    f"Will stop in {self.patience - self._no_improve_count} more evals "
                    f"without improvement.\n"
                )

            if self._no_improve_count >= self.patience:
                if self.verbose >= 1:
                    print(
                        f"\n[EarlyStopping] Stopping training at step "
                        f"{self.num_timesteps:,} — no improvement for "
                        f"{self.patience} consecutive evals. "
                        f"Best reward: {self._best_reward:.4f}\n"
                    )
                return False  # signals SB3 to end .learn()

        return True
