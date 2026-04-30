# =============================================================================
# quetsal/training/curriculum.py
# Curriculum controller for staged PPO training.
#
# CurriculumController tracks the current stage and decides when to promote
# the agent to the next stage based on eval metrics.  On promotion it swaps
# the training circuit pool, updates PPO hyperparameters, and saves a stage
# checkpoint.
#
# Stages (defined in constants.CURRICULUM_STAGES):
#   1 — Foundation:        4 non-param families (QV/Clifford-SU4/Clifford-SU4-SU8/RandomClifford)
#   2 — Parametric blend:  adds IQP (non-param) + QAOA/EfficientSU2 (param, 10%→20% mix)
#   3 — Balanced fine-tune: all 8 families; non-param 17% each, param 5% each; RealAmplitudes introduced here
#
# Used only when train.py is invoked with --curriculum.
# =============================================================================

from __future__ import annotations

__all__ = ["CurriculumController"]

from pathlib import Path
from typing import TYPE_CHECKING

from stable_baselines3.common.utils import get_schedule_fn

import numpy as np

from quetsal.src.constants import (
    CURRICULUM_STAGES,
    HERON_R2_BASIS,
)
from quetsal.src.environment.circuits import generate_weighted_circuits
from quetsal.src.environment.pass_env import PassManagerEnv
from quetsal.training.circuit_cache import sample_weighted_pool

if TYPE_CHECKING:
    from stable_baselines3 import PPO


class CurriculumController:
    """Manages staged curriculum progression for the Quetsal PPO agent.

    Parameters
    ----------
    model           : the SB3 PPO model being trained.
    train_env       : PassManagerEnv whose circuits list is swapped on promotion.
    eval_env_inner  : the inner PassManagerEnv inside the eval Monitor/Wrapper stack,
                      so its circuits can be updated to match the new stage.
    n_qubits_range  : qubit range passed through to circuit generators.
    count_per_family: circuits per family for the training pool.
    circuit_seed    : base seed; eval uses seed + 999 offset.
    ckpt_dir        : directory to save per-stage checkpoint zips.
    verbose         : 0 = silent, 1 = print promotions.
    """

    def __init__(
        self,
        model: "PPO",
        train_env: PassManagerEnv,
        eval_env_inner: PassManagerEnv,
        n_qubits_range: tuple[int, int],
        count_per_family: int,
        circuit_seed: int,
        ckpt_dir: Path,
        verbose: int = 1,
        smoke: bool = False,
        max_stage: int = 3,
        master_pool: dict | None = None,
    ) -> None:
        self.model = model
        self.train_env = train_env
        self.eval_env_inner = eval_env_inner
        self.n_qubits_range = n_qubits_range
        self.count_per_family = count_per_family
        self.circuit_seed = circuit_seed
        self.ckpt_dir = ckpt_dir
        self.verbose = verbose
        self.max_stage = max_stage
        self.master_pool = master_pool  # dict[family->circuits] or None

        self.stage = 1
        self._consecutive_passing = 0
        self._recent_rewards: list[float] = []
        self._stage2_param_mix: float = CURRICULUM_STAGES[2]["param_start_mix"]

        # Smoke mode: override promotion gates so all 3 stages run in a quick run.
        # Gates are overridden in-place on a deep copy so CURRICULUM_STAGES is unchanged.
        if smoke:
            from quetsal.src.constants import TRAINING_MODE_ARGS
            import copy

            smoke_cfg = TRAINING_MODE_ARGS[0].get("curriculum_smoke", {})
            self._stages = copy.deepcopy(CURRICULUM_STAGES)
            for stage_id in (1, 2):
                p = self._stages[stage_id]["promotion"]
                if "eval_mean_reward_min" in smoke_cfg:
                    p["eval_mean_reward_min"] = smoke_cfg["eval_mean_reward_min"]
                if "donothing_per_ep_max" in smoke_cfg and stage_id == 1:
                    p["donothing_per_ep_max"] = smoke_cfg["donothing_per_ep_max"]
                if "reward_delta_max" in smoke_cfg and stage_id == 2:
                    p["reward_delta_max"] = smoke_cfg["reward_delta_max"]
                if "consecutive_evals" in smoke_cfg:
                    p["consecutive_evals"] = smoke_cfg["consecutive_evals"]
            if verbose >= 1:
                print("[Curriculum] Smoke mode — promotion gates lowered for quick run")
        else:
            self._stages = CURRICULUM_STAGES

    # ── Public API ────────────────────────────────────────────────────────────

    def check_and_promote(
        self,
        eval_mean_reward: float,
        donothing_per_ep: float,
        pass_log_counts: dict | None = None,
    ) -> bool:
        """Check promotion criteria and advance stage if met.

        Called by CurriculumCallback after each eval checkpoint.

        Returns True if stage was advanced.
        """
        if self.stage >= self.max_stage:
            return False  # advancement blocked (max_stage reached)
        if self.stage == 1:
            return self._check_stage1(eval_mean_reward, donothing_per_ep)
        elif self.stage == 2:
            return self._check_stage2(eval_mean_reward)
        return False  # stage 3 is the final stage

    def update_stage2_blend(self, eval_mean_reward: float) -> None:
        """Increase parametric mix by blend_step if this eval passes threshold.

        Called every eval checkpoint during stage 2 (even if not promoting).
        """
        cfg = CURRICULUM_STAGES[2]
        if eval_mean_reward >= cfg["promotion"]["eval_mean_reward_min"]:
            old_mix = self._stage2_param_mix
            self._stage2_param_mix = min(
                cfg["param_max_mix"],
                self._stage2_param_mix + cfg["param_blend_step"],
            )
            if self._stage2_param_mix != old_mix:
                self._rebuild_stage2_circuits()
                if self.verbose >= 1:
                    print(
                        f"[Curriculum] Stage 2 param mix: {old_mix:.0%} → "
                        f"{self._stage2_param_mix:.0%}"
                    )

    def update_stage2_entropy(
        self, steps_in_stage: int, total_stage_steps: int
    ) -> None:
        """Linearly decay ent_coef from ent_coef_start → ent_coef_end over stage 2."""
        cfg = CURRICULUM_STAGES[2]
        progress = min(steps_in_stage / max(total_stage_steps, 1), 1.0)
        new_ent = cfg["ent_coef_start"] + progress * (
            cfg["ent_coef_end"] - cfg["ent_coef_start"]
        )
        self.model.ent_coef = float(new_ent)

    # ── Stage-specific promotion checks ──────────────────────────────────────

    def _check_stage1(self, eval_mean_reward: float, donothing_per_ep: float) -> bool:
        cfg = self._stages[1]["promotion"]
        passed = (
            eval_mean_reward > cfg["eval_mean_reward_min"]
            and donothing_per_ep < cfg["donothing_per_ep_max"]
        )
        if passed:
            self._consecutive_passing += 1
            if self.verbose >= 1:
                print(
                    f"[Curriculum] Stage 1 gate passed "
                    f"({self._consecutive_passing}/{cfg['consecutive_evals']}): "
                    f"reward={eval_mean_reward:.3f}, donothing/ep={donothing_per_ep:.2f}"
                )
        else:
            self._consecutive_passing = 0

        if self._consecutive_passing >= cfg["consecutive_evals"]:
            self._promote_to_stage2()
            return True
        return False

    def _check_stage2(self, eval_mean_reward: float) -> bool:
        cfg = self._stages[2]["promotion"]
        self._recent_rewards.append(eval_mean_reward)
        if len(self._recent_rewards) > 2:
            self._recent_rewards.pop(0)

        passed = (
            eval_mean_reward >= cfg["eval_mean_reward_min"]
            and len(self._recent_rewards) == 2
            and abs(self._recent_rewards[1] - self._recent_rewards[0])
            < cfg["reward_delta_max"]
        )
        if passed:
            self._consecutive_passing += 1
            if self.verbose >= 1:
                print(
                    f"[Curriculum] Stage 2 gate passed "
                    f"({self._consecutive_passing}/{cfg['consecutive_evals']}): "
                    f"reward={eval_mean_reward:.3f}, δ={abs(self._recent_rewards[1]-self._recent_rewards[0]):.4f}"
                )
        else:
            self._consecutive_passing = 0

        if self._consecutive_passing >= cfg["consecutive_evals"]:
            self._promote_to_stage3()
            return True
        return False

    # ── Promotion actions ─────────────────────────────────────────────────────

    def _promote_to_stage2(self) -> None:
        """Save stage 1 checkpoint, rebuild circuits, update hyperparams."""
        self._save_stage_checkpoint(from_stage=1)
        self.stage = 2
        self._consecutive_passing = 0
        self._recent_rewards = []
        self._stage2_param_mix = CURRICULUM_STAGES[2]["param_start_mix"]

        # Rebuild circuit pool with stage 2 blend (start at param_start_mix)
        self._rebuild_stage2_circuits()

        # Update hyperparameters — ent_coef and step_penalty change at stage 1→2.
        # LR is unchanged (stage 3 is the only stage that lowers LR to 1e-4).
        cfg = CURRICULUM_STAGES[2]
        self.model.ent_coef = float(cfg["ent_coef_start"])
        self.train_env.step_penalty = float(cfg["step_penalty"])
        self.eval_env_inner.step_penalty = float(cfg["step_penalty"])

        if self.verbose >= 1:
            print(
                f"\n[Curriculum] *** PROMOTED TO STAGE 2 ***\n"
                f"  param_mix={self._stage2_param_mix:.0%}, "
                f"ent_coef={cfg['ent_coef_start']}, "
                f"step_penalty={cfg['step_penalty']}"
            )

    def _promote_to_stage3(self) -> None:
        """Save stage 2 checkpoint, rebuild circuits, update hyperparams.

        Samples from master_pool when available; otherwise regenerates.
        """
        self._save_stage_checkpoint(from_stage=2)
        self.stage = 3
        self._consecutive_passing = 0

        # Rebuild circuit pool — all families in stage 3
        cfg = CURRICULUM_STAGES[3]
        total = self.count_per_family * len(cfg["families"])
        eval_total = max(14, int(total * 0.20))

        if self.master_pool is not None:
            circuits = sample_weighted_pool(
                self.master_pool,
                families=cfg["families"],
                family_weights=cfg["family_weights"],
                total_count=total,
                rng_seed=self.circuit_seed + 300,
            )
            eval_circuits = sample_weighted_pool(
                self.master_pool,
                families=cfg["families"],
                family_weights=cfg["family_weights"],
                total_count=eval_total,
                rng_seed=self.circuit_seed + 999 + 300,
            )
        else:
            circuits = generate_weighted_circuits(
                families=cfg["families"],
                family_weights=cfg["family_weights"],
                total_count=total,
                n_qubits_range=self.n_qubits_range,
                seed=self.circuit_seed + 300,
                basis_gates=HERON_R2_BASIS,
            )
            eval_circuits = generate_weighted_circuits(
                families=cfg["families"],
                family_weights=cfg["family_weights"],
                total_count=eval_total,
                n_qubits_range=self.n_qubits_range,
                seed=self.circuit_seed + 999 + 300,
                basis_gates=HERON_R2_BASIS,
            )

        self.train_env.circuits = circuits
        self.eval_env_inner.circuits = eval_circuits

        # Apply donothing_early_penalty to train and eval envs
        penalty = cfg.get("donothing_early_penalty", 0.0)
        self.train_env.donothing_early_penalty = penalty
        self.eval_env_inner.donothing_early_penalty = penalty

        # Update PPO hyperparams.
        # SB3 stores clip_range and ent_coef internally as callables λ(progress)→float.
        # Assigning a raw float would crash ppo.train() which calls self.clip_range(progress).
        # get_schedule_fn wraps a float into a constant callable automatically.
        self.model.ent_coef = float(cfg["ent_coef"])  # float — used directly in loss
        self.model.clip_range = get_schedule_fn(
            cfg["clip_range"]
        )  # callable — called with progress
        for pg in self.model.policy.optimizer.param_groups:
            pg["lr"] = cfg["lr"]

        if self.verbose >= 1:
            print(
                f"\n[Curriculum] *** PROMOTED TO STAGE 3 ***\n"
                f"  {len(circuits)} train circuits, {len(eval_circuits)} eval circuits\n"
                f"  lr={cfg['lr']}, ent_coef={cfg['ent_coef']}, "
                f"clip_range={cfg['clip_range']}, "
                f"donothing_early_penalty={penalty} (applied as -{penalty} in step())"
            )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _rebuild_stage2_circuits(self) -> None:
        """Rebuild stage 2 circuit pool with the current param_mix.

        Samples from master_pool when available; otherwise regenerates.
        """
        cfg = CURRICULUM_STAGES[2]
        total = self.count_per_family * len(cfg["families"])
        param_count = int(total * self._stage2_param_mix)
        nonparam_count = total - param_count

        # Build per-family weights: split budget between non-param and param groups
        n_np = len(cfg["nonparam_families"])
        n_p = len(cfg["param_families"])
        weights: dict[str, float] = {}
        for f in cfg["nonparam_families"]:
            weights[f] = (nonparam_count / n_np) / total
        for f in cfg["param_families"]:
            weights[f] = (param_count / n_p) / total

        seed_offset = 100 + round(self._stage2_param_mix * 100)
        eval_total = max(len(cfg["families"]) * 2, int(total * 0.20))

        if self.master_pool is not None:
            circuits = sample_weighted_pool(
                self.master_pool,
                families=cfg["families"],
                family_weights=weights,
                total_count=total,
                rng_seed=self.circuit_seed + seed_offset,
            )
            eval_circuits = sample_weighted_pool(
                self.master_pool,
                families=cfg["families"],
                family_weights=weights,
                total_count=eval_total,
                rng_seed=self.circuit_seed + 999 + seed_offset,
            )
        else:
            circuits = generate_weighted_circuits(
                families=cfg["families"],
                family_weights=weights,
                total_count=total,
                n_qubits_range=self.n_qubits_range,
                seed=self.circuit_seed + seed_offset,
                basis_gates=HERON_R2_BASIS,
            )
            eval_circuits = generate_weighted_circuits(
                families=cfg["families"],
                family_weights=weights,
                total_count=eval_total,
                n_qubits_range=self.n_qubits_range,
                seed=self.circuit_seed + 999 + seed_offset,
                basis_gates=HERON_R2_BASIS,
            )

        self.train_env.circuits = circuits
        self.eval_env_inner.circuits = eval_circuits

    def _save_stage_checkpoint(self, from_stage: int) -> None:
        """Save a model zip before transitioning away from a stage."""
        path = self.ckpt_dir / f"stage{from_stage}_final"
        self.model.save(str(path))
        if self.verbose >= 1:
            print(f"[Curriculum] Stage {from_stage} checkpoint -> {path}.zip")
