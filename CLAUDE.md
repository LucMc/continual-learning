# Known issues

## PPO observation-normalisation bug (unfixed — on todo list)

`continual_learning/trainers/ppo_trainer.py` has a bug in `JittedContinualPPOTrainer.train_step` when `normalize_observations=True`. The bootstrap observation passed as `next_obs` to `PPO.update` is never normalised, so `last_values = vf.apply_fn(vf.params, next_obs, ...)` (line ~135) feeds the VF raw env output instead of a normalised input. The resulting garbage bootstrap corrupts GAE targets, the VF trains on those, and values explode — measured 10⁹–10¹⁰ magnitude on Humanoid vs. ~20 when fixed.

**Impact.** On slippery_humanoid (5M steps, seed 42): buggy return ≈ 283, unnormalised ≈ 440, fixed ≈ 1829. The bug is strictly worse than disabling normalisation.

**Affected experiments** (explicitly set `normalize_observations=True`):
- `experiments/slippery_humanoid.py`
- `experiments/slippery_humanoid_tau1.py`

All other experiments use the default `False` and are unaffected. Ant experiments are fine.

**Fix** (not yet applied to the repo): in `train_step`, normalise `observation` with the updated normaliser before passing it to `update`, but keep the raw `observation` in the returned state so the next rollout receives unnormalised input. A working copy with the patch is at `/tmp/ppo_trainer_fixed.py`; measurement script at `/tmp/test_ppo_normalization.py`.

**Todo.** Apply the fix in-repo and regenerate any Humanoid results that depended on the two experiment scripts above.
