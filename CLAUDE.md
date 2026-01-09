# CLAUDE.md

## Commands

```bash
# Install
pip install -e .[cpu]          # or [cuda12], [cuda13], [tpu]

# Test
pytest                         # All tests
pytest tests/test_sac_bro.py -v  # SAC/BRO tests

# Experiments (Tyro CLI)
python -m experiments.metaworld_mt10 --wandb_mode disabled  # BRO smoke test
python -m experiments.split_mnist --wandb_mode disabled     # SL smoke test
```

## Architecture

JAX/Flax continual learning framework for SL and RL.

### Directory Structure
- `configs/`: Flax `@struct.dataclass` frozen configs (all JAX pytrees)
- `models/`: MLP, CNN, ResNet + RL networks in `rl.py` (BroNet, policies, critics)
- `optim/`: Base optimizers + reset methods (CBP, REDO, ReGrAMA, ShrinkAndPerturb)
- `trainers/`: `sac.py` (SAC), `bro_learner.py` (BRO algorithm), `bro.py` (BRO trainer)
- `envs/`: `metaworld.py` (MT10), slippery MuJoCo variants
- `utils/`: `replay_buffer.py`, `training.py` (custom TrainState)

### Critical: Custom Optimizer Pattern

**This codebase's optimizers return params directly, NOT updates.**

```python
# Standard optax (NOT used here):
updates, opt_state = tx.update(grads, opt_state, params)
new_params = optax.apply_updates(params, updates)

# This codebase:
new_params, opt_state = tx.update(grads, opt_state, params, features=features)
# new_params IS the final params - no apply_updates needed
```

The custom `TrainState` in `utils/training.py` handles this - do NOT use standard Flax TrainState.

### Reset Methods

Wrap base optimizers: `RegramaConfig(tx=AdamConfig(...), ...)`. Combined via `attach_reset_method()`.

## BRO Algorithm

BRO (Nauman et al. 2024) is **NOT** just SAC with high replay ratio. Key differences:

| Component | SAC | BRO |
|-----------|-----|-----|
| Critic | Single-value Q | Distributional (100 quantiles, Huber loss) |
| Actor | Single policy | Dual: Conservative + Optimistic |
| Q-combination | min(Q1, Q2) | (Q1+Q2)/2 ± weight*\|Q1-Q2\|/2 |
| Target entropy | -action_dim | -action_dim/2 |
| Optimizer | Adam | AdamW |
| Coefficients | Fixed alpha | Learnable temp, optimism, regularizer |
| Architecture | MLP | BroNet (LayerNorm + Residuals) |

### BRO Files
- `models/rl.py`: BroNet, BRONormalTanhPolicy, BRODualTanhPolicy, BRODistributionalCritic, Temperature, Adjustment
- `trainers/bro_learner.py`: Full BRO algorithm with all update functions
- `trainers/bro.py`: Training loop wrapper for MetaWorld

### BRO Key Details
- **Optimistic actor** takes conservative mean/std as input, outputs action shift
- **KL regularization** between optimistic and conservative policies
- **Reset schedule**: `[15001, 50001, 250001, 500001, 750001, 1000001, 1500001, 2000001]`
- **Gradient clipping**: max norm 1.0 on all updates
- **Adjustment coefficients**: Use bounded tanh output (`log_val_min=-10, log_val_max=7.5`)

## Common Pitfalls

1. **Flax `.apply()` with `return_params=True`**: Returns `(dist, means, stds)` tuple, not just distribution. Type checker complains but works at runtime.

2. **MetaWorld initialization**: Takes ~40-50s. First JIT compilation also slow.

3. **wandb package enumeration bug**: Use `save_code=False` and `settings=wandb.Settings(_save_requirements=False)` in `wandb.init()`.

4. **NaN in MuJoCo**: Add action clipping (`jnp.clip(action, -1, 1)`) and `jnp.nan_to_num()` before stepping.

5. **Replay buffer dtype**: Dones stored as float (for JAX operations), not bool.

## Reference Implementation

Official BRO code at `~/BiggerRegularizedOptimistic`:
- `jaxrl/networks/common.py`: BroNet architecture
- `jaxrl/networks/policies.py`: NormalTanhPolicy, DualTanhPolicy
