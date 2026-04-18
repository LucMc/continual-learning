"""Exploratory test for ContinualRandomIntervalDelayWrapper.

Runs the wrapper around a MetaWorld env and a classic-control env,
exercises every mode / output / embedding combo, and produces a
matplotlib plot that makes the changing delay intervals visible.
"""

from __future__ import annotations

from pathlib import Path

import gymnasium
import matplotlib.pyplot as plt
import metaworld
import numpy as np

from continual_learning.utils.continual_delay_wrapper import (
    ContinualRandomIntervalDelayWrapper,
)


# ---------- env factories ----------


def make_metaworld_env(task_name: str = "reach-v3", seed: int = 0):
    ml1 = metaworld.ML1(task_name, seed=seed)
    env_cls = ml1.train_classes[task_name]
    env = env_cls()
    env.set_task(ml1.train_tasks[0])
    return env


def make_pendulum_env():
    return gymnasium.make("Pendulum-v1")


# ---------- printing helpers ----------


def describe_obs(label, obs):
    if isinstance(obs, tuple):
        print(f"  [{label}] tuple, len={len(obs)}")
        for i, el in enumerate(obs):
            shape = getattr(el, "shape", None)
            print(f"    [{i}] type={type(el).__name__} shape={shape} head={str(el)[:60]}")
    else:
        print(f"  [{label}] type={type(obs).__name__} shape={getattr(obs, 'shape', None)}")


# ---------- scenario runner ----------


def run_scenario(name, base_env_fn=make_metaworld_env, **wrapper_kwargs):
    print("=" * 80)
    print(f"SCENARIO: {name}")
    print(f"  kwargs: {wrapper_kwargs}")
    print("=" * 80)

    env = ContinualRandomIntervalDelayWrapper(base_env_fn(), **wrapper_kwargs)

    print(f"  overall_obs_delay_range={env.overall_obs_delay_range}")
    print(f"  overall_act_delay_range={env.overall_act_delay_range}")
    print(f"  sampled obs_delay_range={env.obs_delay_range}")
    print(f"  sampled act_delay_range={env.act_delay_range}")
    print(f"  observation_space={str(env.observation_space)[:160]}")

    obs, _ = env.reset(seed=0)
    describe_obs("reset", obs)

    print("  Stepping 6 times...")
    for t in range(6):
        act = env.action_space.sample()
        obs, r, term, trun, _ = env.step(act)
        if isinstance(obs, tuple):
            print(
                f"    t={t}: r={float(r):+.3f} term={term} trun={trun} "
                f"obs[0].shape={obs[0].shape} n_act_buf={len(obs[1])} "
                f"alpha={obs[2]} kappa={obs[3]} beta={obs[4]}"
            )
        else:
            print(f"    t={t}: r={float(r):+.3f} term={term} trun={trun} obs.shape={obs.shape}")

    print("  Resetting 4 times...")
    seen = set()
    for i in range(4):
        env.reset(seed=i + 1)
        seen.add((tuple(env.obs_delay_range), tuple(env.act_delay_range)))
        print(f"    reset {i}: obs={env.obs_delay_range} act={env.act_delay_range}")
    mode = wrapper_kwargs.get("mode", "continual")
    expected = {
        "fixed": "exactly 1 (never resample)",
        "multi-task": "up to 4 (resample each reset)",
        "continual": "1 if resample_every is None, else may change within step",
    }[mode]
    print(f"  distinct (obs,act) interval pairs across 4 resets: {len(seen)}  [expected: {expected}]")


# ---------- focused probes ----------


def check_seeding():
    print("=" * 80)
    print("PROBE: reset(seed=X) propagates to the underlying env (Pendulum-v1)")
    print("=" * 80)

    def first_raw_obs(seed):
        # Pendulum has stochastic init, so distinct seeds give distinct obs.
        # interval_emb_type=None → obs[0] is the raw env obs (no added embedding).
        # mode=fixed    → the sub-interval is stable across this probe.
        env = ContinualRandomIntervalDelayWrapper(
            make_pendulum_env(),
            obs_delay_range=range(0, 5),
            act_delay_range=range(0, 5),
            interval_emb_type=None,
            mode="fixed",
            output="dcac",
        )
        obs, _ = env.reset(seed=seed)
        return np.asarray(obs[0])

    o1, o2, o3 = first_raw_obs(42), first_raw_obs(42), first_raw_obs(999)
    print(f"  seed=42 a vs b : max|diff| = {np.max(np.abs(o1 - o2)):.3e}  (expect 0)")
    print(f"  seed=42 vs 999 : max|diff| = {np.max(np.abs(o1 - o3)):.3e}  (expect > 0)")
    assert np.allclose(o1, o2), "seed not propagated to underlying env"
    assert not np.allclose(o1, o3), "different seeds produced identical obs"


def check_give_kappa():
    print("=" * 80)
    print("PROBE: give_kappa=True updates obs space and reset works")
    print("=" * 80)
    env = ContinualRandomIntervalDelayWrapper(
        make_metaworld_env(),
        obs_delay_range=range(0, 5),
        act_delay_range=range(0, 5),
        give_kappa=True,
        mode="continual",
        output="standard",
    )
    obs, _ = env.reset(seed=0)
    declared = env.observation_space.shape[0]
    print(f"  declared obs dim = {declared}, actual = {obs.shape[0]}")
    assert declared == obs.shape[0], "declared obs size != produced obs size"


def check_obs_space_stability():
    print("=" * 80)
    print("PROBE: observation_space stays constant across resets")
    print("=" * 80)
    env = ContinualRandomIntervalDelayWrapper(
        make_metaworld_env(),
        obs_delay_range=range(0, 5),
        act_delay_range=range(0, 5),
        mode="multi-task",
        output="dcac",
    )
    spaces = set()
    for i in range(5):
        env.reset(seed=i)
        spaces.add(str(env.observation_space))
    print(f"  unique observation_spaces across 5 resets: {len(spaces)}  (expect 1)")
    assert len(spaces) == 1, "observation_space changed across resets"


def check_get_interval_edges():
    print("=" * 80)
    print("PROBE: get_interval works with range of length 2 and reaches the max")
    print("=" * 80)
    env = ContinualRandomIntervalDelayWrapper(
        make_metaworld_env(),
        obs_delay_range=range(0, 2),
        act_delay_range=range(0, 2),
        mode="fixed",
        output="dcac",
    )
    print(f"  range(0,2) → sub_obs={env.obs_delay_range} sub_act={env.act_delay_range}")
    # Check the overall max is reachable across many draws
    draws = {env.get_interval(range(0, 5)).stop - 1 for _ in range(200)}
    print(f"  reachable maxima over 200 draws from range(0,5): {sorted(draws)}")
    assert 4 in draws, "get_interval never sampled the overall max"


class _CounterEnv(gymnasium.Env):
    """Toy env whose observation encodes the step counter.

    obs = [step_idx, step_idx, step_idx] before increment. Lets us verify
    which past observation the delay wrapper actually delivers.
    """

    def __init__(self):
        super().__init__()
        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32,
        )
        self.action_space = gymnasium.spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32,
        )
        self._t = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._t = 0
        return np.full((3,), self._t, dtype=np.float32), {}

    def step(self, action):
        self._t += 1
        obs = np.full((3,), self._t, dtype=np.float32)
        return obs, 0.0, False, False, {}


def check_delayed_obs_content():
    """For a fixed [lo, hi] obs-delay interval, the obs delivered at wrapper-step
    m must have been produced by env-step k* with m - k* ∈ [lo, hi].

    Uses a counter env (obs[0] = env._t) so we can identify exactly which past
    env-step's obs was delivered. A burn-in is required because the wrapper's
    reset pre-fills the observation buffer with copies of the env's reset obs
    (counter=0), and those reset entries can be delivered at early steps — they
    get pushed out once enough real steps have added newer entries to the deque.
    """
    print("=" * 80)
    print("PROBE: delivered obs came from an env-step consistent with [lo, hi]")
    print("=" * 80)
    overall = range(0, 5)
    # Internal deque maxlen = overall.stop + overall.stop = 10; burn in past that.
    burn_in = overall.stop * 2 + 2
    seen_lags: dict[tuple[int, int], set[int]] = {}
    for obs_lo, obs_hi in [(0, 1), (1, 3), (2, 3), (0, 4)]:
        for act_lo, act_hi in [(0, 1), (1, 3)]:
            env = ContinualRandomIntervalDelayWrapper(
                _CounterEnv(),
                obs_delay_range=overall,
                act_delay_range=overall,
                interval_emb_type=None,
                mode="fixed",
                output="dcac",
            )
            env.obs_delay_range = range(obs_lo, obs_hi + 1)
            env.act_delay_range = range(act_lo, act_hi + 1)

            delivered_first, _ = env.reset(seed=0)
            assert int(delivered_first[0][0]) == 0, \
                f"reset should deliver counter=0, got {int(delivered_first[0][0])}"

            env_steps = 0
            lags = set()
            for _ in range(burn_in + 30):
                delivered, *_ = env.step(env.action_space.sample())
                env_steps += 1
                counter = int(delivered[0][0])
                lag = env_steps - counter
                if env_steps > burn_in:
                    assert obs_lo <= lag <= obs_hi, (
                        f"interval obs=[{obs_lo},{obs_hi}] act=[{act_lo},{act_hi}] "
                        f"after burn-in: env_steps={env_steps} counter={counter} "
                        f"lag={lag} outside [{obs_lo}, {obs_hi}]"
                    )
                    lags.add(lag)
            seen_lags[(obs_lo, obs_hi)] = lags | seen_lags.get((obs_lo, obs_hi), set())
            print(
                f"  obs=[{obs_lo},{obs_hi}] act=[{act_lo},{act_hi}]: "
                f"30 post-burn-in steps all in-range, lags seen = {sorted(lags)}"
            )

    # Sanity: for non-trivial intervals we should see more than one distinct lag
    # over the runs (delay was actually random, not pinned).
    for (lo, hi), lags in seen_lags.items():
        if hi > lo:
            assert len(lags) >= 2, (
                f"interval [{lo},{hi}] only ever produced lag={lags} — "
                "delay sampling looks degenerate"
            )


def check_delayed_obs_content_continual():
    """Same semantic check under mode='continual' with mid-run resampling:
    after each resample, observed lags must fall inside the new sub-interval.
    """
    print("=" * 80)
    print("PROBE: under continual mode, lags track the current sub-interval")
    print("=" * 80)
    overall = range(0, 5)
    resample_every = 40
    burn_after_resample = overall.stop * 2 + 2  # let buffer turn over

    env = ContinualRandomIntervalDelayWrapper(
        _CounterEnv(),
        obs_delay_range=overall,
        act_delay_range=overall,
        interval_emb_type=None,
        mode="continual",
        output="dcac",
        resample_every=resample_every,
    )
    env.reset(seed=0)
    env_steps = 0
    # Track lags bucketed by the sub-interval active at delivery time.
    per_interval_lags: dict[tuple[int, int], list[int]] = {}
    last_resample_at = 0
    prev_interval = (env.obs_delay_range.start, env.obs_delay_range.stop)
    for _ in range(5 * resample_every):
        delivered, *_ = env.step(env.action_space.sample())
        env_steps += 1
        interval = (env.obs_delay_range.start, env.obs_delay_range.stop)
        if interval != prev_interval:
            last_resample_at = env_steps
            prev_interval = interval
        if env_steps - last_resample_at < burn_after_resample:
            continue  # wait for buffer to reflect the new interval
        counter = int(delivered[0][0])
        lag = env_steps - counter
        lo, hi_excl = interval
        hi = hi_excl - 1
        assert lo <= lag <= hi, (
            f"continual mode: active interval=[{lo},{hi}] env_steps={env_steps} "
            f"counter={counter} lag={lag} outside [{lo},{hi}]"
        )
        per_interval_lags.setdefault(interval, []).append(lag)
    for interval, lags in per_interval_lags.items():
        lo, hi_excl = interval
        print(
            f"  active obs_delay_range=range({lo},{hi_excl}): "
            f"{len(lags)} post-burn-in deliveries, lags seen = {sorted(set(lags))}"
        )


def check_fixed_mode_is_fixed():
    print("=" * 80)
    print("PROBE: mode='fixed' keeps intervals constant across resets")
    print("=" * 80)
    env = ContinualRandomIntervalDelayWrapper(
        make_metaworld_env(),
        obs_delay_range=range(0, 5),
        act_delay_range=range(0, 5),
        mode="fixed",
        output="dcac",
    )
    initial = (tuple(env.obs_delay_range), tuple(env.act_delay_range))
    for i in range(5):
        env.reset(seed=i)
        now = (tuple(env.obs_delay_range), tuple(env.act_delay_range))
        assert now == initial, f"fixed mode changed intervals on reset {i}"
    print(f"  intervals stable across 5 resets: obs={env.obs_delay_range} act={env.act_delay_range}")


# ---------- continual-mode visualization ----------


def run_continual_trace(
    out_path: Path,
    n_steps: int = 400,
    resample_every: int = 50,
    env_label: str = "MetaWorld reach-v3",
    base_env_fn=make_metaworld_env,
):
    print("=" * 80)
    print(f"VISUAL: continual mode, {n_steps} steps, resample every {resample_every}")
    print("=" * 80)

    env = ContinualRandomIntervalDelayWrapper(
        base_env_fn(),
        obs_delay_range=range(0, 6),
        act_delay_range=range(0, 6),
        mode="continual",
        output="dcac",
        resample_every=resample_every,
    )
    env.reset(seed=0)

    steps = list(range(n_steps))
    obs_mins, obs_maxs, act_mins, act_maxs = [], [], [], []
    change_log = []
    prev = None
    for t in steps:
        obs_mins.append(env.obs_delay_range.start)
        obs_maxs.append(env.obs_delay_range.stop - 1)
        act_mins.append(env.act_delay_range.start)
        act_maxs.append(env.act_delay_range.stop - 1)
        current = (obs_mins[-1], obs_maxs[-1], act_mins[-1], act_maxs[-1])
        if current != prev:
            change_log.append((t, *current))
            prev = current
        _, _, term, trun, _ = env.step(env.action_space.sample())
        if term or trun:
            env.reset()

    print(f"  recorded {len(change_log)} interval changes across {n_steps} steps")
    print(f"  {'step':>6} {'obs_min':>8} {'obs_max':>8} {'act_min':>8} {'act_max':>8}")
    for row in change_log:
        print(f"  {row[0]:>6d} {row[1]:>8d} {row[2]:>8d} {row[3]:>8d} {row[4]:>8d}")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.fill_between(steps, obs_mins, obs_maxs, step="post", alpha=0.35,
                    color="tab:blue", label="obs delay range")
    ax.fill_between(steps, act_mins, act_maxs, step="post", alpha=0.35,
                    color="tab:orange", label="act delay range")
    ax.plot(steps, obs_mins, drawstyle="steps-post", color="tab:blue", lw=1)
    ax.plot(steps, obs_maxs, drawstyle="steps-post", color="tab:blue", lw=1)
    ax.plot(steps, act_mins, drawstyle="steps-post", color="tab:orange", lw=1)
    ax.plot(steps, act_maxs, drawstyle="steps-post", color="tab:orange", lw=1)
    ax.set_xlabel("step")
    ax.set_ylabel("delay (env timesteps)")
    ax.set_title(
        f"Continual delay-interval resampling ({env_label}, "
        f"resample_every={resample_every})"
    )
    ax.set_ylim(-0.5, max(env.overall_obs_delay_range.stop, env.overall_act_delay_range.stop) - 0.5)
    ax.legend(loc="upper right")
    ax.grid(alpha=0.2)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  saved plot → {out_path}")


# ---------- main ----------


def main():
    run_scenario(
        "dcac + two_hot + continual (MetaWorld)",
        obs_delay_range=range(0, 5),
        act_delay_range=range(0, 5),
        interval_emb_type="two_hot",
        delay_emb_type="one_hot",
        mode="continual",
        output="dcac",
    )
    run_scenario(
        "standard + two_hot + multi-task (MetaWorld)",
        obs_delay_range=range(0, 5),
        act_delay_range=range(0, 5),
        interval_emb_type="two_hot",
        delay_emb_type="one_hot",
        mode="multi-task",
        output="standard",
    )
    run_scenario(
        "standard + no interval emb + scalar delay (MetaWorld)",
        obs_delay_range=range(0, 5),
        act_delay_range=range(0, 5),
        interval_emb_type=None,
        delay_emb_type="scalar",
        mode="continual",
        output="standard",
    )
    run_scenario(
        "dcac + fixed mode (MetaWorld)",
        obs_delay_range=range(0, 5),
        act_delay_range=range(0, 5),
        interval_emb_type="two_hot",
        delay_emb_type="one_hot",
        mode="fixed",
        output="dcac",
    )
    run_scenario(
        "dcac + multi-task (Pendulum-v1)",
        base_env_fn=make_pendulum_env,
        obs_delay_range=range(0, 5),
        act_delay_range=range(0, 5),
        interval_emb_type="two_hot",
        delay_emb_type="one_hot",
        mode="multi-task",
        output="dcac",
    )

    check_seeding()
    check_give_kappa()
    check_obs_space_stability()
    check_get_interval_edges()
    check_delayed_obs_content()
    check_delayed_obs_content_continual()
    check_fixed_mode_is_fixed()

    repo_root = Path(__file__).resolve().parents[1]
    run_continual_trace(
        out_path=repo_root / "plots" / "png" / "continual_delay_intervals.png",
    )

    print("=" * 80)
    print("ALL CHECKS PASSED")
    print("=" * 80)


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True, linewidth=120)
    main()
