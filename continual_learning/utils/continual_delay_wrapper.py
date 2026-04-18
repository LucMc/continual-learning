from collections import deque
from random import sample
import random
import itertools

import gymnasium as gym
from gymnasium.core import Env
from gymnasium.spaces import Tuple, Discrete, Box, Dict

import numpy as np



class RandomDelayWrapper(gym.Wrapper):
    """
    Wrapper for any non-RTRL environment, modelling random observation and action delays
    NB: alpha refers to the abservation delay, it is >= 0
    NB: The state-space now contains two different action delays:
        kappa is such that alpha+kappa is the index of the first action that was going to be applied when the observation started being captured, it is useful for the model
            (when kappa==0, it means that the delay is actually 1)
        beta is such that alpha+beta is the index of the last action that is known to have influenced the observation, it is useful for credit assignment (e.g. AC/DC)
            (alpha+beta is often 1 step bigger than the action buffer, and it is always >= 1)
    Kwargs:
        obs_delay_range: range in which alpha is sampled
        act_delay_range: range in which kappa is sampled
        initial_action: action (default None): action with which the action buffer is filled at reset() (if None, sampled in the action space)
    """

    def __init__(
        self,
        env,
        obs_delay_range=range(0, 4),
        act_delay_range=range(0, 4),
        initial_action=None,
        skip_initial_actions=False,
        PD=False,
    ):
        super().__init__(env)
        self.wrapped_env = env
        self.obs_delay_range = obs_delay_range
        self.act_delay_range = act_delay_range
        # self.overall_act_delay_range = act_delay_range
        # self.overall_obs_delay_range = obs_delay_range

        self.observation_space = Tuple(
            (
                env.observation_space,  # most recent observation
                Tuple([env.action_space] * (obs_delay_range.stop + act_delay_range.stop - 1)),  # action buffer
                Discrete(obs_delay_range.stop),  # observation delay int64
                Discrete(act_delay_range.stop),  # action delay int64
            )
        )

        self.initial_action = initial_action
        self.skip_initial_actions = skip_initial_actions
        self.past_actions = deque(maxlen=obs_delay_range.stop + act_delay_range.stop)
        self.past_observations = deque(maxlen=obs_delay_range.stop)
        self.arrival_times_actions = deque(maxlen=act_delay_range.stop)
        self.arrival_times_observations = deque(maxlen=obs_delay_range.stop)

        self.PD = PD

        self.t =0
        self.done_signal_sent = False
        self.next_action = None
        self.cum_rew_actor = 0.0
        self.cum_rew_brain = 0.0
        self.prev_action_idx = 0  # TODO : initialize this better

    def reset(self, **kwargs):
        self.cum_rew_actor = 0.0
        self.cum_rew_brain = 0.0
        self.prev_action_idx = 0  # TODO : initialize this better
        self.done_signal_sent = False
        
        first_observation, reset_info = super().reset(**kwargs)

        # fill up buffers
        self.t = -(self.obs_delay_range.stop + self.act_delay_range.stop)  # this is <= -2
        while self.t < 0:
            act = self.action_space.sample() if self.initial_action is None else self.initial_action
            self.send_action(act, init=True)  # TODO : initialize this better
            self.send_observation(
                (first_observation, 0.0, False, False, reset_info, 0, 1)
            )  # TODO : initialize this better
            self.t += 1
        self.receive_action()  # an action has to be applied

        assert self.t == 0
        received_observation, *_ = self.receive_observation()
        return received_observation, reset_info

    def step(self, action):
        """
        When kappa is 0 and alpha is 0, this is equivalent to the RTRL setting
        (The inference time is NOT considered part of beta or kappa)
        """

        # at the brain
        self.send_action(action)

        # at the remote actor
        if self.t < self.act_delay_range.stop and self.skip_initial_actions:
            # assert False, "skip_initial_actions==True is not supported"
            # do nothing until the brain's first actions arrive at the remote actor
            self.receive_action()
        elif self.done_signal_sent:
            # just resend the last observation until the brain gets it
            self.send_observation(self.past_observations[0])
        else:
            # Editted this for PD controller by adding obs
            # print(f"Running: {self.PD}")
            if not self.PD:
                m, r, term, trun, info = self.env.step(
                    self.next_action
                )  # before receive_action (e.g. rtrl setting with 0 delays)
            elif self.PD:
                m, r, term, trun, info = self.env.step(
                    self.next_action, self.past_observations[0][0]
                )  # before receive_action (e.g. rtrl setting with 0 delays)
            d = term | trun
            kappa, beta = self.receive_action()
            self.cum_rew_actor += r
            self.done_signal_sent = d
            self.send_observation((m, self.cum_rew_actor, term, trun, info, kappa, beta))

        # at the brain again
        m, cum_rew_actor_delayed, term, trun, info = self.receive_observation()
        r = cum_rew_actor_delayed - self.cum_rew_brain
        self.cum_rew_brain = cum_rew_actor_delayed

        self.t += 1

        # print("RB SHAPE RD\n", m[1][0].shape, len(m[1]))
        # print("RD obs space\n", self.observation_space)
        return m, r, term, trun, info

    def send_action(self, action, init=False):
        """
        Appends action to the left of self.past_actions
        Simulates the time at which it will reach the agent and stores it on the left of self.arrival_times_actions
        """
        # at the brain
        (kappa,) = (
            sample(self.act_delay_range, 1)
            if not init
            else [
                0,
            ]
        )  # TODO: change this if we implement a different initialization
        self.arrival_times_actions.appendleft(self.t + kappa)
        self.past_actions.appendleft(action)

    def receive_action(self):
        """
        Looks for the last created action that has arrived before t at the agent
        NB: since it is the most recently created action that the agent got, this is the one that is to be applied
        Returns:
            next_action_idx: int: the index of the action that is going to be applied
            prev_action_idx: int: the index of the action previously being applied (i.e. of the action that influenced the observation since it is retrieved instantaneously in usual Gym envs)
        """
        # CAUTION: from the brain point of view, the "previous action"'s age (kappa_t) is not like the previous "next action"'s age (beta_{t-1}) (e.g. repeated observations)
        prev_action_idx = (
            self.prev_action_idx + 1
        )  # + 1 is to account for the fact that this was the right idx 1 time-step before

        next_action_idx = next(i for i, t in enumerate(self.arrival_times_actions) if t <= self.t)
        self.prev_action_idx = next_action_idx
        self.next_action = self.past_actions[next_action_idx]
        # print(f"DEBUG: next_action_idx:{next_action_idx}, prev_action_idx:{prev_action_idx}")
        return next_action_idx, prev_action_idx

    def send_observation(self, obs):
        """
        Appends obs to the left of self.past_observations
        Simulates the time at which it will reach the brain and appends it in self.arrival_times_observations
        """
        # at the remote actor
        (alpha,) = sample(self.obs_delay_range, 1)
        self.arrival_times_observations.appendleft(self.t + alpha)
        self.past_observations.appendleft(obs)

    def receive_observation(self):
        """
        Looks for the last created observation at the agent/observer that reached the brain at time t
        NB: since this is the most recently created observation that the brain got, this is the one currently being considered as the last observation
        Returns:
            augmented_obs: tuple:
                m: object: last observation that reached the brain
                past_actions: tuple: the history of actions that the brain sent so far
                alpha: int: number of micro time steps it took the last observation to travel from the agent/observer to the brain
                kappa: int: action travel delay + number of micro time-steps for which the next action has been applied at the agent
                beta: int: action travel delay + number of micro time-steps for which the previous action has been applied at the agent
            r: float: delayed reward corresponding to the transition that created m
            d: bool: delayed done corresponding to the transition that created m
            info: dict: delayed info corresponding to the transition that created m
        """
        # at the brain
            
        alpha = next(i for i, t in enumerate(self.arrival_times_observations) if t <= self.t)

        m, r, term, trun, info, kappa, beta = self.past_observations[alpha]
        return (
            (m.astype(np.float32),
             tuple(itertools.islice(self.past_actions, 0, self.past_actions.maxlen - 1)),
             alpha,
             kappa,
             beta),
             r,
             term,
             trun,
             info,
        )


class UnseenRandomDelayWrapper(RandomDelayWrapper):
    """
    Wrapper that translates the RandomDelayWrapper back to the usual RL setting
    Use this wrapper to see what happens to vanilla RL algorithms facing random delays
    """

    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        self.observation_space = env.unwrapped.observation_space

    def reset(self, **kwargs):
        t, reset_info = super().reset(**kwargs)  # t: (m, tuple(self.past_actions), alpha, kappa, beta)
        return t[0], reset_info

    def step(self, action):
        t, *aux = super().step(action)  # t: (m, tuple(self.past_actions), alpha, kappa, beta)
        return (t[0], *aux)


## My own only augmented state wrapper
# from gym.spaces import Dict, Box, Tuple
# ASSERT THE SHAPES FROM RESET AND STEP ARE RIGHT

class AugmentedRandomDelayWrapper(RandomDelayWrapper):
    """
    Wrapper that translates the RandomDelayWrapper back to the usual RL setting
    Use this wrapper to see what happens to augmented observation state RL algorithms facing random delays
    """

    # Fix for HER
    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        self.HER = False  # Remove?
        self.delay = len(self.obs_delay_range) - 1 + len(self.obs_delay_range) - 1 + 1

        if type(self.observation_space[0]) == Dict:  # If HER
            self.HER = True
            self.default_observation_space = self.observation_space[0]["observation"]
            self.observation_space = self.observation_space[0]
            self.new_obs_shape = self.observation_space["observation"].shape[0] + (
                env.action_space.shape[0] * self.delay
            )  # ( sum([sum(list(x)) for x in kwargs.values()]) + 1))
            # new_obs_shape = self.observation_space['observation'].shape[0] + (env.action_space.shape[0] * ( sum([sum(list(x)) for x in kwargs.values()]))+1)

            self.observation_space["observation"] = Box(
                low=-np.inf, high=np.inf, shape=(self.new_obs_shape,), dtype=np.float32
            )
            # print(( sum([sum(list(x)) for x in kwargs.values()]) + 1))
            # print("init",self.observation_space['observation'].shape)
        else:
            self.new_obs_shape = env.observation_space.shape[0] + (env.action_space.shape[0] * self.delay)
            self.observation_space = Box(low=-np.inf, high=np.inf, shape=(self.new_obs_shape,), dtype=np.float32)

    def reset(self, **kwargs):
        t, reset_info = super().reset(**kwargs)  # t: (m, tuple(self.past_actions), alpha, kappa, beta)
        if self.HER:
            aug_state = np.append(t[0]["observation"], t[1])
            t[0]["observation"] = aug_state
            aug_state = t[0]
        else:
            aug_state = np.append(t[0], t[1]).astype(np.float32)  # Combine action buffer and state

        return aug_state, reset_info

    # Make this more efficient
    def step(self, action):
        t, *aux = super().step(action)  # t: (m, tuple(self.past_actions), alpha, kappa, beta)
        if self.HER:
            # print("step",t[0]['observation'].shape)
            if t[0]["observation"].shape[0] == self.default_observation_space.shape[0]:  # Fix for any value later
                # print("Fixing bug")
                aug_state = np.append(t[0]["observation"], t[1])
                t[0]["observation"] = aug_state

            return (t[0], *aux)

        else:
            aug_state = np.append(t[0], t[1])  # Combine action buffer and state

        return (aug_state, *aux)


##


def simple_wifi_sampler1():
    return np.random.choice([1, 2, 3, 4, 5, 6], p=[0.3082, 0.5927, 0.0829, 0.0075, 0.0031, 0.0056])


def simple_wifi_sampler2():
    return np.random.choice([1, 2, 3, 4], p=[0.3082, 0.5927, 0.0829, 0.0162])


class WifiDelayWrapper1(RandomDelayWrapper):
    """
    Simple sampler built from a dataset of 10000 real-world wifi communications
    The atomic time-step is 0.02s
    All communication times above 0.1s have been clipped to 0.1s
    """

    def __init__(self, env, initial_action=None, skip_initial_actions=False):
        super().__init__(
            env,
            obs_delay_range=range(0, 7),
            act_delay_range=range(0, 7),
            initial_action=initial_action,
            skip_initial_actions=skip_initial_actions,
        )

    def send_observation(self, obs):
        # at the remote actor
        alpha = simple_wifi_sampler1()
        self.arrival_times_observations.appendleft(self.t + alpha)
        self.past_observations.appendleft(obs)

    def send_action(self, action, init=False):
        # at the brain
        kappa = (
            simple_wifi_sampler1() if not init else 0
        )  # TODO: change this if we implement a different initialization
        self.arrival_times_actions.appendleft(self.t + kappa)
        self.past_actions.appendleft(action)


class WifiDelayWrapper2(RandomDelayWrapper):
    """
    Simple sampler built from a dataset of 10000 real-world wifi communications
    The atomic time-step is 0.02s
    All communication times above 0.1s have been clipped to 0.1s
    """

    def __init__(self, env, initial_action=None, skip_initial_actions=False):
        super().__init__(
            env,
            obs_delay_range=range(0, 5),
            act_delay_range=range(0, 5),
            initial_action=initial_action,
            skip_initial_actions=skip_initial_actions,
        )

    def send_observation(self, obs):
        # at the remote actor
        alpha = simple_wifi_sampler2()
        self.arrival_times_observations.appendleft(self.t + alpha)
        self.past_observations.appendleft(obs)

    def send_action(self, action, init=False):
        # at the brain
        kappa = (
            simple_wifi_sampler2() if not init else 0
        )  # TODO: change this if we implement a different initialization
        self.arrival_times_actions.appendleft(self.t + kappa)
        self.past_actions.appendleft(action)


'''
Rewrite a new version of this which keeps output variables in DCAC format as that is what I'll mainly
be working off ig. Also create continual version so I can check the continuous neuron resetting
Plan for Tuesday:
    - Deep how the wrapper works again
    - Get it working for DCAC normal
    - BONUS: Working on typical gym setting too

- Fixed working on DCAC []
- Multi-task working on DCAC []
- Continual working on DCAC []

- Plot neural plasticity []
- Add continual backprop []
- Test against continuout continual backprop []
- Add baselines, SAC []

# backup - PCGrad for multi-delay learning?/split into 2 papers
'''

class ContinualRandomIntervalDelayWrapper(RandomDelayWrapper):
    """Random-delay wrapper whose obs/act delay *intervals* can change during training.

    The ``overall_obs_delay_range`` / ``overall_act_delay_range`` bound what is
    ever possible. At any moment the wrapper uses a *sub-interval* of each as
    the actual [min, max] delay. When that sub-interval is re-sampled depends
    on ``mode``:

    - ``"fixed"``:      sample once at ``__init__``, never again.
    - ``"multi-task"``: re-sample on every ``reset()``.
    - ``"continual"``:  re-sample every ``resample_every`` calls to ``step()``.
    """

    AVAILABLE_MODES = ("continual", "multi-task", "fixed")
    AVAILABLE_OUTPUTS = ("dcac", "standard")
    AVAILABLE_DELAY_EMB_TYPES = ("one_hot", "float", "scalar", None)
    AVAILABLE_INTERVAL_EMB_TYPES = ("two_hot", "float", "scalar", None)

    def __init__(
        self,
        env,
        obs_delay_range: range = range(0, 4),
        act_delay_range: range = range(0, 4),
        interval_emb_type: str | None = "two_hot",
        delay_emb_type: str | None = "one_hot",
        give_kappa: bool = False,
        mode: str = "continual",
        output: str = "dcac",
        resample_every: int | None = None,
        **kwargs,
    ):
        self.mode = mode.lower() if isinstance(mode, str) else mode
        self.output = output.lower() if isinstance(output, str) else output
        self.interval_emb_type = interval_emb_type.lower() if interval_emb_type else None
        self.delay_emb_type = delay_emb_type.lower() if delay_emb_type else None

        assert self.mode in self.AVAILABLE_MODES, \
            f"mode={self.mode!r} not in {self.AVAILABLE_MODES}"
        assert self.output in self.AVAILABLE_OUTPUTS, \
            f"output={self.output!r} not in {self.AVAILABLE_OUTPUTS}"
        assert self.delay_emb_type in self.AVAILABLE_DELAY_EMB_TYPES, \
            f"delay_emb_type={self.delay_emb_type!r} not in {self.AVAILABLE_DELAY_EMB_TYPES}"
        assert self.interval_emb_type in self.AVAILABLE_INTERVAL_EMB_TYPES, \
            f"interval_emb_type={self.interval_emb_type!r} not in {self.AVAILABLE_INTERVAL_EMB_TYPES}"
        assert len(obs_delay_range) >= 1, "obs_delay_range must have length >= 1"
        assert len(act_delay_range) >= 1, "act_delay_range must have length >= 1"
        assert self.output == "standard" or self.delay_emb_type == "one_hot", \
            "dcac output always uses one_hot delay embedding internally"

        self.overall_obs_delay_range = obs_delay_range
        self.overall_act_delay_range = act_delay_range
        self.n_obs_delays = len(obs_delay_range)
        self.n_act_delays = len(act_delay_range)

        # Initialise parent with the *overall* ranges so its internal deques
        # are sized to hold the largest possible history. We then override
        # ``self.obs_delay_range`` / ``self.act_delay_range`` with the actual
        # sub-interval currently in effect — those are what the parent samples
        # delays from in ``send_action`` / ``send_observation``.
        super().__init__(env, obs_delay_range, act_delay_range, **kwargs)
        self.obs_delay_range = self.get_interval(obs_delay_range)
        self.act_delay_range = self.get_interval(act_delay_range)

        self.give_kappa = give_kappa
        self.resample_every = resample_every
        self._step_counter = 0

        # Interval-embedding length
        if self.interval_emb_type == "two_hot":
            int_emb_len = (self.n_obs_delays - 1) + (self.n_act_delays - 1)
        elif self.interval_emb_type in ("float", "scalar"):
            int_emb_len = 4
        else:
            int_emb_len = 0

        # Build observation_space using OVERALL ranges → stable across resamples.
        base_obs_dim = env.observation_space.shape[0]
        if self.output == "dcac":
            self.observation_space = Tuple((
                Box(
                    shape=(base_obs_dim + int_emb_len,),
                    low=-np.inf, high=np.inf, dtype=np.float32,
                ),
                Tuple([env.action_space] * (self.overall_obs_delay_range.stop
                                            + self.overall_act_delay_range.stop - 1)),
                Discrete(self.overall_obs_delay_range.stop),
                Discrete(self.overall_act_delay_range.stop),
            ))
        else:  # "standard"
            action_dim = int(np.prod(env.action_space.shape))
            act_buf_len = action_dim * (
                self.overall_obs_delay_range.stop + self.overall_act_delay_range.stop - 1
            )
            obs_emb_len, act_emb_len, beta_emb_len = self._delay_emb_lens()
            extra = act_buf_len + obs_emb_len + act_emb_len + int_emb_len
            if self.give_kappa:
                extra += beta_emb_len
            self.observation_space = Box(
                shape=(base_obs_dim + extra,),
                low=-np.inf, high=np.inf, dtype=np.float32,
            )

    # --- helpers ---

    def _delay_emb_lens(self):
        """Lengths of (obs_emb, act_emb, beta_emb) produced by format_obs."""
        if self.delay_emb_type == "one_hot":
            return (self.n_obs_delays, self.n_act_delays, self.n_obs_delays)
        if self.delay_emb_type in ("scalar", "float"):
            return (1, 1, 1)
        return (0, 0, 0)

    def get_interval(self, interval_range):
        """Sample a contiguous sub-range [a, b] with a <= b, inclusive at both ends."""
        assert len(interval_range) >= 1, \
            f"interval_range must have length >= 1 (got {interval_range})"
        if len(interval_range) == 1:
            return interval_range
        lo, hi = min(interval_range), max(interval_range)
        a, b = sorted(random.sample(range(lo, hi + 1), 2))
        return range(a, b + 1)

    def _resample_intervals(self):
        self.obs_delay_range = self.get_interval(self.overall_obs_delay_range)
        self.act_delay_range = self.get_interval(self.overall_act_delay_range)

    def get_delay_interval_embedding(self):
        rel_obs_min = self.obs_delay_range.start - self.overall_obs_delay_range.start
        rel_obs_max = (self.obs_delay_range.stop - 1) - self.overall_obs_delay_range.start
        rel_act_min = self.act_delay_range.start - self.overall_act_delay_range.start
        rel_act_max = (self.act_delay_range.stop - 1) - self.overall_act_delay_range.start

        if self.interval_emb_type == "two_hot":
            two_hot_obs = np.zeros(self.n_obs_delays - 1, dtype=np.float32)
            two_hot_act = np.zeros(self.n_act_delays - 1, dtype=np.float32)
            two_hot_obs[min(rel_obs_min, self.n_obs_delays - 2)] = 1.0
            two_hot_obs[min(rel_obs_max, self.n_obs_delays - 2)] = 1.0
            two_hot_act[min(rel_act_min, self.n_act_delays - 2)] = 1.0
            two_hot_act[min(rel_act_max, self.n_act_delays - 2)] = 1.0
            return (two_hot_obs, two_hot_act)

        if self.interval_emb_type == "scalar":
            return ((self.obs_delay_range.start, self.obs_delay_range.stop - 1),
                    (self.act_delay_range.start, self.act_delay_range.stop - 1))

        if self.interval_emb_type == "float":
            n_obs = max(self.n_obs_delays - 1, 1)
            n_act = max(self.n_act_delays - 1, 1)
            return ((rel_obs_min / n_obs, rel_obs_max / n_obs),
                    (rel_act_min / n_act, rel_act_max / n_act))

        raise ValueError(f"Unsupported interval_emb_type: {self.interval_emb_type}")

    def format_obs(self, obs):
        """Flatten the tuple-form obs into a single vector (standard output)."""
        obs_delay = obs[2]
        act_delay = obs[3]
        beta_delay = obs[4]  # Parent returns (m, past_actions, alpha, kappa, beta)

        if self.delay_emb_type == "one_hot":
            obs_emb = np.zeros(self.n_obs_delays, dtype=np.float32)
            act_emb = np.zeros(self.n_act_delays, dtype=np.float32)
            beta_emb = np.zeros(self.n_obs_delays, dtype=np.float32)
            obs_emb[obs_delay - self.overall_obs_delay_range.start] = 1.0
            act_emb[act_delay - self.overall_act_delay_range.start] = 1.0
            beta_emb[min(beta_delay - self.overall_obs_delay_range.start,
                         self.n_obs_delays - 1)] = 1.0
        elif self.delay_emb_type == "scalar":
            obs_emb = np.array([obs_delay], dtype=np.float32)
            act_emb = np.array([act_delay], dtype=np.float32)
            beta_emb = np.array([beta_delay], dtype=np.float32)
        elif self.delay_emb_type == "float":
            n_obs = max(self.n_obs_delays - 1, 1)
            n_act = max(self.n_act_delays - 1, 1)
            obs_emb = np.array(
                [(obs_delay - self.overall_obs_delay_range.start) / n_obs],
                dtype=np.float32,
            )
            act_emb = np.array(
                [(act_delay - self.overall_act_delay_range.start) / n_act],
                dtype=np.float32,
            )
            beta_emb = np.array(
                [(beta_delay - self.overall_obs_delay_range.start) / n_obs],
                dtype=np.float32,
            )
        else:
            raise ValueError(f"Invalid delay_emb_type: {self.delay_emb_type}")

        act_buf_flat = np.concatenate([np.asarray(a).reshape(-1) for a in obs[1]])
        parts = [np.asarray(obs[0]).reshape(-1), act_buf_flat, obs_emb, act_emb]
        if self.give_kappa:
            parts.append(beta_emb)
        return np.concatenate(parts, axis=0).astype(np.float32)

    def get_padded_act_buf(self, received_obs):
        target = self.overall_act_delay_range.stop + self.overall_obs_delay_range.stop - 1
        buf = received_obs[1]
        if len(buf) == target:
            return buf
        if len(buf) > target:
            return buf[:target]
        zero_action = (
            np.zeros_like(np.asarray(buf[-1]))
            if len(buf) > 0
            else np.zeros(self.env.action_space.shape, dtype=np.float32)
        )
        padded = tuple(zero_action for _ in range(target - len(buf))) + buf
        if self.output == "dcac":
            assert len(padded) == len(self.observation_space[1]), (
                f"padded buf len {len(padded)} != obs_space buf len "
                f"{len(self.observation_space[1])}"
            )
        return padded

    def get_int_emb_obs(self, received_obs, padded_act_buf):
        if self.interval_emb_type is None:
            return (received_obs[0], padded_act_buf, *received_obs[2:])

        if self.interval_emb_type == "two_hot":
            obs_emb, act_emb = self.get_delay_interval_embedding()
            new_head = np.concatenate(
                (received_obs[0], obs_emb, act_emb), axis=0, dtype=np.float32,
            )
        else:
            (obs_min, obs_max), (act_min, act_max) = self.get_delay_interval_embedding()
            new_head = np.concatenate(
                (received_obs[0],
                 np.array([obs_min], dtype=np.float32),
                 np.array([obs_max], dtype=np.float32),
                 np.array([act_min], dtype=np.float32),
                 np.array([act_max], dtype=np.float32)),
                axis=0, dtype=np.float32,
            )
        return (new_head, padded_act_buf, *received_obs[2:])

    # --- gym API ---

    def step(self, action, **kwargs):
        received_obs, *aux = super().step(action)
        padded_act_buf = self.get_padded_act_buf(received_obs)
        received_obs = self.get_int_emb_obs(received_obs, padded_act_buf)
        obs = received_obs if self.output == "dcac" else self.format_obs(received_obs)

        self._step_counter += 1
        if (self.mode == "continual"
                and self.resample_every
                and self._step_counter % self.resample_every == 0):
            self._resample_intervals()

        return (obs, *aux)

    def reset(self, *, seed=None, **kwargs):
        if self.mode == "multi-task":
            self._resample_intervals()
        self._step_counter = 0
        received_obs, reset_info = super().reset(seed=seed, **kwargs)
        padded_act_buf = self.get_padded_act_buf(received_obs)
        received_obs = self.get_int_emb_obs(received_obs, padded_act_buf)
        obs = received_obs if self.output == "dcac" else self.format_obs(received_obs)
        return obs, reset_info


'''
Version for algos using standard gym environments
'''
class GymRandomIntervalDelayWrapper(RandomDelayWrapper):
    def __init__(self, env, obs_delay_range=range(0, 4),
                 act_delay_range=range(0, 4),
                 interval_emb_type="two_hot", # twohot, float, scalar
                 delay_emb_type="one_hot", # (one_hot, float, scalar, multi_hot?[0,0,1,2,0,0]?) transition delay embedding
                 give_kappa=False,
                 **kwargs):

        self.interval_emb_type = interval_emb_type
        self.delay_emb_type = delay_emb_type

        self.give_kappa = give_kappa
        self.overall_act_delay_range = act_delay_range
        self.overall_obs_delay_range = obs_delay_range
        self.obs_delay_range = self.get_interval(obs_delay_range)
        self.act_delay_range = self.get_interval(act_delay_range)
        super().__init__(env, self.obs_delay_range, self.act_delay_range, **kwargs)

    def get_interval(self, interval_range):
        nums = []

        while len(set(nums)) != 2:  # Pick 2 different numbers
            nums = [
                random.randint(min(interval_range), max(interval_range)),
                random.randint(min(interval_range), max(interval_range)),
            ]

        return range(min(nums), max(nums))

    def get_delay_interval_embedding(self) -> tuple:
        # min and maxs for given interval
        rel_obs_min = self.obs_delay_range.start - self.overall_obs_delay_range.start
        rel_obs_max = self.obs_delay_range.stop - self.overall_obs_delay_range.start
        rel_act_min = self.act_delay_range.start - self.overall_act_delay_range.start
        rel_act_max = self.act_delay_range.stop - self.overall_act_delay_range.start

        n_pos_obs_delays = len(self.overall_obs_delay_range)
        n_pos_act_delays = len(self.overall_act_delay_range)

        if self.interval_emb_type == "two_hot": # provides the delays as a two hot embedding
            two_hot_obs = np.zeros(n_pos_obs_delays)
            two_hot_act = np.zeros(n_pos_act_delays)

            # print("n_pos_act_delays", n_pos_act_delays)
            # print("rel_obs_min", rel_obs_min)
            # print("rel_obs_max", rel_obs_max)
            # print("rel_act_min", rel_act_min)
            # print("rel_act_max", rel_act_max)

            two_hot_obs[[rel_obs_min, rel_obs_max-1]] = 1
            two_hot_act[[rel_act_min, rel_act_max-1]] = 1

            return (two_hot_obs, two_hot_act)

        elif self.interval_emb_type == "scalar": # provides the range simply as scalars
            return ((self.obs_delay_range.start, self.obs_delay_range.stop + 1), (self.act_delay_range.start, self.act_delay_range.stop + 1))

        elif self.interval_emb_type == "float": # normalise over range, show delay as percentage of max range
            # print(f"rel_obs_min", rel_obs_min)
            # print("rel_obs_max", rel_obs_max)
            # print("rel_act_min", rel_act_min)
            # print("rel_act_max", rel_act_max)
            # print("n_pos_obs_delays", n_pos_obs_delays)
            # print("n_pos_act_delays", n_pos_act_delays)

            obs_min_float = rel_obs_min / (n_pos_obs_delays)
            obs_max_float = rel_obs_max / (n_pos_obs_delays)

            act_min_float = rel_act_min / (n_pos_act_delays)
            act_max_float = rel_act_max / (n_pos_act_delays)

            return ((obs_min_float, obs_max_float), (act_min_float, act_max_float))
        else:
            raise Exception(f"Unsupported embedding type: {self.interval_emb_type}")

    def format_obs(self, obs):
        action_buffer = obs[1] # Can be variable length based on delay interval so needs padding
        act_buf_max = self.overall_act_delay_range.stop + max(self.overall_act_delay_range)
        action_buffer += (0,)*(act_buf_max - len(action_buffer))

        obs_diff = max(self.overall_obs_delay_range) - min(self.overall_obs_delay_range)
        act_diff = max(self.overall_act_delay_range) - min(self.overall_act_delay_range)
        
        # get delays
        obs_delay = obs[2]
        act_delay = obs[3]
        kappa_delay = obs[4]

        if self.delay_emb_type == "one_hot":
            obs_emb = np.zeros(obs_diff+1) # Should have similar to get delay interval embedding
            act_emb = np.zeros(act_diff+1)
            kappa_emb = np.zeros(obs_diff+1)
            obs_emb[obs_delay-self.overall_obs_delay_range.start] = 1.0 # Error when delay=array size, is this index -1 or array not large enough?
            act_emb[act_delay-self.overall_act_delay_range.start] = 1.0
            kappa_emb[kappa_delay-self.overall_obs_delay_range.start] = 1.0

        elif  self.delay_emb_type == "scalar":
            obs_emb = [obs_delay]
            act_emb = [act_delay]
            kappa_emb = [kappa_delay]

        elif  self.delay_emb_type == "float":
            # Provide as float, relative to the overall interval

            obs_emb = [(obs_delay - self.overall_obs_delay_range.start) / obs_diff]
            act_emb = [(act_delay - self.overall_act_delay_range.start) / act_diff]
            kappa_emb = [(kappa_delay - self.overall_obs_delay_range.start) / obs_diff]
        else:
            raise ValueError("Invalide delay_emb_type", self.delay_emb_type)

        # print("obs+act delay:", obs_delay, act_delay)
        # print("obs len", len(obs))
        # print("obs0:", obs[0], len(obs[0]))
        # print("obs1:", obs[1], len(obs[1]))
        # print("action_buffer:", action_buffer, len(action_buffer))
        # print("obs_emb", obs_emb, len(obs_emb))
        # print("act_emb", act_emb, len(obs_emb))

        if self.give_kappa:
            new_obs = np.concatenate((obs[0], obs[1], obs_emb, act_emb, kappa_emb), axis=0)
        else:
            new_obs = np.concatenate((obs[0], obs[1], obs_emb, act_emb), axis=0)

        return new_obs

    def step(self, action, **kwargs):
        t, *aux = super().step(action)  # t: (m, tuple(self.past_actions), alpha, kappa, beta)
        return (self.format_obs(t), *aux) # aux=rew,term,trun,info

    def reset(self, **kwargs):
        """
        observation_space:
        Tuple((
            obs_space,  # most recent observation
            Tuple([act_space] * (obs_delay_range.stop + act_delay_range.stop)),  # action buffer
            Discrete(obs_delay_range.stop),  # observation delay int64
            Discrete(act_delay_range.stop),  # action delay int64
        ))
        """
        self.obs_delay_range = self.get_interval(self.overall_obs_delay_range)
        self.act_delay_range = self.get_interval(self.overall_act_delay_range)

        # print(f"obs delay interval", self.obs_delay_range)
        # print(f"act delay interval", self.act_delay_range)

        recieved_obs, reset_info = super().reset(**kwargs)

        obs = self.format_obs(recieved_obs)  # one hot encoding of the transition delay
        return obs.astype(np.float32)

        # print(f"obs_delay_max: ", obs_delay_max)
        # print(f"obs_delay_min: ", obs_delay_min)
        #
        # print(f"act_delay_max: ", act_delay_max)
        # print(f"act_delay_min: ", act_delay_min)
        #
        # print(f"overall_obs_min: ", overall_obs_min)
        # print(f"overall_obs_max: ", overall_obs_max)
        #
        # print(f"overall_act_min: ", overall_act_min)
        # print(f"overall_act_max: ", overall_act_max)
        #
        # print(f"rel_obs_min: ", rel_obs_min)
        # print(f"rel_obs_max: ", rel_obs_max)
        #
        # print(f"rel_act_min: ", rel_act_min)
        # print(f"rel_act_max: ", rel_act_max)
        # Random choice to make uniform probability over interval range
        # min_or_max = random.choice([True, False])
        # mid = random.randint(min(interval_range), max(interval_range))
        #
        # if min_or_max:
        #     min_range = max(mid-1,0)
        #     max_range = random.randint(mid, max(interval_range))
        # else:
        #     max_range = min(mid+1,
        #     min_range = random.randint(min(interval_range), mid
