import random

import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Dict, Discrete, Tuple


class NoneWrapper(gym.Wrapper):
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


class UnseenRandomDelayWrapper(RandomDelayWrapper):
    """
    Wrapper that translates the RandomDelayWrapper back to the usual RL setting
    Use this wrapper to see what happens to vanilla RL algorithms facing random delays
    """

    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        self.observation_space = env.unwrapped.observation_space

    def reset(self, **kwargs):
        t, reset_info = super().reset(
            **kwargs
        )  # t: (m, tuple(self.past_actions), alpha, kappa, beta)
        return t[0], reset_info

    def step(self, action):
        t, *aux = super().step(action)  # t: (m, tuple(self.past_actions), alpha, kappa, beta)
        return (t[0], *aux)


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
            self.new_obs_shape = env.observation_space.shape[0] + (
                env.action_space.shape[0] * self.delay
            )
            self.observation_space = Box(
                low=-np.inf, high=np.inf, shape=(self.new_obs_shape,), dtype=np.float32
            )

    def reset(self, **kwargs):
        t, reset_info = super().reset(
            **kwargs
        )  # t: (m, tuple(self.past_actions), alpha, kappa, beta)
        if self.HER:
            aug_state = np.append(t[0]["observation"], t[1])
            t[0]["observation"] = aug_state
            aug_state = t[0]
        else:
            aug_state = np.append(t[0], t[1]).astype(
                np.float32
            )  # Combine action buffer and state

        return aug_state, reset_info

    # Make this more efficient
    def step(self, action):
        t, *aux = super().step(action)  # t: (m, tuple(self.past_actions), alpha, kappa, beta)
        if self.HER:
            # print("step",t[0]['observation'].shape)
            if (
                t[0]["observation"].shape[0] == self.default_observation_space.shape[0]
            ):  # Fix for any value later
                # print("Fixing bug")
                aug_state = np.append(t[0]["observation"], t[1])
                t[0]["observation"] = aug_state

            return (t[0], *aux)

        else:
            aug_state = np.append(t[0], t[1])  # Combine action buffer and state

        return (aug_state, *aux)


class ContinualRandomIntervalDelayWrapper(RandomDelayWrapper):
    def __init__(
        self,
        env,
        change_every: int = 10_000,
        obs_delay_range=range(0, 4),
        act_delay_range=range(0, 4),
        interval_emb_type="two_hot",  # "twohot", "float", "scalar" | None
        delay_emb_type="one_hot",  # "one_hot", "float", "scalar", | None >> "multi_hot"?[0,0,1,2,0,0]?) transition delay embedding
        give_kappa=False,
        output="standard",  # "dcac", "standard" >> Extra outputs for dcac or standard gym outputs
        # interval_aware=True, # Include interval emb in state
        **init_kwargs,
    ):
        # Constants
        self.AVAILABLE_MODES = ["continual", "multi-task", "fixed"]
        self.AVAILABLE_OUTPUTS = ["dcac", "standard"]
        self.AVAILABLE_DELAY_EMB_TYPES = ["one_hot", "float", "scalar", None]
        self.AVAILABLE_INTERVAL_EMB_TYPES = ["two_hot", "float", "scalar", None]

        self.init_kwargs = init_kwargs
        # Multi-choice settings
        self.output = output.lower()
        if interval_emb_type:
            self.interval_emb_type = interval_emb_type.lower()
        if interval_emb_type:
            self.delay_emb_type = delay_emb_type.lower()

        # Checks
        assert self.output in self.AVAILABLE_OUTPUTS, (
            f"{self.output} not in ( {' | '.join(self.AVAILABLE_OUTPUTS)} )"
        )
        assert self.delay_emb_type in self.AVAILABLE_DELAY_EMB_TYPES, (
            f"{self.delay_emb_types} not in ( {' | '.join(self.AVAILABLE_DELAY_EMB_TYPES)} )"
        )
        assert self.interval_emb_type in self.AVAILABLE_INTERVAL_EMB_TYPES, (
            f"{self.interval_emb_types} not in ( {' | '.join(self.AVAILABLE_INTERVAL_EMB_TYPES)} )"
        )
        assert len(obs_delay_range) >= 2, (
            "obs delay must be sufficiently large to enable sub-interval"
        )
        assert len(act_delay_range) >= 2, (
            "act delay must be sufficiently large to enable sub-interval"
        )
        assert self.output == "standard" or self.delay_emb_type == "one_hot", (
            "dcac always uses one_hot embedding internally"
        )

        # Delay ranges
        self.overall_act_delay_range = act_delay_range  # Now in RD Wrapper
        self.overall_obs_delay_range = obs_delay_range

        self.obs_delay_range = self.get_interval(obs_delay_range)
        self.act_delay_range = self.get_interval(act_delay_range)
        print(f"obs delay is: {self.obs_delay_range}")
        print(f"act delay is: {self.act_delay_range}")
        print(f"changing every: {change_every}")

        self.n_obs_delays = len(self.overall_obs_delay_range)
        self.n_act_delays = len(self.overall_act_delay_range)

        # Initialise RandomDelayWrapper
        super().__init__(env, self.obs_delay_range, self.act_delay_range, **init_kwargs)

        # Other
        self.give_kappa = give_kappa
        self.time_steps = 0
        self.change_every = change_every

        # Need to be sure no new envs are created after testing etc...
        # Fix observation space for added interval information
        match self.interval_emb_type:
            case "two_hot":
                int_emb_len = (
                    len(self.overall_act_delay_range) + len(self.overall_obs_delay_range) - 2
                )
            case "float":
                int_emb_len = 4
            case "scalar":
                int_emb_len = 4
            case None:
                int_emb_len = 0

        match self.delay_emb_type:
            case "one_hot":
                delay_emb_len = (
                    len(self.overall_act_delay_range) + len(self.overall_obs_delay_range) - 2
                )
            case "float":
                delay_emb_len = 4
            case "scalar":
                delay_emb_len = 4
            case None:
                delay_emb_len = 0

        if self.output == "dcac":  # DCAC has it's own methods for delay embeddings
            self.observation_space = Tuple(
                (
                    # self.env.observation_space,  # most recent observation
                    Box(
                        shape=(self.observation_space[0].shape[0] + int_emb_len,),
                        low=-np.inf,
                        high=np.inf,
                        dtype=np.float32,
                    ),
                    Tuple(
                        [self.env.action_space]
                        * (
                            self.overall_obs_delay_range.stop
                            + self.overall_act_delay_range.stop
                            - 1
                        )
                    ),  # action buffer
                    Discrete(self.obs_delay_range.stop),  # observation delay int64
                    Discrete(self.act_delay_range.stop),  # action delay int64
                )
            )

        elif self.output == "standard":
            # new_obs = np.concatenate((obs[0], obs[1], obs_emb, act_emb), axis=0)
            if self.give_kappa:
                pass
            else:
                act_buf_len = np.sum(
                    [self.env.action_space.shape]
                    * (
                        self.overall_obs_delay_range.stop
                        + self.overall_act_delay_range.stop
                        - 1
                    )
                )
                obs_emb_len = (
                    self.n_obs_delays + 2
                )  # add 2? Scalar for now TODO: Implement other delay emb types
                act_emb_len = self.n_act_delays
                extra_dims = act_buf_len + obs_emb_len + act_emb_len + int_emb_len
                self.observation_space = Box(
                    shape=(self.observation_space[0].shape[0] + extra_dims,),
                    low=-np.inf,
                    high=np.inf,
                    dtype=np.float32,
                )

    def get_interval(self, interval_range):
        # print("interval_range:\n", interval_range)
        assert len(interval_range) > 2, (
            f"Not enough range between delay max and min, start: {interval_range.start} stop: {interval_range.stop}"
        )
        nums = []
        iters = 0

        while len(set(nums)) != 2:  # Pick 2 different numbers
            nums = [
                random.randint(min(interval_range), max(interval_range)),
                random.randint(min(interval_range), max(interval_range)),
            ]
            iters += 1

        return range(min(nums), max(nums))

    def get_delay_interval_embedding(self) -> tuple:
        # min and maxs for given interval
        rel_obs_min = self.obs_delay_range.start - self.overall_obs_delay_range.start
        rel_obs_max = self.obs_delay_range.stop - self.overall_obs_delay_range.start
        rel_act_min = self.act_delay_range.start - self.overall_act_delay_range.start
        rel_act_max = self.act_delay_range.stop - self.overall_act_delay_range.start

        if self.interval_emb_type == "two_hot":  # provides the delays as a two hot embedding
            two_hot_obs = np.zeros(
                self.n_obs_delays - 1
            )  # -1 since we cant get max only min when using range()
            two_hot_act = np.zeros(self.n_act_delays - 1)

            two_hot_obs[[rel_obs_min, rel_obs_max - 1]] = 1
            two_hot_act[[rel_act_min, rel_act_max - 1]] = 1

            return (two_hot_obs, two_hot_act)

        elif self.interval_emb_type == "scalar":  # provides the range simply as scalars
            return (
                (self.obs_delay_range.start, self.obs_delay_range.stop + 1),
                (self.act_delay_range.start, self.act_delay_range.stop + 1),
            )

        elif (
            self.interval_emb_type == "float"
        ):  # normalise over range, show delay as percentage of max range
            obs_min_float = rel_obs_min / (self.n_obs_delays)
            obs_max_float = rel_obs_max / (self.n_obs_delays)
            act_min_float = rel_act_min / (self.n_act_delays)
            act_max_float = rel_act_max / (self.n_act_delays)

            return ((obs_min_float, obs_max_float), (act_min_float, act_max_float))
        else:
            raise Exception(f"Unsupported embedding type: {self.interval_emb_type}")

    def format_obs(self, obs):
        # Only for standard RL Gym env, not DCAC
        action_buffer = obs[
            1
        ]  # Can be variable length based on delay interval so needs padding
        act_buf_max = self.overall_act_delay_range.stop + self.overall_obs_delay_range.stop - 1
        action_buffer += (0,) * (act_buf_max - len(action_buffer))

        # get delays
        obs_delay = obs[2]
        act_delay = obs[3]
        kappa_delay = obs[4]

        if self.delay_emb_type == "one_hot":
            obs_emb = np.zeros(
                self.n_obs_delays + 1
            )  # Should have similar to get delay interval embedding
            act_emb = np.zeros(self.n_act_delays + 1)
            kappa_emb = np.zeros(self.n_obs_delays + 1)
            obs_emb[obs_delay - self.overall_obs_delay_range.start] = (
                1.0  # Error when delay=array size, is this index -1 or array not large enough?
            )
            act_emb[act_delay - self.overall_act_delay_range.start] = 1.0
            kappa_emb[kappa_delay - self.overall_obs_delay_range.start] = 1.0

        elif self.delay_emb_type == "scalar":
            obs_emb = [obs_delay]
            act_emb = [act_delay]
            kappa_emb = [kappa_delay]

        elif self.delay_emb_type == "float":
            # Provide as float, relative to the overall interval

            obs_emb = [(obs_delay - self.overall_obs_delay_range.start) / self.n_obs_delays]
            act_emb = [(act_delay - self.overall_act_delay_range.start) / self.n_act_delays]
            kappa_emb = [
                (kappa_delay - self.overall_obs_delay_range.start) / self.n_obs_delays
            ]
        else:
            raise ValueError("Invalid delay_emb_type", self.delay_emb_type)

        # print(obs[0].shape, obs[1][0].shape, obs_emb.shape, act_emb.shape, kappa_emb.shape)
        if self.give_kappa:
            new_obs = np.concatenate((obs[0], obs[1], obs_emb, act_emb, kappa_emb), axis=0)
        else:
            new_obs = np.concatenate(
                (obs[0], np.concatenate(obs[1]), obs_emb, act_emb), axis=0
            )

        return new_obs

    def get_padded_act_buf(self, recieved_obs: tuple):
        n_padding_needed = (
            self.overall_act_delay_range.stop
            + self.overall_obs_delay_range.stop
            - len(recieved_obs[1])
            - 1
        )
        padding = tuple(
            np.zeros_like([recieved_obs[1][-1]]).squeeze() for _ in range(n_padding_needed)
        )
        padded_act_buf = padding + recieved_obs[1]

        if self.output == "dcac":
            assert len(padded_act_buf) == len(self.observation_space[1])
        return padded_act_buf

    def get_int_emb_obs(self, recieved_obs, padded_act_buf):
        if self.interval_emb_type:  # if interval aware, i.e. interval embedding != None
            # Add int embedding info
            # Could move this into reset for more efficiency since it stays the same whole episode
            if self.interval_emb_type == "two_hot":
                obs_emb, act_emb = self.get_delay_interval_embedding()
                recieved_obs = (
                    np.concatenate(
                        (recieved_obs[0], obs_emb, act_emb), axis=0, dtype=np.float32
                    ),
                    padded_act_buf,
                    *recieved_obs[2:],
                )
            else:
                (obs_min, obs_max), (act_min, act_max) = self.get_delay_interval_embedding()
                recieved_obs = (
                    np.concatenate(
                        (
                            recieved_obs[0],
                            np.expand_dims(obs_min, axis=-1),
                            np.expand_dims(obs_max, axis=-1),
                            np.expand_dims(act_min, axis=-1),
                            np.expand_dims(act_max, axis=-1),
                        ),
                        axis=0,
                        dtype=np.float32,
                    ),
                    padded_act_buf,
                    *recieved_obs[2:],
                )  # Add delay interval to obs
        else:
            recieved_obs = recieved_obs[0], padded_act_buf, *recieved_obs[2:]

        return recieved_obs

    def step(self, action, **kwargs):
        self.time_steps += 1

        recieved_obs, *aux = super().step(
            action
        )  # t: (m, tuple(self.past_actions), alpha, kappa, beta)
        padded_act_buf = self.get_padded_act_buf(recieved_obs)

        recieved_obs = self.get_int_emb_obs(recieved_obs, padded_act_buf)

        obs = (
            recieved_obs if self.output == "dcac" else self.format_obs(recieved_obs)
        )  # one hot encoding of the transition delay
        # print("RB SHAPE cont\n", recieved_obs[1][0].shape, len(recieved_obs[1]))
        # assert len(obs) == 5
        return (obs, *aux)  # aux=rew,term,trun,info

    def reset(self, seed=None, **reset_kwargs):
        """
        observation_space:
        Tuple((
            obs_space,  # most recent observation
            Tuple([act_space] * (obs_delay_range.stop + act_delay_range.stop)),  # action buffer
            Discrete(obs_delay_range.stop),  # observation delay int64
            Discrete(act_delay_range.stop),  # action delay int64
        ))
        """

        # print(f"obs delay interval", self.obs_delay_range)
        # print(f"act delay interval", self.act_delay_range)

        # Comment out changing on reset for now
        if self.time_steps >= self.change_every:
            # self.obs_delay_range = self.get_interval(self.overall_obs_delay_range)
            # self.act_delay_range = self.get_interval(self.overall_act_delay_range)

            self.__init__(
                env=self.wrapped_env,
                change_every=self.change_every,
                obs_delay_range=self.overall_obs_delay_range,
                act_delay_range=self.overall_act_delay_range,
                interval_emb_type=self.interval_emb_type,
                delay_emb_type=self.delay_emb_type,
                give_kappa=self.give_kappa,
                output=self.output,
                **self.init_kwargs,
            )
            print("time steps after reset", self.time_steps)

        recieved_obs, reset_info = super().reset(**reset_kwargs)
        padded_act_buf = self.get_padded_act_buf(recieved_obs)
        recieved_obs = self.get_int_emb_obs(recieved_obs, padded_act_buf)

        # if self.interval_emb_type: # if interval aware, i.e. interval embedding != None
        #     # Add int embedding info
        #     (obs_min, obs_max), (act_min, act_max) = self.get_delay_interval_embedding()
        #     recieved_obs = np.concatenate((recieved_obs[0],
        #                                    np.expand_dims(obs_min, axis=-1),
        #                                    np.expand_dims(obskkk_max, axis=-1),
        #                                    np.expand_dims(act_min, axis=-1),
        #                                    np.expand_dims(act_max, axis=-1),), axis=0, dtype=np.float32), padded_act_buf, *recieved_obs[2:] # Add delay interval to obs
        # else:
        #     recieved_obs = recieved_obs[0], padded_act_buf, *recieved_obs[2:]

        obs = (
            recieved_obs if self.output == "dcac" else self.format_obs(recieved_obs)
        )  # one hot encoding of the transition delay
        # assert len(obs) == 5
        return obs, {}
