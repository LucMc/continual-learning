from collections import deque
from random import sample
import random
import itertools

import gymnasium as gym
from gymnasium.core import Env
from gymnasium.spaces import Tuple, Discrete, Box, Dict

import numpy as np
from typing import Literal


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
                Tuple(
                    [env.action_space]
                    * (obs_delay_range.stop + act_delay_range.stop - 1)
                ),  # action buffer
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

        self.t = 0
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
        self.t = -(
            self.obs_delay_range.stop + self.act_delay_range.stop
        )  # this is <= -2
        while self.t < 0:
            act = (
                self.action_space.sample()
                if self.initial_action is None
                else self.initial_action
            )
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
            m, r, term, trun, info = self.env.step(
                self.next_action
            )  # before receive_action (e.g. rtrl setting with 0 delays)
            d = term | trun
            kappa, beta = self.receive_action()
            self.cum_rew_actor += r
            self.done_signal_sent = d
            self.send_observation(
                (m, self.cum_rew_actor, term, trun, info, kappa, beta)
            )

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

        next_action_idx = next(
            i for i, t in enumerate(self.arrival_times_actions) if t <= self.t
        )

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

        alpha = next(
            i for i, t in enumerate(self.arrival_times_observations) if t <= self.t
        )

        m, r, term, trun, info, kappa, beta = self.past_observations[alpha]
        info.update(delay_mag=alpha + beta)
        return (
            (
                m.astype(np.float32),
                tuple(
                    itertools.islice(self.past_actions, 0, self.past_actions.maxlen - 1)
                ),
                alpha,
                kappa,
                beta,
            ),
            r,
            term,
            trun,
            info,
        )


class ContinualIntervalDelayWrapper(RandomDelayWrapper):
    def __init__(
        self,
        env,
        change_every: int = 10_000,
        obs_delay_range=range(0, 4),
        act_delay_range=range(0, 4),
        delay_type: Literal["random", "constant", "incremental"] = "incremental",
        **init_kwargs,
    ):
        self.init_kwargs = init_kwargs

        # Delay ranges
        self.delay_type = delay_type
        self.overall_act_delay_range = act_delay_range
        self.overall_obs_delay_range = obs_delay_range

        self.n_changes = 0

        # Determine initial ranges BEFORE calling super().__init__
        self.obs_delay_range = self.get_interval(self.overall_obs_delay_range)
        self.act_delay_range = self.get_interval(self.overall_act_delay_range)

        print(f"Initial obs delay is: {self.obs_delay_range}")
        print(f"Initial act delay is: {self.act_delay_range}")
        print(f"changing every: {change_every}")
        print(f"Delay type: {self.delay_type}")

        # Initialise RandomDelayWrapper with INITIAL ranges
        super().__init__(env, self.obs_delay_range, self.act_delay_range, **init_kwargs)

        # Target length for padding should match the space definition
        self.target_act_buf_tuple_len = (
            self.overall_obs_delay_range.stop
            + self.overall_act_delay_range.stop
            - 1
        )

        # Calculate the flattened size correctly (handle multi-dimensional actions)
        action_element_size = np.prod(self.env.action_space.shape) if self.env.action_space.shape else 1
        flat_act_buf_len = self.target_act_buf_tuple_len * action_element_size

        # Define the final observation space based on OVERALL ranges
        base_obs_shape = self.env.observation_space.shape # Get shape from underlying env
        if not base_obs_shape: # Handle scalar observation spaces
             base_obs_len = 1
        else:
             base_obs_len = np.prod(base_obs_shape)


        self.observation_space = Box(
            shape=(
                base_obs_len + flat_act_buf_len + 3, # base_obs + flat_padded_act_buf + alpha/kappa/beta
            ),
            low=-np.inf,
            high=np.inf,
            dtype=np.float32, # Ensure consistent dtype
        )
        print("Final observation_space shape:", self.observation_space.shape)
        print("action buffer tuple length:", self.target_act_buf_tuple_len)
        print("action buffer length (flat):", flat_act_buf_len)

        # Other
        self.time_steps = 0
        self.change_every = change_every

    def get_interval(self, interval_range):
        if self.delay_type == "random":
            assert len(interval_range) >= 2, (
                "obs delay must be sufficiently large to enable sub-interval"
            )
            nums = []
            while len(set(nums)) != 2:  # Pick 2 different numbers
                nums = [
                    random.randint(min(interval_range), max(interval_range)),
                    random.randint(min(interval_range), max(interval_range)),
                ]
            return range(min(nums), max(nums)+1) # Ensure range includes max num chosen

        elif self.delay_type == "constant":
            delay = random.choice(interval_range)
            return range(delay, delay + 1)

        elif self.delay_type == "incremental":
            # Ensure delay stays within overall bounds
            max_delay = interval_range.stop -1
            current_delay = min(interval_range.start + self.n_changes, max_delay)
            # Range is [current_delay, current_delay + 1) -> just current_delay
            return range(current_delay, current_delay + 1)
        else:
            raise ValueError(f"Invalid delay type: {self.delay_type}")


    def get_padded_act_buf(self, recieved_obs: tuple):
        # recieved_obs[1] is the tuple of past actions returned by super().receive_observation
        # Its length is determined by the parent's maxlen, based on INITIAL ranges.
        actual_actions_tuple = recieved_obs[1]
        actual_len = len(actual_actions_tuple)

        # Calculate padding needed to reach the target length defined by OVERALL ranges
        padding_needed = self.target_act_buf_tuple_len - actual_len

        if padding_needed < 0:
            # This implies the initial ranges were larger than the overall ranges, shouldn't happen with correct setup
            print(f"Warning: Padding needed is negative ({padding_needed}). Target len: {self.target_act_buf_tuple_len}, Actual len: {actual_len}. Clipping to zero.")
            padding_needed = 0
            # Or raise error: raise ValueError("Initial ranges exceed overall ranges, padding calculation failed.")

        # Create padding elements with correct shape and dtype
        # Use np.zeros matching the action space element shape and dtype
        padding_element = np.zeros(self.env.action_space.shape, dtype=self.env.action_space.dtype)
        back_padding = tuple(padding_element for _ in range(padding_needed))

        # Combine actual actions and padding
        padded_act_buf_tuple = actual_actions_tuple + back_padding

        # Flatten the buffer correctly
        # Use hstack for simple Box spaces, might need reshape/concatenate for more complex spaces
        # Ensure consistent dtype before flattening/stacking if actions aren't float32
        try:
            # Convert elements to float32 numpy arrays before stacking
            elements_to_stack = [np.array(a, dtype=np.float32).flatten() for a in padded_act_buf_tuple]
            if not elements_to_stack: # Handle case where buffer might be empty
                 act_buf_flat = np.array([], dtype=np.float32)
            else:
                 act_buf_flat = np.concatenate(elements_to_stack)
                 # Alternative using hstack if elements are already 1D numpy arrays
                 # act_buf_flat = np.hstack([np.array(a, dtype=np.float32) for a in padded_act_buf_tuple]).astype(np.float32)

        except ValueError as e:
            print("Error during action buffer flattening:")
            print("Action space shape:", self.env.action_space.shape)
            print("Padded action tuple length:", len(padded_act_buf_tuple))
            # print("Elements:", [a.shape for a in padded_act_buf_tuple]) # Debug shapes
            raise e


        # Optional check: Verify flattened shape (good for debugging)
        # action_element_size = np.prod(self.env.action_space.shape) if self.env.action_space.shape else 1
        # expected_flat_len = self.target_act_buf_tuple_len * action_element_size
        # if act_buf_flat.shape[0] != expected_flat_len:
        #      print(f"Warning: Padded action buffer has wrong flat shape. Got {act_buf_flat.shape[0]}, expected {expected_flat_len}")
             # raise ValueError("Flattened action buffer shape mismatch")

        return act_buf_flat.astype(np.float32) # Ensure final dtype

    def _format_observation(self, recieved_obs):
        """Helper function to format observation for both step and reset."""
        # recieved_obs = (m, past_actions_tuple, alpha, kappa, beta)
        base_obs = recieved_obs[0].astype(np.float32).flatten() # Flatten base obs too for consistency
        padded_act_buf = self.get_padded_act_buf(recieved_obs)
        delay_info = np.array(recieved_obs[2:], dtype=np.float32) # alpha, kappa, beta

        final_obs = np.concatenate((base_obs, padded_act_buf, delay_info))

        # Final check on shape
        if final_obs.shape != self.observation_space.shape:
             raise ValueError(f"Final observation shape mismatch! Got {final_obs.shape}, expected {self.observation_space.shape}. "
                              f"Base: {base_obs.shape}, PadAct: {padded_act_buf.shape}, Delay: {delay_info.shape}")

        return final_obs

    def step(self, action, **kwargs):
        self.time_steps += 1

        # Call parent step
        recieved_obs_tuple, r, term, trun, info = super().step(action)
        # recieved_obs_tuple format: (m, past_actions_tuple, alpha, kappa, beta)

        # Format the observation using the helper
        final_obs = self._format_observation(recieved_obs_tuple)

        return final_obs, r, term, trun, info # Return consistently shaped observation

    def reset(self, seed=None, **reset_kwargs):
        """ Probably don't need all of this but I'm just being cautious.
        Perhaps remove one of the super().reset()
        todo: try to remove first line here
        """
        
        super().reset(seed=seed, **reset_kwargs) # Call parent reset FIRST to handle its state/env reset

        if self.time_steps >= self.change_every:
            print("-" * 20)
            print(f"Changing delay interval at step {self.time_steps}")
            self.n_changes += 1
            self.time_steps = 0 # Reset counter AFTER check

            # Reset vars from RandomDelayWrapper
            self.t = 0
            self.done_signal_sent = False
            self.next_action = None
            self.cum_rew_actor = 0.0
            self.cum_rew_brain = 0.0
            self.prev_action_idx = 0

            # Recalculate CURRENT delay ranges
            self.obs_delay_range = self.get_interval(self.overall_obs_delay_range)
            self.act_delay_range = self.get_interval(self.overall_act_delay_range)

            print(f"New obs delay range: {self.obs_delay_range}")
            print(f"New act delay range: {self.act_delay_range}")
            print("-" * 20)

            # Re-clear buffers (parent's reset already does some clearing/refilling)
            # Clearing here ensures consistency if parent doesn't clear everything needed.
            self.past_actions = deque(maxlen=self.obs_delay_range.stop + self.act_delay_range.stop)
            self.past_observations = deque(maxlen=self.obs_delay_range.stop)
            self.arrival_times_actions = deque(maxlen=self.act_delay_range.stop)
            self.arrival_times_observations = deque(maxlen=self.obs_delay_range.stop)

            # Call parent reset AGAIN after updating ranges and clearing buffers
            # to ensure it refills buffers correctly based on the *new* logic/ranges
            # for sampling initial actions/observations if its reset logic uses them.
            recieved_obs_tuple, reset_info = super().reset(seed=seed, **reset_kwargs)

        else:
             recieved_obs_tuple, reset_info = super().reset(seed=seed, **reset_kwargs)

        final_obs = self._format_observation(recieved_obs_tuple)

        return final_obs, reset_info # Return consistently shaped observation


"""
TODO:
 - Get rid of RTRL framework stuff and optimise RandomDelayWrapper etc
 - For some reason RandomDelayWrapper only uses end delay when making buffer
   For our purposes we usually start delays at zero so probably isn't worth the optimisation
   Meaning delays of (18,20) have loads of unnessary padding
   # padded_act_buf = front_padding + recieved_obs[1] + back_padding

backup snippets:
self.__init__(
    env=self.wrapped_env,
    change_every=self.change_every,
    obs_delay_range=self.overall_obs_delay_range,
    act_delay_range=self.overall_act_delay_range,
    **self.init_kwargs,
)
front_padding = tuple(
    np.zeros(self.env.action_space.shape
self.obs_delay_range = self.get_interval(self.overall_obs_delay_range)
self.act_delay_range = self.get_interval(self.overall_act_delay_range)

self.past_actions.clear()
self.past_observations.clear()
self.arrival_times_actions.clear()
self.arrival_times_observations.clear()

super().obs_delay_range = self.obs_delay_range
super().act_delay_range = self.act_delay_range
self.time_steps = 0
"""
