"""Tests for SAC and BRO implementation."""

import jax
import jax.numpy as jnp
import pytest

from continual_learning.configs.models import MLPConfig
from continual_learning.configs.optim import AdamConfig
from continual_learning.configs.rl import PolicyNetworkConfig, QNetworkConfig, SACConfig
from continual_learning.models import get_model_cls
from continual_learning.models.rl import QNetwork, TanhPolicy
from continual_learning.trainers.sac import SAC, SACTrainState
from continual_learning.types import Activation, StdType
from continual_learning.utils.replay_buffer import ReplayBuffer, ReplayBufferState, ReplayBatch


class TestReplayBuffer:
    """Tests for replay buffer."""

    def test_init(self):
        """Test buffer initialization."""
        buffer = ReplayBuffer(capacity=1000, obs_dim=4, action_dim=2)
        state = buffer.init()

        assert state.observations.shape == (1000, 4)
        assert state.actions.shape == (1000, 2)
        assert state.rewards.shape == (1000, 1)
        assert state.next_observations.shape == (1000, 4)
        assert state.dones.shape == (1000, 1)
        assert state.position == 0
        assert state.size == 0

    def test_add_single(self):
        """Test adding single transitions."""
        buffer = ReplayBuffer(capacity=100, obs_dim=4, action_dim=2)
        state = buffer.init()

        obs = jnp.ones((1, 4))
        action = jnp.ones((1, 2))
        reward = jnp.array([[1.0]])
        next_obs = jnp.ones((1, 4)) * 2
        done = jnp.array([[False]])

        state = ReplayBuffer.add(state, obs, action, reward, next_obs, done)

        assert state.size == 1
        assert state.position == 1

    def test_add_batch(self):
        """Test adding batch of transitions."""
        buffer = ReplayBuffer(capacity=100, obs_dim=4, action_dim=2)
        state = buffer.init()

        batch_size = 10
        obs = jnp.ones((batch_size, 4))
        action = jnp.ones((batch_size, 2))
        reward = jnp.ones((batch_size, 1))
        next_obs = jnp.ones((batch_size, 4))
        done = jnp.zeros((batch_size, 1), dtype=bool)

        state = ReplayBuffer.add(state, obs, action, reward, next_obs, done)

        assert state.size == batch_size
        assert state.position == batch_size

    def test_circular_buffer(self):
        """Test circular buffer wrapping."""
        buffer = ReplayBuffer(capacity=10, obs_dim=2, action_dim=1)
        state = buffer.init()

        # Add 15 transitions to a buffer of capacity 10
        for i in range(15):
            obs = jnp.ones((1, 2)) * i
            action = jnp.ones((1, 1))
            reward = jnp.array([[float(i)]])
            next_obs = jnp.ones((1, 2)) * (i + 1)
            done = jnp.array([[False]])
            state = ReplayBuffer.add(state, obs, action, reward, next_obs, done)

        # Buffer should be full
        assert state.size == 10
        # Position should wrap around
        assert state.position == 5

    def test_sample(self):
        """Test sampling from buffer."""
        buffer = ReplayBuffer(capacity=100, obs_dim=4, action_dim=2)
        state = buffer.init()

        # Add some data
        for i in range(50):
            obs = jnp.ones((1, 4)) * i
            action = jnp.ones((1, 2))
            reward = jnp.array([[float(i)]])
            next_obs = jnp.ones((1, 4)) * (i + 1)
            done = jnp.array([[False]])
            state = ReplayBuffer.add(state, obs, action, reward, next_obs, done)

        # Sample batch
        key = jax.random.PRNGKey(0)
        batch = ReplayBuffer.sample(state, key, batch_size=16)

        assert isinstance(batch, ReplayBatch)
        assert batch.observations.shape == (16, 4)
        assert batch.actions.shape == (16, 2)
        assert batch.rewards.shape == (16, 1)
        assert batch.next_observations.shape == (16, 4)
        assert batch.dones.shape == (16, 1)


class TestQNetwork:
    """Tests for Q-network."""

    def test_forward(self):
        """Test Q-network forward pass."""
        config = QNetworkConfig(
            optimizer=AdamConfig(learning_rate=3e-4),
            network=MLPConfig(
                num_layers=2,
                hidden_size=64,
                output_size=1,
                activation_fn=Activation.ReLU,
            ),
        )

        network_cls = get_model_cls(config.network)
        qnet = QNetwork(network_cls, config)

        key = jax.random.PRNGKey(0)
        obs = jnp.ones((32, 10))  # batch of 32, obs_dim=10
        action = jnp.ones((32, 4))  # action_dim=4

        params = qnet.init(key, obs, action, training=False)
        q1, q2 = qnet.apply(params, obs, action, training=False)

        assert q1.shape == (32, 1)
        assert q2.shape == (32, 1)
        # Q1 and Q2 should be different (separate networks)
        assert not jnp.allclose(q1, q2)


class TestTanhPolicy:
    """Tests for TanhPolicy."""

    def test_forward(self):
        """Test policy forward pass."""
        config = PolicyNetworkConfig(
            optimizer=AdamConfig(learning_rate=3e-4),
            network=MLPConfig(
                num_layers=2,
                hidden_size=64,
                output_size=4,  # action_dim
                activation_fn=Activation.ReLU,
            ),
            std_type=StdType.MLP_HEAD,
        )

        network_cls = get_model_cls(config.network)
        policy = TanhPolicy(network_cls, config)

        key = jax.random.PRNGKey(0)
        obs = jnp.ones((32, 10))  # batch of 32, obs_dim=10

        params = policy.init(key, obs, training=False)
        dist = policy.apply(params, obs, training=False)

        # Sample actions
        action_key = jax.random.PRNGKey(1)
        actions = dist.sample(seed=action_key)

        assert actions.shape == (32, 4)
        # Actions should be bounded by tanh: [-1, 1]
        assert jnp.all(actions >= -1.0)
        assert jnp.all(actions <= 1.0)


class TestSAC:
    """Tests for SAC algorithm."""

    @pytest.fixture
    def sac_config(self):
        """Create SAC config for testing."""
        return SACConfig(
            actor_config=PolicyNetworkConfig(
                optimizer=AdamConfig(learning_rate=3e-4),
                network=MLPConfig(
                    num_layers=2,
                    hidden_size=64,
                    output_size=4,
                    activation_fn=Activation.ReLU,
                ),
                std_type=StdType.MLP_HEAD,
            ),
            critic_config=QNetworkConfig(
                optimizer=AdamConfig(learning_rate=3e-4),
                network=MLPConfig(
                    num_layers=2,
                    hidden_size=64,
                    output_size=1,
                    activation_fn=Activation.ReLU,
                ),
            ),
            gamma=0.99,
            tau=0.005,
            alpha=0.2,
            auto_entropy=True,
            buffer_size=1000,
            batch_size=32,
            learning_starts=100,
            replay_ratio=1,
        )

    def test_init_state(self, sac_config):
        """Test SAC state initialization."""
        key = jax.random.PRNGKey(0)
        state = SAC.init_state(
            key=key,
            obs_dim=10,
            action_dim=4,
            cfg=sac_config,
        )

        assert isinstance(state, SACTrainState)
        assert state.actor is not None
        assert state.critic is not None
        assert state.target_critic_params is not None
        assert state.log_alpha is not None

    def test_update_step(self, sac_config):
        """Test single SAC update step."""
        key = jax.random.PRNGKey(0)
        state = SAC.init_state(
            key=key,
            obs_dim=10,
            action_dim=4,
            cfg=sac_config,
        )

        # Create dummy batch
        batch = ReplayBatch(
            observations=jnp.ones((32, 10)),
            actions=jnp.ones((32, 4)) * 0.5,
            rewards=jnp.ones((32, 1)),
            next_observations=jnp.ones((32, 10)),
            dones=jnp.zeros((32, 1), dtype=bool),
        )

        # Run update
        target_entropy = -4.0  # -action_dim
        new_state, logs = SAC.update(state, batch, sac_config, target_entropy)

        assert isinstance(new_state, SACTrainState)
        assert "critic/loss" in logs
        assert "actor/loss" in logs
        assert "alpha/value" in logs

    def test_soft_target_update(self, sac_config):
        """Test soft target update."""
        key = jax.random.PRNGKey(0)
        state = SAC.init_state(
            key=key,
            obs_dim=10,
            action_dim=4,
            cfg=sac_config,
        )

        # Store original target params
        original_target = jax.tree.map(lambda x: x.copy(), state.target_critic_params)

        # Modify critic params
        new_critic_params = jax.tree.map(
            lambda x: x + 1.0, state.critic.params
        )
        state = state._replace(
            critic=state.critic.replace(params=new_critic_params)
        )

        # Apply soft update
        tau = 0.5
        new_state = SAC.soft_update_target(state, tau)

        # Check that target params moved towards critic params
        def check_interpolation(target, orig, new):
            expected = tau * new + (1 - tau) * orig
            assert jnp.allclose(target, expected)

        jax.tree.map(
            check_interpolation,
            new_state.target_critic_params,
            original_target,
            new_critic_params,
        )


class TestBROLearner:
    """Tests for full BRO algorithm learner."""

    def test_init(self):
        """Test BRO learner initialization."""
        from continual_learning.trainers.bro_learner import BROLearner, BROConfig

        cfg = BROConfig(hidden_dims=64, depth=1, n_quantiles=10)
        learner = BROLearner(seed=0, obs_dim=10, action_dim=4, cfg=cfg)

        assert learner.target_entropy == -2.0  # -action_dim / 2
        assert learner.state is not None
        assert learner.state.step == 1

    def test_sample_actions_optimistic(self):
        """Test action sampling from optimistic actor."""
        from continual_learning.trainers.bro_learner import BROLearner, BROConfig

        cfg = BROConfig(hidden_dims=64, depth=1, n_quantiles=10)
        learner = BROLearner(seed=0, obs_dim=10, action_dim=4, cfg=cfg)

        obs = jnp.zeros((1, 10))
        actions = learner.sample_actions(obs, use_optimistic=True)

        assert actions.shape == (1, 4)
        assert jnp.all(actions >= -1.0)
        assert jnp.all(actions <= 1.0)

    def test_sample_actions_conservative(self):
        """Test action sampling from conservative actor."""
        from continual_learning.trainers.bro_learner import BROLearner, BROConfig

        cfg = BROConfig(hidden_dims=64, depth=1, n_quantiles=10)
        learner = BROLearner(seed=0, obs_dim=10, action_dim=4, cfg=cfg)

        obs = jnp.zeros((1, 10))
        actions = learner.sample_actions(obs, use_optimistic=False)

        assert actions.shape == (1, 4)
        assert jnp.all(actions >= -1.0)
        assert jnp.all(actions <= 1.0)

    def test_update(self):
        """Test BRO update step."""
        from continual_learning.trainers.bro_learner import BROLearner, BROConfig
        import numpy as np

        cfg = BROConfig(hidden_dims=64, depth=1, n_quantiles=10)
        learner = BROLearner(seed=0, obs_dim=10, action_dim=4, cfg=cfg)

        # Create dummy batch
        batch_size = 32
        observations = jnp.ones((batch_size, 10))
        actions = jnp.ones((batch_size, 4)) * 0.5
        rewards = jnp.ones((batch_size,))
        next_observations = jnp.ones((batch_size, 10))
        dones = jnp.zeros((batch_size,))

        logs = learner.update(
            observations=observations,
            actions=actions,
            rewards=rewards,
            next_observations=next_observations,
            dones=dones,
            env_step=0,
        )

        # Check all expected metrics are present
        assert "critic/loss" in logs
        assert "actor/loss" in logs
        assert "actor_o/loss" in logs
        assert "temperature/value" in logs
        assert "optimism/value" in logs
        assert "regularizer/kl_weight" in logs

        # Check no NaN
        for key, val in logs.items():
            assert not np.isnan(float(val)), f"{key} is NaN"

    def test_reset(self):
        """Test network reset."""
        from continual_learning.trainers.bro_learner import BROLearner, BROConfig

        cfg = BROConfig(hidden_dims=64, depth=1, n_quantiles=10)
        learner = BROLearner(seed=0, obs_dim=10, action_dim=4, cfg=cfg)

        # Store original params
        original_actor_params = jax.tree.map(
            lambda x: x.copy(), learner.state.actor_params
        )

        # Modify params via update
        observations = jnp.ones((32, 10))
        actions = jnp.ones((32, 4)) * 0.5
        rewards = jnp.ones((32,))
        next_observations = jnp.ones((32, 10))
        dones = jnp.zeros((32,))

        learner.update(observations, actions, rewards, next_observations, dones, 0)

        # Reset
        learner.reset()

        # Params should be back to original (same seed)
        def check_equal(a, b):
            assert jnp.allclose(a, b)

        jax.tree.map(check_equal, learner.state.actor_params, original_actor_params)


class TestBroNet:
    """Tests for BroNet architecture."""

    def test_forward(self):
        """Test BroNet forward pass."""
        from continual_learning.models.rl import BroNet

        net = BroNet(hidden_dims=64, depth=1)

        key = jax.random.PRNGKey(0)
        x = jnp.ones((32, 10))

        params = net.init(key, x)
        output = net.apply(params, x)

        assert output.shape == (32, 64)

    def test_residual_connection(self):
        """Test that residual connections work (depth > 0 should not be zero)."""
        from continual_learning.models.rl import BroNet

        net = BroNet(hidden_dims=64, depth=1)

        key = jax.random.PRNGKey(0)
        x = jnp.ones((32, 10))

        params = net.init(key, x)
        output = net.apply(params, x)

        # Output should not be all zeros due to residual
        assert not jnp.allclose(output, jnp.zeros_like(output))


class TestDistributionalCritic:
    """Tests for distributional critic."""

    def test_forward(self):
        """Test distributional critic forward pass."""
        from continual_learning.models.rl import BRODistributionalCritic

        critic = BRODistributionalCritic(n_quantiles=10, hidden_dims=64, depth=1)

        key = jax.random.PRNGKey(0)
        obs = jnp.ones((32, 10))
        action = jnp.ones((32, 4))

        params = critic.init(key, obs, action)
        q1, q2 = critic.apply(params, obs, action)

        assert q1.shape == (32, 10)  # n_quantiles
        assert q2.shape == (32, 10)

    def test_two_critics_different(self):
        """Test that Q1 and Q2 are different."""
        from continual_learning.models.rl import BRODistributionalCritic

        critic = BRODistributionalCritic(n_quantiles=10, hidden_dims=64, depth=1)

        key = jax.random.PRNGKey(0)
        obs = jnp.ones((32, 10))
        action = jnp.ones((32, 4))

        params = critic.init(key, obs, action)
        q1, q2 = critic.apply(params, obs, action)

        # Q1 and Q2 should be different
        assert not jnp.allclose(q1, q2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
