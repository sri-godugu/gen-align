"""Unit tests for GenAlign model components."""
import pytest
import jax
import jax.numpy as jnp

from src.models.base import ModelConfig, TransformerBackbone
from src.models.policy import PolicyModel, create_policy_train_state
from src.models.reward_model import RewardModel, create_reward_train_state
from src.models.value_model import ValueModel, create_value_train_state
from src.models.reference_model import ReferenceModel


@pytest.fixture
def small_config():
    return ModelConfig(
        vocab_size  = 256,
        max_seq_len = 32,
        hidden_size = 64,
        n_layers    = 2,
        n_heads     = 2,
        mlp_ratio   = 2.0,
    )


@pytest.fixture
def rng():
    return jax.random.PRNGKey(0)


# ── Backbone ──────────────────────────────────────────────────────────────────

class TestTransformerBackbone:
    def test_output_shape(self, small_config, rng):
        model  = TransformerBackbone(small_config)
        ids    = jnp.ones((2, 8), dtype=jnp.int32)
        params = model.init(rng, ids)
        out    = model.apply(params, ids)
        assert out.shape == (2, 8, small_config.hidden_size)

    def test_mask_does_not_crash(self, small_config, rng):
        model  = TransformerBackbone(small_config)
        ids    = jnp.ones((2, 8), dtype=jnp.int32)
        mask   = jnp.ones((2, 8), dtype=jnp.float32)
        params = model.init(rng, ids)
        out    = model.apply(params, ids, mask)
        assert out.shape == (2, 8, small_config.hidden_size)


# ── Policy ────────────────────────────────────────────────────────────────────

class TestPolicyModel:
    def test_logits_shape(self, small_config, rng):
        model  = PolicyModel(small_config)
        ids    = jnp.ones((2, 8), dtype=jnp.int32)
        params = model.init(rng, ids)
        logits = model.apply(params, ids)
        assert logits.shape == (2, 8, small_config.vocab_size)

    def test_sequence_log_prob_shape(self, small_config, rng):
        model  = PolicyModel(small_config)
        ids    = jnp.ones((2, 8), dtype=jnp.int32)
        params = model.init(rng, ids)
        lp     = model.apply(params, ids, method=model.sequence_log_prob)
        assert lp.shape == (2,)

    def test_train_state_creation(self, small_config):
        state = create_policy_train_state(small_config)
        assert state.step == 0


# ── Reward model ──────────────────────────────────────────────────────────────

class TestRewardModel:
    def test_reward_shape(self, small_config, rng):
        model  = RewardModel(small_config)
        ids    = jnp.ones((3, 8), dtype=jnp.int32)
        params = model.init(rng, ids)
        r      = model.apply(params, ids)
        assert r.shape == (3,)

    def test_preference_loss(self, small_config, rng):
        model     = RewardModel(small_config)
        chosen    = jnp.ones((2, 8), dtype=jnp.int32)
        rejected  = jnp.zeros((2, 8), dtype=jnp.int32) + 1
        params    = model.init(rng, chosen)
        loss, acc = model.apply(params, chosen, rejected,
                                 method=model.preference_loss)
        assert loss.shape == ()
        assert 0.0 <= float(acc) <= 1.0


# ── Value model ───────────────────────────────────────────────────────────────

class TestValueModel:
    def test_value_shape(self, small_config, rng):
        model  = ValueModel(small_config)
        ids    = jnp.ones((2, 8), dtype=jnp.int32)
        params = model.init(rng, ids)
        v      = model.apply(params, ids)
        assert v.shape == (2, 8)


# ── Reference model ───────────────────────────────────────────────────────────

class TestReferenceModel:
    def test_log_probs_shape(self, small_config, rng):
        policy = PolicyModel(small_config)
        ids    = jnp.ones((2, 8), dtype=jnp.int32)
        params = policy.init(rng, ids)
        ref    = ReferenceModel(small_config, params)
        lp     = ref.log_probs(ids)
        assert lp.shape == (2,)

    def test_kl_from_policy_shape(self, small_config, rng):
        policy = PolicyModel(small_config)
        ids    = jnp.ones((2, 8), dtype=jnp.int32)
        params = policy.init(rng, ids)
        ref    = ReferenceModel(small_config, params)
        ptlp   = ref.per_token_log_probs(ids)          # (2, 7)
        kl     = ref.kl_from_policy(ptlp, ids)
        assert kl.shape == (2,)
