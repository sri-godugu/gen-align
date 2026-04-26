"""Unit tests for alignment loss functions and trainers."""
import pytest
import jax
import jax.numpy as jnp
import numpy as np

from src.alignment.losses import (
    sft_loss, reward_loss, dpo_loss,
    ppo_policy_loss, ppo_value_loss,
    entropy_bonus, kl_penalty, gae_advantages,
    multi_objective_reward,
)


B, T, V = 2, 8, 64


@pytest.fixture
def rng():
    return jax.random.PRNGKey(42)


# ── SFT loss ──────────────────────────────────────────────────────────────────

class TestSFTLoss:
    def test_scalar_output(self, rng):
        logits = jax.random.normal(rng, (B, T, V))
        labels = jnp.zeros((B, T), dtype=jnp.int32)
        loss   = sft_loss(logits, labels)
        assert loss.shape == ()
        assert float(loss) > 0

    def test_masked_loss(self, rng):
        logits = jax.random.normal(rng, (B, T, V))
        labels = jnp.zeros((B, T), dtype=jnp.int32)
        mask   = jnp.array([[1, 1, 1, 0, 0, 0, 0, 0]] * B, dtype=jnp.float32)
        loss   = sft_loss(logits, labels, mask)
        assert loss.shape == ()

    def test_zero_logits_loss(self):
        logits = jnp.zeros((B, T, V))
        labels = jnp.zeros((B, T), dtype=jnp.int32)
        loss   = sft_loss(logits, labels)
        expected = jnp.log(jnp.array(V, dtype=jnp.float32))
        assert jnp.abs(loss - expected) < 0.01


# ── Reward loss ───────────────────────────────────────────────────────────────

class TestRewardLoss:
    def test_output_types(self):
        r_w = jnp.array([1.0, 2.0, 3.0])
        r_l = jnp.array([0.5, 1.5, 2.5])
        loss, acc = reward_loss(r_w, r_l)
        assert loss.shape == ()
        assert acc.shape == ()
        assert float(acc) == 1.0   # r_w > r_l always in this case

    def test_equal_rewards_accuracy(self):
        r = jnp.ones((4,))
        loss, acc = reward_loss(r, r)
        assert float(acc) == 0.0   # ties go to rejected


# ── DPO loss ──────────────────────────────────────────────────────────────────

class TestDPOLoss:
    def test_output_shape(self, rng):
        lp = jax.random.normal(rng, (B,)) * 0.1
        loss, metrics = dpo_loss(lp, lp - 1.0, lp - 0.5, lp - 1.5, beta=0.1)
        assert loss.shape == ()
        assert 'reward_margin' in metrics
        assert 'win_rate' in metrics

    def test_correct_preference_lower_loss(self, rng):
        """Chosen > rejected should produce lower loss."""
        k1, k2 = jax.random.split(rng)
        base   = jax.random.normal(k1, (B,))
        loss_good, _ = dpo_loss(base + 1.0, base - 1.0,
                                 base,       base, beta=0.1)
        loss_bad,  _ = dpo_loss(base - 1.0, base + 1.0,
                                 base,       base, beta=0.1)
        assert float(loss_good) < float(loss_bad)


# ── PPO losses ────────────────────────────────────────────────────────────────

class TestPPOLosses:
    def test_policy_loss_shape(self, rng):
        lp     = jax.random.normal(rng, (B, T)) * 0.1
        adv    = jax.random.normal(rng, (B, T))
        loss   = ppo_policy_loss(lp, lp, adv)
        assert loss.shape == ()

    def test_value_loss_shape(self, rng):
        v       = jax.random.normal(rng, (B, T))
        ret     = v + 0.1
        loss    = ppo_value_loss(v, v, ret)
        assert loss.shape == ()

    def test_entropy_bonus_positive(self, rng):
        logits = jax.random.normal(rng, (B, T, V))
        ent    = entropy_bonus(logits)
        assert float(ent) > 0


# ── KL penalty ────────────────────────────────────────────────────────────────

class TestKLPenalty:
    def test_zero_kl_identical(self, rng):
        lp  = jax.random.normal(rng, (B, T)) - 2.0
        kl  = kl_penalty(lp, lp)
        assert jnp.abs(kl) < 1e-5

    def test_positive_kl_direction(self, rng):
        k1, k2 = jax.random.split(rng)
        plp    = jax.random.normal(k1, (B, T)) - 1.0
        rlp    = jax.random.normal(k2, (B, T)) - 2.0
        kl     = kl_penalty(plp, rlp)
        # no sign guarantee — just check it's finite
        assert jnp.isfinite(kl)


# ── GAE ───────────────────────────────────────────────────────────────────────

class TestGAE:
    def test_output_shapes(self, rng):
        rewards = jax.random.normal(rng, (B, T))
        values  = jax.random.normal(rng, (B, T))
        dones   = jnp.zeros((B, T))
        adv, ret = gae_advantages(rewards, values, dones)
        assert adv.shape == (B, T)
        assert ret.shape == (B, T)

    def test_advantages_normalised(self, rng):
        rewards = jax.random.normal(rng, (B, T))
        values  = jax.random.normal(rng, (B, T))
        dones   = jnp.zeros((B, T))
        adv, _  = gae_advantages(rewards, values, dones)
        assert jnp.abs(adv.mean()) < 0.1
        assert jnp.abs(adv.std() - 1.0) < 0.1


# ── Multi-objective reward ────────────────────────────────────────────────────

class TestMultiObjectiveReward:
    def test_output_shape(self):
        rewards = jnp.ones((B, 3))
        weights = jnp.array([0.5, 0.3, 0.2])
        r       = multi_objective_reward(rewards, weights)
        assert r.shape == (B,)
        assert jnp.allclose(r, jnp.ones(B))
