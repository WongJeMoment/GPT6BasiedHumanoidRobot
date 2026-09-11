"""持续站立奖励与旧策略迁移的行为检查。"""
import unittest
import torch
from controllers.residual_rl import stability_reward
from g1_throw.warm_start import transfer_state


class ActiveLegTests(unittest.TestCase):
    def reward(self, hold=0., fallen=False, terminal=False, error=0.):
        return stability_reward(torch.ones(1), torch.tensor([.7]), torch.full((1, 3), error),
                                torch.zeros(1, 3), torch.zeros(1, 2, 3), torch.ones(1, 2),
                                torch.tensor([hold]), torch.tensor([fallen]),
                                torch.tensor([terminal]), 1/60).item()

    def test_fall_cannot_cash_terminal_hold_bonus(self):
        self.assertLess(self.reward(hold=3, fallen=True, terminal=True), 0)
        self.assertGreater(self.reward(hold=3, terminal=True), 50)
        self.assertLess(self.reward(hold=.4, terminal=True), 5)

    def test_support_and_sustained_hold(self):
        self.assertGreater(self.reward(), self.reward(error=.3))
        self.assertGreater(self.reward(hold=2), self.reward(hold=.4))

    def test_expand_input_preserves_old_actor_computation(self):
        old = {"actor.0.weight": torch.randn(4, 5), "actor.0.bias": torch.randn(4),
               "actor_obs_normalizer._var": torch.ones(1, 5)}
        new = {"actor.0.weight": torch.randn(4, 8), "actor.0.bias": torch.zeros(4),
               "actor_obs_normalizer._var": torch.ones(1, 8)}
        copied = transfer_state(old, new)
        x = torch.randn(2, 8)
        torch.testing.assert_close(x @ copied["actor.0.weight"].T, x[:, :5] @ old["actor.0.weight"].T)
        self.assertTrue((copied["actor_obs_normalizer._var"] > 0).all())


if __name__ == "__main__":
    unittest.main()
