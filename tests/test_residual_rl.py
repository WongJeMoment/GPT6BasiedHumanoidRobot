"""验证 RL 奖励不会把单臂接触或地面接触当成完整抱抓。"""
import unittest
import torch
from controllers.residual_rl import task_reward
from env_configs.rl_catch import CONFIG
from g1_throw.evaluation import launch_dataset


class ResidualRLTests(unittest.TestCase):
    def reward(self, contact, active=True, upright=1., success=False, fallen=False, dt=1/60):
        return task_reward(torch.tensor([active]), torch.tensor([upright]), torch.tensor([1.]),
                           torch.tensor([contact]), torch.tensor([0.]), torch.tensor([0.]),
                           torch.tensor([success]), torch.tensor([fallen]),
                           torch.zeros(1, 23), torch.zeros(1, 23), dt).item()

    def test_progressive_contact_shaping(self):
        levels = [self.reward(c) for c in ([False]*3, [True, False, False],
                                          [True, True, False], [True]*3)]
        self.assertEqual(levels, sorted(levels))
        self.assertGreater(levels[-1], levels[-2])

    def test_parked_or_fallen_contact_cannot_farm_reward(self):
        self.assertAlmostEqual(self.reward([True]*3, active=False), self.reward([False]*3, active=False))
        self.assertAlmostEqual(self.reward([True]*3, upright=.5), self.reward([False]*3, upright=.5))

    def test_events_not_scaled_by_dt(self):
        for dt in (1/60, 1/120):
            base = self.reward([True]*3, dt=dt)
            self.assertAlmostEqual(self.reward([True]*3, success=True, dt=dt) - base, 20., places=5)
            self.assertAlmostEqual(self.reward([True]*3, fallen=True, dt=dt) - base, -5., places=5)

    def test_fixed_launch_plan_independent_of_global_rng(self):
        a = launch_dataset(CONFIG, 12, 2026)
        torch.rand(200)
        b = launch_dataset(CONFIG, 12, 2026)
        for key in a:
            torch.testing.assert_close(a[key], b[key])
        self.assertEqual(torch.bincount(a["object"]).tolist(), [4, 4, 4])
        for key, bounds in (("speed", CONFIG.speed_range), ("target_lateral", CONFIG.target_lateral_range)):
            self.assertTrue(((a[key] >= bounds[0]) & (a[key] <= bounds[1])).all())
        self.assertFalse(torch.equal(a["speed"], launch_dataset(CONFIG, 12, 2027)["speed"]))


if __name__ == "__main__":
    unittest.main()
