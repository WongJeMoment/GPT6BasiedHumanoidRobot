import unittest
import torch
from controllers.hug_reward import dropped_object, hug_reward


class HugRewardTests(unittest.TestCase):
    def reward(self, hold=0., contact=False, drop=False, fall=False, timeout=False):
        return hug_reward(torch.tensor([True]), torch.ones(1), torch.full((1, 3), contact),
                          torch.ones(1), torch.tensor([hold]), 2., torch.zeros(1), torch.zeros(1),
                          torch.zeros(1, 3), torch.zeros(1, 23), torch.zeros(1, 23),
                          torch.tensor([8.]), torch.tensor([drop]), torch.tensor([fall]),
                          torch.tensor([timeout]), 1/60).item()

    def test_drop_ends_only_launched_attempt(self):
        active = torch.tensor([False, True, True, True])
        height = torch.tensor([-20., .49, .8, .8])
        relative = torch.tensor([[0., 0., 0.], [.2, 0., -.3], [.3, 0., .1], [-.7, 0., .1]])
        self.assertEqual(dropped_object(active, height, relative, .5).tolist(), [False, True, False, True])

    def test_drop_overrides_prior_hold_and_timeout_bonus(self):
        self.assertLess(self.reward(hold=3, contact=True, drop=True, timeout=True), -59)
        self.assertLess(self.reward(hold=3, contact=True, fall=True, timeout=True), -79)

    def test_empty_standing_is_not_success(self):
        self.assertLess(self.reward(), 0)
        self.assertLess(self.reward(timeout=True), -29)
        self.assertLess(self.reward(hold=.4, contact=True, timeout=True), 0)
        self.assertGreater(self.reward(hold=2, contact=True, timeout=True), 100)

    def test_sustained_hug_beats_brief_touch(self):
        self.assertGreater(self.reward(hold=2, contact=True), self.reward(hold=.1, contact=True))
