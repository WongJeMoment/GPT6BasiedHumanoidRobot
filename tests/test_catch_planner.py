"""不启动 Isaac Sim 的控制数学与任务状态回归测试。"""
import unittest
import torch
from controllers.config import CatchControlCfg
from controllers.planner import CatchPlanner, Phase
from controllers.whole_body import damped_ik, rotate_yaw


class CatchPlannerTests(unittest.TestCase):
    def setUp(self):
        self.cfg = CatchControlCfg()
        self.planner = CatchPlanner(2, "cpu", self.cfg)
        self.position = torch.tensor([[1.0, 0., .4], [1., 0., .4]])
        self.velocity = torch.tensor([[-4., 0., 0.], [4., 0., 0.]])
        self.gravity = torch.tensor([[0., 0., -9.81]]).expand(2, -1)
        self.active = torch.ones(2, dtype=torch.bool)
        self.contact = torch.zeros(2, 3, dtype=torch.bool)
        self.upright = self.active.clone()

    def step(self, dt=1/60):
        return self.planner.update(self.position, self.velocity, self.gravity,
                                   self.active, self.contact, self.upright, dt)

    def test_ballistic_intercept_and_receding_object(self):
        point, phase = self.step()
        self.assertEqual(phase.tolist(), [Phase.INTERCEPT, Phase.READY])
        t = (1 - self.cfg.intercept_x) / 4
        self.assertAlmostEqual(point[0, 2].item(), .4 - .5 * 9.81 * t*t, places=5)

    def test_contact_hold_and_partial_reset(self):
        self.step()
        self.position[0] = torch.tensor([.3, 0., .2])
        self.velocity[0] = 0
        self.contact[0] = True
        self.step()
        self.assertEqual(self.planner.phase[0], Phase.ABSORB)
        for _ in range(40):
            self.step()
        self.assertEqual(self.planner.phase[0], Phase.HOLD)
        self.assertTrue(self.planner.success[0])
        self.planner.reset(torch.tensor([1]))
        self.assertTrue(self.planner.success[0])
        self.planner.reset(torch.tensor([0]))
        self.assertFalse(self.planner.success.any())

    def test_proximity_alone_never_counts_as_catch(self):
        self.step()
        self.position[0] = torch.tensor([.3, 0., .2])
        self.velocity[0] = 0
        for _ in range(80):
            self.step()
        self.assertFalse(self.planner.success.any())

    def test_dropped_contact_restarts_stability_timer(self):
        self.step()
        self.position[0] = torch.tensor([.3, 0., .2])
        self.velocity[0] = 0
        self.contact[0] = True
        for _ in range(12):
            self.step()
        self.contact[0, 1] = False
        self.step()
        self.assertEqual(self.planner.stable_time[0], 0)
        self.assertFalse(self.planner.success[0])

    def test_parked_and_fallen(self):
        self.active[:] = False
        self.step()
        self.assertTrue((self.planner.phase == Phase.READY).all())
        self.upright[0] = False
        self.step()
        self.assertEqual(self.planner.phase[0], Phase.RECOVER)

    def test_two_arms_without_torso_is_not_body_hug(self):
        self.step()
        self.position[0] = torch.tensor([.3, 0., .2])
        self.velocity[0] = 0
        self.contact[0, :2] = True
        for _ in range(50):
            self.step()
        self.assertFalse(self.planner.success.any())
        self.assertEqual(self.planner.stable_time[0], 0)

    def test_torso_only_triggers_absorb_but_not_success(self):
        self.step()
        self.position[0] = torch.tensor([.3, 0., .2])
        self.velocity[0] = 0
        self.contact[0, 2] = True
        self.step()
        self.assertEqual(self.planner.phase[0], Phase.ABSORB)
        for _ in range(50):
            self.step()
        self.assertFalse(self.planner.success.any())

    def test_lost_torso_contact_restarts_timer(self):
        self.step()
        self.position[0] = torch.tensor([.3, 0., .2])
        self.velocity[0] = 0
        self.contact[0] = True
        for _ in range(10):
            self.step()
        self.assertGreater(self.planner.stable_time[0], 0)
        self.contact[0, 2] = False
        self.step()
        self.assertEqual(self.planner.stable_time[0], 0)

    def test_wrap_targets_close_to_chest_without_hold_jump(self):
        from controllers.hug_targets import hug_targets
        chest = torch.tensor([[.08, 0., .25]])
        start = torch.tensor([[.4, .05, .3]])
        size = torch.tensor([[.2, .18, .15]])
        phase = torch.tensor([Phase.ABSORB])
        before, _, _ = hug_targets(start, chest, start, phase, torch.tensor([0.]), size, self.cfg)
        end, palms, elbows = hug_targets(start, chest, start, phase,
                                         torch.tensor([self.cfg.absorb_seconds]), size, self.cfg)
        hold, _, _ = hug_targets(start, chest, start, torch.tensor([Phase.HOLD]),
                                 torch.tensor([0.]), size, self.cfg)
        torch.testing.assert_close(before, start)
        torch.testing.assert_close(hold, end)
        self.assertLess(end[0, 0], start[0, 0])
        self.assertTrue((palms[0, :, 0] > end[0, 0]).all())
        self.assertTrue((elbows[0, :, 0] < palms[0, :, 0]).all())
        self.assertGreater(palms[0, 0, 1], 0)
        self.assertLess(palms[0, 1, 1], 0)

    def test_yaw_round_trip_and_singular_ik(self):
        yaw = torch.tensor([1.2, -2.1])
        rotated = rotate_yaw(self.position, yaw)
        torch.testing.assert_close(rotate_yaw(rotated, yaw, inverse=True), self.position)
        zero = torch.zeros(2, 3, 5)
        self.assertTrue(torch.isfinite(damped_ik(zero, self.position, .08)).all())
        jac = torch.eye(3)[None]
        delta = damped_ik(jac, torch.tensor([[.1, .2, .3]]), .08)
        self.assertLess((delta - torch.tensor([[.1, .2, .3]])).norm(), .003)


if __name__ == "__main__":
    unittest.main()
