"""接箱放架的顺序、真实支撑、松手、持续静置与失败优先级。"""
from copy import deepcopy
import math
import unittest

import torch

from env_configs.catch_and_place import CONFIG
from env_configs.common import Settings
from g1_throw.launch import launch_dataset
from g1_throw.shelf_task import ShelfPhase, ShelfTask, box_geometry


class ShelfTaskTests(unittest.TestCase):
    def setUp(self):
        self.cfg = deepcopy(CONFIG.shelf)
        self.cfg.catch_seconds = .2
        self.cfg.settle_seconds = .3
        self.task = ShelfTask(2, "cpu", self.cfg, CONFIG.objects[0].size)

    def step(self, **overrides):
        args = dict(active=torch.ones(2, dtype=torch.bool),
                    position=torch.tensor([[.25, 0., .95]]).repeat(2, 1),
                    quaternion=torch.tensor([[1., 0., 0., 0.]]).repeat(2, 1),
                    velocity=torch.zeros(2, 3), angular_velocity=torch.zeros(2, 3),
                    robot_position=torch.tensor([[0., 0., .74]]).repeat(2, 1),
                    robot_velocity=torch.zeros(2, 3),
                    contact=torch.tensor([[True, True, False]]).repeat(2, 1),
                    robot_touch=torch.ones(2, dtype=torch.bool), supported=torch.zeros(2, dtype=torch.bool),
                    shelf_touch=torch.zeros(2, dtype=torch.bool),
                    upright=torch.ones(2, dtype=torch.bool), fallen=torch.zeros(2, dtype=torch.bool), dt=.1)
        args.update(overrides)
        self.task.update(**args)

    def place(self, **overrides):
        args = dict(position=self.task.goal.repeat(2, 1), contact=torch.zeros(2, 3, dtype=torch.bool),
                    robot_touch=torch.zeros(2, dtype=torch.bool), supported=torch.ones(2, dtype=torch.bool),
                    shelf_touch=torch.ones(2, dtype=torch.bool))
        args.update(overrides)
        self.step(**args)

    def catch(self):
        self.step()
        self.step()
        self.assertTrue(self.task.caught.all())

    def reward(self, fallen=False, timeout=False):
        return self.task.reward(torch.ones(2, dtype=torch.bool), torch.ones(2), torch.ones(2, dtype=torch.bool),
                                torch.full((2,), fallen), torch.full((2,), timeout),
                                torch.zeros(2, 23), torch.zeros(2, 23), torch.ones(2), .1)

    def test_catch_is_intermediate_and_release_on_shelf_completes(self):
        self.step()
        self.assertFalse(self.task.caught.any())
        self.step()
        self.assertTrue(self.task.new_catch.all())
        self.assertFalse(self.task.success.any())
        self.assertTrue((self.task.phase == ShelfPhase.PLACE).all())
        for _ in range(2):
            self.place()
            self.assertFalse(self.task.success.any())
        self.place()
        self.assertTrue(self.task.success.all())
        self.assertTrue((self.reward() > 90).all())

    def test_box_landing_directly_on_shelf_never_counts(self):
        for _ in range(8):
            self.place()
        self.assertTrue(self.task.on_shelf.all())
        self.assertFalse(self.task.caught.any())
        self.assertFalse(self.task.success.any())
        # 落架后再夹住不算在空中接住。
        for _ in range(4):
            self.place(contact=torch.ones(2, 3, dtype=torch.bool), robot_touch=torch.ones(2, dtype=torch.bool))
        self.assertFalse(self.task.caught.any())
        self.assertTrue((self.reward(timeout=True) < 0).all())
        for _ in range(4):
            self.step()  # 再从架上提起，不能洗掉先落架的历史。
        self.assertTrue(self.task.shelf_first.all())
        self.assertFalse(self.task.caught.any())

    def test_contact_must_be_sustained_and_low_relative_speed(self):
        self.step()
        self.step(contact=torch.tensor([[True, False, False]]).repeat(2, 1))
        self.step()
        self.assertFalse(self.task.caught.any())
        self.step(velocity=torch.ones(2, 3))
        self.assertFalse(self.task.caught.any())
        self.catch()

    def test_no_support_or_any_robot_contact_prevents_placement(self):
        self.catch()
        for _ in range(5):
            self.place(supported=torch.zeros(2, dtype=torch.bool))
        self.assertFalse(self.task.success.any())
        for _ in range(5):
            # 即使双臂/胸部已经松开，腿等其他刚体还碰箱子也不能成功。
            self.place(robot_touch=torch.ones(2, dtype=torch.bool))
        self.assertFalse(self.task.success.any())

    def test_settling_timer_restarts_after_motion_or_contact(self):
        self.catch()
        self.place()
        self.place(velocity=torch.ones(2, 3))
        self.assertTrue((self.task.settle_time == 0).all())
        self.place()
        self.place(angular_velocity=torch.ones(2, 3))
        self.assertTrue((self.task.settle_time == 0).all())
        self.place()
        self.place(robot_touch=torch.ones(2, dtype=torch.bool))
        self.assertTrue((self.task.settle_time == 0).all())

    def test_rotated_box_must_fit_entire_surface(self):
        self.catch()
        # 45 度时 XY 半尺寸约 .2475m；再偏移 .03m 会越过有余量的架面。
        q = torch.tensor([[math.cos(math.pi/8), 0., 0., math.sin(math.pi/8)]]).repeat(2, 1)
        position = self.task.goal.repeat(2, 1)
        position[:, 1] += .03
        for _ in range(5):
            self.place(position=position, quaternion=q)
        self.assertFalse(self.task.on_shelf.any())
        self.assertFalse(self.task.success.any())

    def test_height_tilt_and_edge_are_not_valid_placements(self):
        self.catch()
        for offset in ((.2, 0., 0.), (0., 0., .10), (0., 0., -.20)):
            self.place(position=self.task.goal.repeat(2, 1) + torch.tensor(offset))
            self.assertFalse(self.task.on_shelf.any())
        self.place(quaternion=torch.tensor([[math.cos(.2), math.sin(.2), 0., 0.]]).repeat(2, 1))
        self.assertFalse(self.task.on_shelf.any())

    def test_fall_or_ground_contact_overrides_completion(self):
        self.catch()
        self.place()
        self.place()
        self.place(fallen=torch.ones(2, dtype=torch.bool))
        self.assertFalse(self.task.success.any())
        self.assertTrue((self.reward(fallen=True, timeout=True) < -79).all())
        self.place(position=torch.tensor([[0., 0., .16]]).repeat(2, 1))
        self.assertTrue(self.task.dropped.all())
        self.assertTrue((self.reward(timeout=True) < -59).all())

    def test_inactive_pool_and_partial_reset_are_isolated(self):
        self.catch()
        self.task.reset(torch.tensor([0]))
        self.assertFalse(self.task.caught[0])
        self.assertTrue(self.task.caught[1])
        self.step(active=torch.zeros(2, dtype=torch.bool), position=torch.full((2, 3), -20.))
        self.assertFalse(self.task.success.any())
        self.assertFalse(self.task.dropped.any())

    def test_progress_reward_cannot_be_farmed_by_moving_back_and_forth(self):
        self.catch()
        near = (self.task.goal + torch.tensor([.25, 0., .95])) / 2
        self.step(position=near.repeat(2, 1))
        self.assertTrue((self.task.progress > 0).all())
        self.step()
        self.step(position=near.repeat(2, 1))
        self.assertTrue((self.task.progress == 0).all())
        self.assertFalse(self.task.new_catch.any())

    def test_box_geometry_tracks_rotation_and_bottom(self):
        q = torch.tensor([[math.sqrt(.5), math.sqrt(.5), 0., 0.]])
        extent, vertical = box_geometry(q, torch.tensor([.175, .175, .15]))
        torch.testing.assert_close(extent, torch.tensor([[.175, .15, .175]]))
        self.assertLess(vertical.item(), 1e-5)

    def test_config_rejects_incompatible_tasks_and_geometry(self):
        CONFIG.validate()
        cfg = deepcopy(CONFIG)
        cfg.continuous = True
        with self.assertRaises(ValueError):
            cfg.validate()
        for name, value in (("size", (.3, .3, .04)), ("position", (.45, .65, .2)),
                            ("position", (CONFIG.env_spacing, .65, .75)), ("settle_seconds", 12.),
                            ("support_weight_fraction", 1.2), ("linear_speed", float("nan"))):
            cfg = deepcopy(CONFIG)
            setattr(cfg.shelf, name, value)
            with self.assertRaises(ValueError):
                cfg.validate()

    def test_long_throws_need_reachable_ballistics_and_room(self):
        for name, value in (("speed_range", (6., 8.)), ("env_spacing", 8.), ("env_spacing", 11.)):
            cfg = deepcopy(CONFIG)
            setattr(cfg, name, value)
            with self.assertRaises(ValueError):
                cfg.validate()
        cfg = deepcopy(CONFIG)
        cfg.shelf.escape_distance = 5.01  # X=5m 合法，但 (5, ±0.4)m 的实际距离更大。
        with self.assertRaises(ValueError):
            cfg.validate()

    def test_launch_lateral_config_rejects_invalid_ranges(self):
        for bounds in ((.4, -.4), (float("nan"), .4), (-.4, float("inf"))):
            cfg = deepcopy(CONFIG)
            cfg.launch_lateral_range = bounds
            with self.assertRaisesRegex(ValueError, "launch_lateral_range"):
                cfg.validate()
        cfg = deepcopy(CONFIG)
        cfg.launch_lateral_range = (-.4, -.4)
        cfg.validate()  # 固定偏移也可以用于复现边界案例。

    def test_fixed_launch_plan_preserves_lateral_samples(self):
        plan = launch_dataset(CONFIG, 128, seed=2026)
        torch.rand(37)  # 模拟策略/重置消耗全局随机数。
        repeated = launch_dataset(CONFIG, 128, seed=2026)
        for key in plan:
            torch.testing.assert_close(plan[key], repeated[key], atol=0, rtol=0)
        x, y = plan["distance"], plan["launch_lateral"]
        self.assertTrue(((3. <= x) & (x <= 5.)).all())
        self.assertTrue(((-.4 <= y) & (y <= .4)).all())
        # 远近两段都要有来自左右两侧的发射样本。
        for mask in (x < 4., x >= 4.):
            self.assertTrue((y[mask] < -.2).any())
            self.assertTrue((y[mask] > .2).any())
        self.assertNotIn("launch_lateral", launch_dataset(Settings(), 8, seed=2026))


if __name__ == "__main__":
    unittest.main()
