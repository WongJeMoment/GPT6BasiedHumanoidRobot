"""无窗口的真实 MuJoCo 回归检查，不需要 Isaac Sim。"""
from copy import deepcopy
import io
from pathlib import Path
import sys
import tempfile
import unittest

ENVIRONMENT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ENVIRONMENT.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(ENVIRONMENT))

import mujoco
import numpy as np
import torch

from env_configs.catch_and_place import CONFIG
from g1_throw.config_loader import load_settings
from g1_throw.launch import launch_dataset
from g1_throw.terminal_status import TerminalStatusRecorder
from g1_mujoco import G1MujocoEnv
from g1_mujoco.checks import check_launch_and_reset, check_shelf_physics, hold_robot_fixture


class MuJoCoEnvironmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.env = G1MujocoEnv(num_envs=2)

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def setUp(self):
        self.env.launch_plan = None
        self.env.reset(seed=42)

    def test_isaac_action_order_pose_limits_and_observation_layout(self):
        env = self.env
        self.assertEqual(env.observation_dim, 152)
        self.assertEqual(env.robot.joint_names[:7], [
            "left_hip_pitch_joint", "right_hip_pitch_joint", "torso_joint", "left_hip_roll_joint",
            "right_hip_roll_joint", "left_shoulder_pitch_joint", "right_shoulder_pitch_joint"])
        expected_pose = [-.2, -.2, 0, 0, 0, -.85, -.85, 0, 0, .25, -.25, .42, .42, 0, 0,
                         -.23, -.23, 1.25, 1.25, 0, 0, 0, 0]
        np.testing.assert_allclose(env.robot.data.default_joint_pos[0], expected_pose, atol=1e-7)
        # 与现有 Isaac Lab 环境实测的关节软限位对照。
        np.testing.assert_allclose(env.soft_limits[[0, 2, 3, 5, 15, 17, 19, 21]], [
            [-2.08, 2.78], [-2.3562, 2.3562], [-.1205, 2.3905], [-2.67912, 2.50452],
            [-.6095, .6595], [-.04442, 3.23842], [-.23562, .23562], [-1.88487, 1.88487],
        ], atol=2e-5)
        obs = env.reset()[0]["policy"]
        self.assertEqual(obs.shape, (2, 152))
        torch.testing.assert_close(obs[:, :6], torch.zeros(2, 6), atol=1e-6, rtol=0)
        torch.testing.assert_close(obs[:, 6:9], torch.tensor([[0., 0., -1.]]).repeat(2, 1))
        torch.testing.assert_close(obs[:, 9:55], torch.zeros(2, 46), atol=1e-6, rtol=0)
        env._throw([0])
        obs = env._get_observations()["policy"]
        torch.testing.assert_close(obs[0, 78:81], env.objects[0].data.root_pos_w[0] - env.robot.data.root_pos_w[0])
        self.assertEqual(float(obs[0, 84]), 1.)
        self.assertEqual(float(obs[1, 84]), 0.)

    def test_launch_ballistics_fixed_hand_and_partial_reset(self):
        check_launch_and_reset(self.env)

    def test_real_shelf_support_success_drop_and_timeout(self):
        check_shelf_physics(self.env)

    def test_actual_left_hand_box_contact_and_release(self):
        env = self.env
        env._throw([0])
        palm = env.robot.data.body_pos_w[0, env.shelf_hand_ids[0]].numpy().copy()
        env._set_object_state(0, 0, palm)
        env.forward()
        env._get_dones()
        self.assertTrue(env.shelf_task.contact[0, 0])
        self.assertFalse(env.shelf_task.contact[0, 1])
        self.assertTrue(env.shelf_task.robot_touch[0])
        self.assertFalse(env.shelf_task.supported[0])
        env._set_object_state(0, 0, (3., 0., 2.))
        env.forward()
        env._get_dones()
        self.assertFalse(env.shelf_task.robot_touch[0])
        self.assertFalse(env.shelf_task.contact[0].any())

    def test_action_clipping_speed_limit_and_rejection_before_mutation(self):
        env = self.env
        original = env.joint_target.clone()
        env.step(torch.full((2, 23), 10.))
        self.assertTrue((env.actions == 1).all())
        self.assertLessEqual(float((env.joint_target - original).abs().max()), .100001)
        self.assertTrue((env.joint_target >= env.soft_limits[:, 0]).all())
        self.assertTrue((env.joint_target <= env.soft_limits[:, 1]).all())
        position = env.data.qpos.copy()
        for invalid in (torch.zeros(23), torch.full((2, 23), float("nan"))):
            with self.assertRaises(ValueError):
                env.step(invalid)
            np.testing.assert_array_equal(position, env.data.qpos)

    def test_same_seed_reproduces_launch_and_supports_shared_plan(self):
        env = self.env
        env._throw([0, 1])
        first = env.objects[0].data.root_state_w.clone()
        env.reset(seed=42)
        env._throw([0, 1])
        torch.testing.assert_close(first, env.objects[0].data.root_state_w, atol=0, rtol=0)
        env.reset()
        env.launch_plan = launch_dataset(CONFIG, 2, seed=2026)
        env._throw([0, 1])
        np.testing.assert_allclose(env.objects[0].data.root_pos_w[:, 0], env.launch_plan["distance"], atol=1e-6)
        np.testing.assert_allclose(env.objects[0].data.root_pos_w[:, 1], env.launch_plan["launch_lateral"], atol=1e-6)

    def test_drop_snapshot_survives_reset_and_s_is_only_save_trigger(self):
        env = self.env
        env._throw([0])
        env._set_object_state(0, 0, (1., 0., .155))
        env.forward()
        obs, reward, ended, _, info = hold_robot_fixture(env)
        self.assertTrue(ended[0])
        self.assertTrue(info["terminal"][0]["dropped"])
        self.assertGreater(info["terminal"][0]["box"][2], 0.)
        self.assertEqual(env.active_object[0], -1)
        self.assertFalse(env.dropped[0])
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "records"
            recorder = TerminalStatusRecorder(directory, "MuJoCo", 42, stream=io.StringIO())
            recorder.record(info["terminal"])
            recorder.handle_keys("q\n")
            self.assertFalse(directory.exists())
            recorder.handle_keys("s")
            path, = directory.glob("*.txt")
            self.assertIn("是否掉落=是", path.read_text(encoding="utf-8"))

    def test_all_scenes_remain_finite_and_inactive_objects_are_isolated(self):
        for filename, dimension in (("single_throw.py", 88), ("continuous_throw.py", 93)):
            with self.subTest(scene=filename):
                settings = load_settings(PROJECT_ROOT / "env_configs" / filename)
                env = G1MujocoEnv(settings, num_envs=2)
                try:
                    env._throw([0, 1])
                    for _ in range(90):
                        obs, reward, _, _, _ = env.step(torch.zeros(2, 23))
                        self.assertEqual(obs["policy"].shape, (2, dimension))
                        self.assertTrue(torch.isfinite(obs["policy"]).all() and torch.isfinite(reward).all())
                    for i in range(env.num_envs):
                        for j, obj in enumerate(env.objects):
                            if int(env.active_object[i]) != j:
                                self.assertEqual(float(obj.data.root_pos_w[i, 2]), -20 - 2 * j)
                        self.assertFalse(env.datas[i].warning.number.any())
                finally:
                    env.close()

    def test_continuous_launch_replaces_only_active_object_on_schedule(self):
        from env_configs.single_throw import CONFIG as SIMPLE

        settings = deepcopy(SIMPLE)
        settings.continuous = True
        settings.interval_range = (.05, .05)
        settings.first_throw_delay = 0.
        env = G1MujocoEnv(settings)
        try:
            chosen = []
            previous = 0
            for _ in range(35):
                hold_robot_fixture(env)
                if env.total_throws[0] != previous:
                    chosen.append(int(env.last_object[0]))
                    previous = int(env.total_throws[0])
            self.assertGreaterEqual(len(chosen), 8)
            self.assertTrue(all(a != b for a, b in zip(chosen, chosen[1:])))
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
