"""验证手动保存、真实终端按键、事件留存及多环境复位前快照。"""
from copy import deepcopy
import io
import os
from pathlib import Path
import pty
import tempfile
import termios
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from env_configs.catch_and_place import CONFIG
from g1_throw.shelf_task import ShelfTask
from g1_throw.terminal_status import TerminalKeys, TerminalStatusRecorder, capture_terminal_state


def fixture_env():
    state = torch.zeros(2, 13)
    state[:, 3] = 1.0
    state[:, 2] = .16
    state[1, 0] = 12.0
    robot = state.clone()
    robot[:, 2] = .74
    forces = torch.zeros(2, 1, 4, 3)
    forces[0, 0, 1, 2] = 2.0
    task = ShelfTask(2, "cpu", deepcopy(CONFIG.shelf), CONFIG.objects[0].size)
    task.dropped[0] = True
    task.contact[0, 0] = True
    task.robot_touch[0] = True
    task.phase[:] = 1
    return SimpleNamespace(
        num_envs=2, indices=torch.arange(2), objects=[SimpleNamespace(data=SimpleNamespace(root_state_w=state))],
        robot=SimpleNamespace(data=SimpleNamespace(root_state_w=robot,
            projected_gravity_b=torch.tensor([[0., 0., -1.], [0., 0., -1.]]))),
        scene=SimpleNamespace(env_origins=torch.tensor([[0., 0., 0.], [12., 0., 0.]])),
        active_object=torch.zeros(2, dtype=torch.long), throw_count=torch.ones(2, dtype=torch.long),
        episode_length_buf=torch.tensor([60, 30]), common_step_counter=60, step_dt=1 / 60,
        fallen=torch.zeros(2, dtype=torch.bool), reset_terminated=torch.tensor([True, False]),
        reset_time_outs=torch.zeros(2, dtype=torch.bool), settings=CONFIG, shelf_task=task,
        shelf_sensor=SimpleNamespace(data=SimpleNamespace(force_matrix_w=forces)),
        shelf_contact_groups=[[1], [2], [3]],
        shelf_robot_bodies=["left_palm_link", "right_palm_link", "torso_link"],
    )


class TerminalStatusTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name) / "status"
        self.output = io.StringIO()
        self.recorder = TerminalStatusRecorder(self.directory, "catch_and_place.py", 42, self.output)
        self.env = fixture_env()

    def test_no_file_or_directory_without_s(self):
        self.recorder.record(capture_terminal_state(self.env))
        self.recorder.handle_keys("a\nq\x03")
        self.assertIn("是否掉落=是", self.output.getvalue())
        self.assertIn("左臂/手=是(2.000N)", self.output.getvalue())
        self.assertFalse(self.directory.exists())

    def test_s_saves_utf8_txt_and_later_records_require_another_s(self):
        self.recorder.record(capture_terminal_state(self.env))
        self.recorder.handle_keys("s")
        first, = self.directory.iterdir()
        initial = first.read_text(encoding="utf-8")
        self.assertEqual(first.suffix, ".txt")
        self.assertIn("回合结束（复位前）: 箱子掉落/越界", initial)
        self.assertIn("环境 1", initial)
        self.assertIn("left_palm_link", initial)
        self.assertIn("姿态(wxyz)", initial)
        self.env.common_step_counter = 120
        self.env.reset_terminated.zero_()
        self.env.shelf_task.dropped.zero_()
        self.recorder.record(capture_terminal_state(self.env))
        self.assertEqual(first.read_text(encoding="utf-8"), initial)
        self.recorder.handle_keys("S")
        files = sorted(self.directory.iterdir())
        self.assertEqual(len(files), 2)
        latest = next(path for path in files if path != first).read_text(encoding="utf-8")
        self.assertIn("仿真 2.000s", latest)
        self.assertIn("是否掉落=是", latest)  # 历史失败没有被新回合抹掉。
        self.assertIn("环境 0 | 回合 2", latest)
        self.assertIn("环境 1 | 回合 1", latest)

    def test_contact_and_drop_events_between_periodic_samples_are_retained(self):
        self.env.reset_terminated.zero_()
        self.env.shelf_task.dropped.zero_()
        self.recorder.record(capture_terminal_state(self.env))
        self.env.common_step_counter += 1
        self.recorder.record(capture_terminal_state(self.env))
        self.assertEqual(len(self.recorder.records), 2)
        self.env.shelf_task.contact[0, 1] = True
        self.env.common_step_counter += 1
        self.recorder.record(capture_terminal_state(self.env))
        self.assertEqual(len(self.recorder.records), 3)
        self.env.reset_terminated[0] = True
        self.env.shelf_task.dropped[0] = True
        self.env.common_step_counter += 1
        self.recorder.record(capture_terminal_state(self.env))
        self.assertEqual(len(self.recorder.records), 4)
        self.assertIn("是否掉落=是", self.recorder.records[-1])

    def test_snapshot_survives_partial_reset_and_uses_local_coordinates(self):
        snapshot = capture_terminal_state(self.env)
        self.env.shelf_task.reset(torch.tensor([0]))
        self.env.active_object[0] = -1
        self.env.objects[0].data.root_state_w[0, 2] = -20
        self.env.robot.data.root_state_w[0, 2] = .9
        self.env.reset_terminated[0] = False
        self.assertTrue(snapshot[0]["dropped"])
        self.assertTrue(snapshot[0]["terminated"])
        self.assertTrue(snapshot[0]["contact"][0])
        self.assertAlmostEqual(snapshot[0]["box"][2], .16)
        self.assertAlmostEqual(snapshot[0]["robot"][2], .74)
        self.assertEqual(snapshot[1]["box"][0], 0.)
        self.assertFalse(snapshot[1]["dropped"])

    def test_inactive_and_non_shelf_objects_do_not_report_false_results(self):
        self.env.active_object[0] = -1
        self.env.shelf_task.reset(torch.tensor([0]))
        row = capture_terminal_state(self.env)[0]
        self.assertEqual(row["touching_bodies"], [])
        self.assertEqual(row["contact_forces"], [0.] * 4)
        self.env.shelf_task = None
        self.recorder.record(capture_terminal_state(self.env))
        self.assertIn("箱子/物体: 未投放", self.output.getvalue())
        self.assertIn("是否掉落=未启用判定", self.output.getvalue())

    def test_failed_save_keeps_memory_for_retry(self):
        self.recorder.record(capture_terminal_state(self.env))
        with patch.object(Path, "mkdir", side_effect=PermissionError("只读目录")):
            self.recorder.handle_keys("s")
        self.assertIn("保存失败", self.output.getvalue())
        self.assertEqual(len(self.recorder.records), 2)
        self.recorder.handle_keys("s")
        self.assertEqual(len(list(self.directory.glob("*.txt"))), 1)

    def test_non_terminal_input_is_not_read_or_saved(self):
        with TerminalKeys(io.StringIO("s")) as keys:
            self.assertFalse(keys.enabled)
            self.assertEqual(keys.poll(), "")
        self.assertFalse(self.directory.exists())

    def test_real_terminal_s_needs_no_enter_and_restores_settings_on_error(self):
        master, slave = pty.openpty()
        self.addCleanup(os.close, master)
        with os.fdopen(slave, "r") as stream:
            original = termios.tcgetattr(stream.fileno())
            self.recorder.record(capture_terminal_state(self.env))
            with self.assertRaisesRegex(RuntimeError, "退出"):
                with TerminalKeys(stream) as keys:
                    current = termios.tcgetattr(stream.fileno())
                    self.assertFalse(current[3] & termios.ICANON)
                    self.assertFalse(current[3] & termios.ECHO)
                    self.assertTrue(current[3] & termios.ISIG)  # Ctrl+C 仍能中断。
                    self.assertEqual(keys.poll(), "")
                    os.write(master, b"s")
                    ready, _, _ = __import__("select").select([stream.fileno()], [], [], 1)
                    self.assertTrue(ready)
                    self.recorder.handle_keys(keys.poll())
                    self.assertEqual(len(list(self.directory.glob("*.txt"))), 1)
                    raise RuntimeError("退出")
            self.assertEqual(termios.tcgetattr(stream.fileno()), original)


if __name__ == "__main__":
    unittest.main()
