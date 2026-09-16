"""MuJoCo 预览和无窗口检查入口：不启动 Isaac Sim。"""
import argparse
from contextlib import ExitStack, nullcontext
from pathlib import Path
import sys
import threading
import time

# 外层 mujoco 文件夹不设 __init__.py，避免遮蔽已安装的 mujoco 库。
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from g1_throw.config_loader import load_settings
from g1_throw.terminal_status import TerminalKeys, TerminalStatusRecorder, capture_terminal_state
from g1_mujoco import G1MujocoEnv


def main():
    parser = argparse.ArgumentParser(description="MuJoCo G1：与 Isaac Lab 同规则的抛物和接箱放架环境")
    parser.add_argument("--config", default="env_configs/catch_and_place.py")
    parser.add_argument("--mode", choices=("preview", "smoke"), default="preview")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--steps", type=int, help="0 持续运行；可视化默认持续运行，无窗口默认 600 步")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episode_seconds", type=float)
    parser.add_argument("--status_dir", type=Path, default=PROJECT_ROOT / "logs" / "mujoco_robot_status")
    args = parser.parse_args()
    if args.steps is None:
        args.steps = 600 if args.headless or args.mode == "smoke" else 0
    if args.steps < 0 or args.num_envs < 1:
        parser.error("steps 必须 >= 0，num_envs 必须 >= 1")
    config = Path(args.config).expanduser()
    if not config.is_absolute() and not config.is_file():
        config = PROJECT_ROOT / config
    settings = load_settings(config)
    if args.episode_seconds is not None:
        settings.episode_seconds = args.episode_seconds
    torch.set_num_threads(1)
    env = G1MujocoEnv(settings, num_envs=args.num_envs, seed=args.seed)
    try:
        with ExitStack() as stack:
            print(f"MuJoCo G1: 23 个身体关节，手部 0 DOF；{env.num_envs} 个独立环境；"
                  f"{env.observation_dim} 维观测；物体 {[spec.name for spec in settings.objects]}", flush=True)
            if args.mode == "smoke":
                from g1_mujoco.checks import check_launch_and_reset, check_shelf_physics
                check_launch_and_reset(env)
                check_shelf_physics(env)
                env.reset(seed=args.seed)
            viewer = None
            paused = threading.Event()
            if not args.headless:
                from g1_mujoco.viewer import passive_viewer

                def on_key(keycode):
                    if keycode == 32:
                        paused.clear() if paused.is_set() else paused.set()

                viewer = stack.enter_context(passive_viewer(env.model, env.data, key_callback=on_key))
                with viewer.lock():
                    viewer.cam.lookat[:] = (2., 0., .5) if settings.ground_ruler else (0., 0., .8)
                    viewer.cam.distance = 10. if settings.ground_ruler else 4.5
                    viewer.cam.azimuth = -45
                    viewer.cam.elevation = -35
                    # group 3 是碰撞代理，显示原始机器人外观和 group 1 标尺即可。
                    viewer.opt.geomgroup[3] = 0
                print("窗口显示环境 0；按空格可暂停/继续，关闭窗口或终端 Ctrl+C 退出。", flush=True)
            recorder = keys = None
            if args.mode == "preview":
                keys = stack.enter_context(TerminalKeys())
                recorder = TerminalStatusRecorder(args.status_dir, str(config), args.seed)
                recorder.header = "仿真后端: MuJoCo\n" + recorder.header
                print("零动作场景预览，不包含自动接箱/放架策略。状态仅驻留内存，退出不自动保存。", flush=True)
                print(f"点击当前终端按 s（无需回车）保存 TXT：{args.status_dir}" if keys.enabled else
                      "当前不是交互终端，无法读取 s；请从终端启动以手动保存。", flush=True)
                recorder.record(capture_terminal_state(env))
            before_throws = int(env.total_throws.sum())
            successes = 0
            step = 0
            while (not args.steps or step < args.steps) and (viewer is None or viewer.is_running()):
                start = time.monotonic()
                if paused.is_set():
                    if recorder is not None:
                        recorder.handle_keys(keys.poll())
                    viewer.sync()
                    time.sleep(.02)
                    continue
                with viewer.lock() if viewer is not None else nullcontext():
                    obs, reward, ended, timeout, info = env.step(torch.zeros_like(env.actions))
                    if viewer is not None:
                        env._select_collisions(0)
                if recorder is not None:
                    pressed = keys.poll()
                    recorder.record(info["terminal"], force="s" in pressed.lower())
                    recorder.handle_keys(pressed)
                if env.shelf_task is not None:
                    successes += int(info["shelf"]["success"].sum())
                if args.mode == "smoke":
                    assert obs["policy"].shape == (env.num_envs, env.observation_dim)
                    assert torch.isfinite(obs["policy"]).all() and torch.isfinite(reward).all()
                    assert all(row["terminated"] == bool(ended[i]) and row["timeout"] == bool(timeout[i])
                               for i, row in enumerate(info["terminal"]))
                if viewer is not None:
                    viewer.sync()
                    time.sleep(max(0., env.step_dt - (time.monotonic() - start)))
                step += 1
            throws = int(env.total_throws.sum()) - before_throws
            if args.mode == "smoke":
                assert throws > 0, "未发生投放，请增加 --steps"
                print(f"SMOKE PASSED: {env.num_envs} 环境，{step} 步，{throws} 次抛掷，观测/奖励有限", flush=True)
            print(f"MuJoCo 运行统计: 抛掷 {throws}，完整放架成功 {successes}；零动作预览。", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        env.close()


if __name__ == "__main__":
    main()
