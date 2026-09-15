"""环境入口：预览场景或运行 PhysX 检查，不加载控制策略或训练器。"""
import argparse
from pathlib import Path
import itertools
import time

from isaaclab.app import AppLauncher
from g1_throw.config_loader import load_settings

parser = argparse.ArgumentParser(description="G1 抛物与接箱放架环境")
parser.add_argument("--config", default="env_configs/catch_and_place.py")
parser.add_argument("--mode", choices=("preview", "smoke"), default="preview")
parser.add_argument("--num_envs", type=int)
parser.add_argument("--steps", type=int, default=None, help="运行步数；0 持续运行，可视化预览默认持续运行")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--episode_seconds", type=float, help="覆盖环境回合时长")
parser.add_argument("--status_dir", type=Path, default=Path(__file__).resolve().parent / "logs" / "robot_status",
                    help="按 s 保存状态 TXT 的目录；不按 s 不创建目录或文件")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.steps is None:
    args.steps = 0 if args.mode == "preview" and not args.headless else 600
if args.steps < 0:
    parser.error("--steps 必须 >= 0")
settings = load_settings(args.config)
if args.num_envs is not None:
    settings.num_envs = args.num_envs
elif args.mode == "preview":
    settings.num_envs = 1
if args.episode_seconds is not None:
    settings.episode_seconds = args.episode_seconds
settings.validate()
launcher = AppLauncher(args)
app = launcher.app


def simulation_steps(env):
    """可视化按实时速度运行，直到关闭窗口或达到指定步数。"""
    for index in itertools.count():
        if not app.is_running() or (args.steps and index >= args.steps):
            return
        start = time.monotonic()
        yield index
        if env.sim.has_gui() and args.mode == "preview":
            time.sleep(max(0.0, env.step_dt - (time.monotonic() - start)))


env = None
terminal_keys = None
exit_code = 0
try:
    import torch
    from g1_throw.environment import G1ThrowEnv, make_env_cfg

    env = G1ThrowEnv(make_env_cfg(settings, args.device or "cuda:0", args.seed))
    env.terminal_status_enabled = True
    print(f"G1: {env.robot.num_joints} 个身体关节；手部 0 自由度；物体 {[o.name for o in settings.objects]}", flush=True)
    env.reset()
    if args.mode == "smoke":
        from g1_throw.checks import check_launch_and_partial_reset, check_shelf_task
        check_launch_and_partial_reset(env)
        check_shelf_task(env)
        env.reset()  # 清除检查夹具，下面重新统计实际运行。
    else:
        print("环境预览：发送零动作，不含自动接箱/放架策略；关闭窗口或按 Ctrl+C 退出。", flush=True)
        from g1_throw.terminal_status import TerminalKeys, TerminalStatusRecorder, capture_terminal_state
        recorder = TerminalStatusRecorder(args.status_dir, args.config, args.seed)
        terminal_keys = TerminalKeys()
        terminal_keys.__enter__()
        print("状态记录仅保存在内存，退出不自动保存。", flush=True)
        if terminal_keys.enabled:
            print(f"点击当前终端后按 s（无需回车），保存截至当前的全部状态为 TXT：{args.status_dir}", flush=True)
        else:
            print("当前输入不是交互终端，无法读取 s；请从终端运行 bash preview.sh 以手动保存。", flush=True)
        recorder.record(capture_terminal_state(env))
    seen_throws = 0
    completed = 0
    for _ in simulation_steps(env):
        with torch.inference_mode():
            before = env.throw_count.clone()
            obs, reward, terminated, truncated, extras = env.step(torch.zeros_like(env.actions))
            if args.mode == "preview":
                keys = terminal_keys.poll()
                recorder.record(extras["terminal"], force="s" in keys.lower())
                recorder.handle_keys(keys)
            seen_throws += int((env.throw_count > before).sum())
            if settings.shelf_task:
                completed += int(extras["shelf"]["success"].sum())
            if args.mode == "smoke":
                assert obs["policy"].shape == (env.num_envs, env.cfg.observation_space)
                assert torch.isfinite(obs["policy"]).all() and torch.isfinite(reward).all()
                assert len(extras["terminal"]) == env.num_envs
                for index, status in enumerate(extras["terminal"]):
                    assert status["terminated"] == bool(terminated[index])
                    assert status["timeout"] == bool(truncated[index])
    if args.mode == "smoke":
        assert seen_throws > 0, "没有发生抛掷：增加 --steps 或检查机器人是否过早跌倒"
        print(f"SMOKE PASSED: {seen_throws} 次抛掷，观测/奖励有限，手部 0 DOF", flush=True)
    if settings.shelf_task:
        print(f"接箱放架统计: 抛掷 {seen_throws}，完整任务成功 {completed}；当前运行仅发送零动作", flush=True)
except KeyboardInterrupt:
    pass
except Exception:
    import traceback
    traceback.print_exc()
    exit_code = 1
finally:
    import sys
    # 必须早于 env.close/app.close 恢复输入设置；退出不自动导出 TXT。
    if terminal_keys is not None:
        terminal_keys.close()
    if env is not None:
        env.close()
    # Kit 快速关闭可能直接退出进程；先刷新日志并保留失败退出码。
    sys.stdout.flush()
    sys.stderr.flush()
    if exit_code:
        import os
        os._exit(exit_code)
    app.close()
