"""运行入口：先启动 Isaac Sim，再导入训练环境。"""
import argparse
import itertools
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="G1 黑色零自由度手 / 随机物体抛掷")
parser.add_argument("--config", default="env_configs/single_throw.py")
parser.add_argument("--mode", choices=("preview", "train", "play", "smoke"), default="preview")
parser.add_argument("--num_envs", type=int)
parser.add_argument("--steps", type=int, default=None, help="运行步数；0 为持续运行，预览/回放默认持续运行")
parser.add_argument("--iterations", type=int, default=1500)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--checkpoint")
parser.add_argument("--controller", choices=("joint", "hierarchical"), help="覆盖场景中的控制方式")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.steps is None:
    args.steps = 0 if args.mode in ("preview", "play") and not args.headless else 600
if args.steps < 0:
    parser.error("--steps 必须 >= 0")


def simulation_steps(env):
    """可视化按实时速度运行，直到关闭窗口或达到显式指定的步数。"""
    for index in itertools.count():
        if not app.is_running() or (args.steps and index >= args.steps):
            return
        start = time.monotonic()
        yield index
        if env.sim.has_gui() and args.mode in ("preview", "play"):
            time.sleep(max(0.0, env.step_dt - (time.monotonic() - start)))

from g1_throw.config_loader import load_settings

settings = load_settings(args.config)
if args.num_envs is not None:
    settings.num_envs = args.num_envs
if args.controller is not None:
    settings.controller = args.controller
settings.validate()
if args.mode == "play" and not args.checkpoint:
    parser.error("--mode play 需要 --checkpoint 路径")
launcher = AppLauncher(args)
app = launcher.app
env = None
exit_code = 0
try:
    import torch
    from g1_throw.environment import G1ThrowEnv, make_env_cfg

    env = G1ThrowEnv(make_env_cfg(settings, args.device or "cuda:0", args.seed))
    print(f"G1: {env.robot.num_joints} 个身体关节；手部 0 自由度；物体 {[o.name for o in settings.objects]}")
    print(f"控制方式: {settings.controller}", flush=True)
    if args.mode in ("train", "play"):
        from datetime import datetime
        import shutil
        from rsl_rl.runners import OnPolicyRunner
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from g1_throw.ppo import PPORunnerCfg

        log_dir = Path("logs") / Path(args.config).stem / datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.config, log_dir / "environment_config.py")
        from dataclasses import asdict
        import json
        (log_dir / "resolved_settings.json").write_text(json.dumps(asdict(settings), indent=2))
        agent_cfg = PPORunnerCfg()
        agent_cfg.seed = args.seed
        wrapped = RslRlVecEnvWrapper(env, clip_actions=1.0)
        runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=str(log_dir), device=env.device)
        if args.checkpoint:
            runner.load(args.checkpoint)
        if args.mode == "train":
            runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=False)
        else:
            policy = runner.get_inference_policy(device=env.device)
            obs = wrapped.get_observations()
            for _ in simulation_steps(env):
                with torch.inference_mode():
                    obs, _, _, _ = wrapped.step(policy(obs))
    else:
        obs, _ = env.reset()
        if args.mode == "smoke":
            from g1_throw.checks import check_launch_and_partial_reset
            check_launch_and_partial_reset(env)
        seen_throws = 0
        caught = 0
        catch_latched = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        visited_phases = set()
        if args.mode == "preview":
            print("预览已启动：关闭窗口或按 Ctrl+C 退出。", flush=True)
        for _ in simulation_steps(env):
            with torch.inference_mode():
                before = env.throw_count.clone()
                obs, reward, terminated, truncated, _ = env.step(torch.zeros_like(env.actions))
                seen_throws += int((env.throw_count > before).sum())
                if env.catch_controller is not None:
                    metrics = env.extras.get("catch", {})
                    success = metrics.get("success", torch.zeros_like(catch_latched))
                    catch_latched[env.throw_count > before] = False
                    caught += int((success & ~catch_latched).sum())
                    catch_latched = success.clone()
                    catch_latched[terminated | truncated] = False
                    visited_phases.update(metrics["phase"].tolist())
                if args.mode == "smoke":
                    assert obs["policy"].shape == (env.num_envs, env.cfg.observation_space)
                    assert torch.isfinite(obs["policy"]).all() and torch.isfinite(reward).all()
                    if env.catch_controller is not None:
                        target = env.joint_target
                        limits = env.robot.data.soft_joint_pos_limits
                        assert torch.isfinite(target).all(), "控制器生成了非有限关节目标"
                        # 实际发送前还会裁剪；这里检查名义控制器本身没有越过软限位。
                        nominal = env.catch_controller.motor.target
                        assert (nominal >= limits[..., 0] - 1e-5).all()
                        assert (nominal <= limits[..., 1] + 1e-5).all()
        if args.mode == "smoke":
            assert seen_throws > 0, "没有发生抛掷：增加 --steps 或检查机器人是否过早跌倒"
            print(f"SMOKE PASSED: {seen_throws} 次抛掷，观测/奖励有限，手部 0 DOF", flush=True)
        if env.catch_controller is not None:
            from controllers.planner import Phase
            print(f"接物统计: 抛掷 {seen_throws}，成功抱稳 {caught}，阶段 {[Phase(p).name for p in sorted(visited_phases)]}", flush=True)
except KeyboardInterrupt:
    pass
except Exception:
    import traceback
    traceback.print_exc()
    exit_code = 1
finally:
    import sys
    if env is not None:
        env.close()
    # Kit 快速关闭可能直接退出进程；先刷新日志并显式保留失败退出码。
    sys.stdout.flush()
    sys.stderr.flush()
    if exit_code:
        import os
        os._exit(exit_code)
    app.close()
