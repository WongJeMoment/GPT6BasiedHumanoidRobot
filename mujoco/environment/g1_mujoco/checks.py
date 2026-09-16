"""MuJoCo 真实物理检查；放架测试注入接住历史，不代表已实现控制策略。"""
import numpy as np
import torch

from .model import JOINT_NAMES


def check_launch_and_reset(env):
    assert env.model.nu == 23 and env.robot.joint_names == list(JOINT_NAMES)
    assert env.model.nv == 6 + 23 + 6 * len(env.objects), "手指关节没有完全固定"
    for spec, body in zip(env.settings.objects, env.object_body_ids):
        assert abs(env.model.body_mass[body] - spec.mass) < 1e-6
    env.reset(seed=42)
    untouched = [data.qpos.copy() for data in env.datas[1:]]
    ids = torch.tensor([0])
    seen = set()
    for _ in range(max(8, 8 * len(env.objects))):
        previous = int(env.last_object[0])
        env._throw(ids)
        current = int(env.active_object[0])
        seen.add(current)
        assert len(env.objects) == 1 or previous != current, "相邻投放种类重复"
        state = env.objects[current].data.root_state_w[0]
        offset = state[:2] - env.robot.data.root_pos_w[0, :2]
        low, high = env.settings.distance_range
        if env.settings.launch_lateral_range is None:
            assert low - 1e-5 <= offset.norm() <= high + 1e-5
        else:
            assert low - 1e-5 <= offset[0] <= high + 1e-5
            low, high = env.settings.launch_lateral_range
            assert low - 1e-5 <= offset[1] <= high + 1e-5
        torch.testing.assert_close(state[7:10].norm(), env.launch_speed[0], atol=1e-5, rtol=0)
        flight = -offset[0] / state[7]
        assert flight > 0
        impact = state[:3] + state[7:10] * flight
        impact[2] -= .5 * abs(env.model.opt.gravity[2]) * flight.square()
        lateral = impact[1] - env.robot.data.root_pos_w[0, 1]
        low, high = env.settings.target_lateral_range
        assert low - 1e-5 <= lateral <= high + 1e-5
        assert abs(float(impact[2]) - env.settings.target_height) < 1e-4
    assert len(seen) == len(env.objects), "没有覆盖全部投放物体"
    env.reset(env_ids=ids)
    for before, data in zip(untouched, env.datas[1:]):
        np.testing.assert_array_equal(before, data.qpos)
    if env.settings.ground_ruler:
        geoms = [i for i in range(env.model.ngeom) if env.model.geom(i).name.startswith("ruler_")]
        assert geoms and not env.model.geom_contype[geoms].any() and not env.model.geom_conaffinity[geoms].any()
    print("LAUNCH CHECK PASSED: 23 关节/固定手部/物体质量/投放范围与弹道/相邻种类不重复/局部复位/标尺无碰撞", flush=True)


def hold_robot_fixture(env):
    """仅测试用：每步重布机器人站姿，箱子始终进行真实自由积分。"""
    for index, data in enumerate(env.datas):
        data.qpos[:7] = env.model.qpos0[:7]
        data.qpos[env.joint_qpos] = env.robot.data.default_joint_pos[index].numpy()
        data.qvel[:6] = 0
        data.qvel[env.joint_dofs] = 0
    env.forward()
    return env.step(torch.zeros_like(env.actions))


def check_shelf_physics(env):
    task = env.shelf_task
    if task is None:
        return
    env.reset()
    initial = env.joint_target.clone()
    env.step(torch.ones_like(env.actions))
    assert (env.joint_target - initial).abs().max() <= task.cfg.joint_speed * env.step_dt + 1e-6
    for name in ("ShelfTop", "ShelfLower", "ShelfLeg0", "ShelfLeg1", "ShelfLeg2", "ShelfLeg3"):
        assert env.model.body_jntnum[env.model.body(name).id] == 0
    np.testing.assert_allclose(env.model.geom_size[env.shelf_top_geom], np.asarray(task.cfg.size) / 2)

    def place(caught):
        env.reset()
        env._throw([0])
        env.next_throw[:] = float("inf")
        env._set_object_state(0, 0, task.goal.numpy() + [0., 0., .025])
        task.caught[0] = caught
        task.best_distance[0] = .025
        env.forward()

    steps = int((task.cfg.settle_seconds + .5) / env.step_dt) + 2
    place(False)
    supported = False
    for _ in range(steps):
        _, _, ended, _, info = hold_robot_fixture(env)
        supported |= bool(info["shelf"]["supported"][0])
        assert not ended[0] and not info["shelf"]["success"][0], "直接落架不能算接箱成功"
    assert supported, "没有检测到箱体对架面的真实承重，请核对接触力方向"
    bottom = env.objects[0].data.root_pos_w[0, 2] - task.half_size[2]
    assert abs(float(bottom) - task.cfg.position[2]) < task.cfg.height_tolerance

    place(True)
    completed = False
    for _ in range(steps):
        _, reward, ended, timeout, info = hold_robot_fixture(env)
        if ended[0] or timeout[0]:
            snapshot = info["terminal"][0]
            assert ended[0] and not timeout[0] and info["shelf"]["success"][0]
            assert reward[0] > 90 and snapshot["supported"] and not snapshot["robot_touch"]
            assert snapshot["active"] == 0 and env.active_object[0] == -1
            assert snapshot["caught"] and not task.caught[0]
            if env.num_envs > 1:
                assert not (ended[1:] | timeout[1:]).any()
            completed = True
            break
    assert completed, "松手放稳未触发成功"

    place(True)
    env._set_object_state(0, 0, (1., 0., float(task.half_size[2]) + .005))
    env.forward()
    _, reward, ended, _, info = hold_robot_fixture(env)
    assert ended[0] and info["shelf"]["dropped"][0] and reward[0] < -59
    assert info["terminal"][0]["box"][2] > 0 and env.active_object[0] == -1
    for _ in range(int(env.settings.first_throw_delay / env.step_dt) + 2):
        hold_robot_fixture(env)
    assert env.active_object[0] == 0, "掉落复位后没有重新投放"

    env.reset()
    env.next_throw[:] = float("inf")
    env.episode_length_buf[0] = env.max_episode_length - 2
    _, reward, ended, timeout, info = hold_robot_fixture(env)
    assert timeout[0] and not ended[0] and reward[0] <= -30
    assert info["terminal"][0]["timeout"]
    env.reset()
    print("SHELF CHECK PASSED: 限幅限速/固定架体/真实架面承重/直接落架不成功/放稳成功/掉落与超时/复位前快照", flush=True)
