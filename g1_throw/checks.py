"""需要实际 PhysX 运行时的核心行为检查。"""
import torch


def check_launch_and_partial_reset(env):
    from isaaclab.sim.utils import get_current_stage
    from pxr import Usd, UsdPhysics, UsdShade
    from .robot import FINGER

    root = get_current_stage().GetPrimAtPath("/World/envs/env_0/Robot")
    fingers = [p for p in Usd.PrimRange(root) if FINGER.fullmatch(p.GetName())]
    assert len(fingers) == 14
    for prim in fingers:
        assert prim.IsA(UsdPhysics.FixedJoint)
        assert not prim.HasAPI(UsdPhysics.DriveAPI, "angular")
        joint = UsdPhysics.Joint(prim)
        for path in joint.GetBody0Rel().GetTargets() + joint.GetBody1Rel().GetTargets():
            body = get_current_stage().GetPrimAtPath(path)
            material, _ = UsdShade.MaterialBindingAPI(body).ComputeBoundMaterial()
            assert material and material.GetPrim().GetName() == "BlackPlastic", f"手部材质错误: {path}"
    ids = torch.tensor([0], device=env.device)
    untouched = env.robot.data.root_state_w[1:].clone()
    env._throw(ids)
    first = int(env.active_object[0])
    seen = {first}
    for attempt in range(max(8, len(env.objects) * 16)):
        previous = int(env.active_object[0])
        env._throw(ids)
        current = int(env.active_object[0])
        seen.add(current)
        if len(env.objects) > 1:
            assert previous != current, "相邻两次物体重复"
        state = env.objects[current].data.root_state_w[0]
        offset = state[:2] - env.robot.data.root_pos_w[0, :2]
        low, high = env.settings.distance_range
        if env.settings.launch_lateral_range is None:
            assert low - 1e-5 <= float(offset.norm()) <= high + 1e-5, "实际发射位置距机器人不在配置范围内"
        else:
            assert low - 1e-5 <= float(offset[0]) <= high + 1e-5, "发射 X 偏移超出配置范围"
            low, high = env.settings.launch_lateral_range
            assert low - 1e-5 <= float(offset[1]) <= high + 1e-5, "发射 Y 偏移超出配置范围"
        speed = state[7:10].norm()
        assert torch.allclose(speed, env.launch_speed[0], atol=1e-5), "初速度大小不符"
        low, high = env.settings.speed_range
        assert low - 1e-5 <= float(speed) <= high + 1e-5
        # 偏侧瞄准仍经过机器人根位置的 X 平面，不能用到中心的距离计算飞行时间。
        target = env.robot.data.root_pos_w[0, :2]
        flight_time = (target[0] - state[0]) / state[7]
        assert flight_time > 0, "弹道未朝向目标平面"
        lateral = state[1] + state[8] * flight_time - target[1]
        low, high = env.settings.target_lateral_range
        assert low - 1e-5 <= float(lateral) <= high + 1e-5, "横向目标偏移超出配置范围"
        impact_z = state[2] + state[9] * flight_time + 0.5 * env.cfg.sim.gravity[2] * flight_time**2
        assert abs(float(impact_z - env.scene.env_origins[0, 2]) - env.settings.target_height) < 1e-4
        if len(seen) == len(env.objects) and attempt >= 7:
            break
    assert len(seen) == len(env.objects), "未覆盖全部物体"
    # STL 缓存中的真实尺寸、中心和动态碰撞设置。
    from pxr import UsdGeom
    for obj, spec in zip(env.objects, env.settings.objects):
        masses = obj.root_physx_view.get_masses()
        assert torch.allclose(masses, torch.full_like(masses, spec.mass), atol=1e-5), f"质量设置未生效: {spec.name}"
        if spec.shape != "stl":
            continue
        mesh_prim = get_current_stage().GetPrimAtPath(f"/World/envs/env_0/Object_{spec.name}/geometry")
        assert UsdPhysics.MeshCollisionAPI(mesh_prim).GetApproximationAttr().Get() == "convexHull"
        points = torch.tensor(list(UsdGeom.Mesh(mesh_prim).GetPointsAttr().Get()))
        low, high = points.amin(0), points.amax(0)
        assert abs(float((high - low).max()) - spec.size[0]) < 1e-5
        assert (high + low).abs().max() < 1e-5
    env._reset_idx(ids)
    assert torch.equal(untouched, env.robot.data.root_state_w[1:]), "局部重置影响其他环境"
    assert int(env.active_object[0]) == -1
    if env.settings.ground_ruler:
        for i in range(env.num_envs):
            prim = get_current_stage().GetPrimAtPath(f"/World/envs/env_{i}/GroundRuler")
            assert prim and prim.GetCustomDataByKey("ruler_unit") == "dm"
            assert prim.GetCustomDataByKey("ruler_step_m") == .1
            assert not prim.HasAPI(UsdPhysics.CollisionAPI) and not prim.HasAPI(UsdPhysics.RigidBodyAPI)
    print(f"LAUNCH CHECK PASSED: 覆盖 {len(seen)} 个物体，种类切换/发射距离/速度/弹道/局部重置/STL 尺寸碰撞及地面标尺正常")


def check_shelf_task(env):
    """放架物理测试夹具：验证架面承重与终点逻辑，不代表策略自主完成了接箱。"""
    if not env.settings.shelf_task:
        return
    from isaaclab.sim.utils import get_current_stage
    from pxr import UsdPhysics
    task, shelf_cfg = env.shelf_task, env.settings.shelf
    ids = torch.tensor([0], device=env.device)
    # 在投放区域边界使用最慢速度、反向目标，检查固定计划、弹道及越界规则。
    original_plan = env.launch_plan
    try:
        s = env.settings
        corners = [(s.distance_range[1], None)] if s.launch_lateral_range is None else [
            (x, y) for x in s.distance_range for y in s.launch_lateral_range]
        for x, y in corners:
            env.reset()
            env.next_throw[:] = float("inf")
            target_y = max(s.target_lateral_range, key=lambda target: abs(target - (y or 0.)))
            values = {"object": 0, "distance": x, "speed": s.speed_range[0],
                      "angle": s.azimuth_range[0], "height": s.launch_height_range[0],
                      "target_lateral": target_y if y is not None else s.target_lateral_range[1]}
            if y is not None:
                values["launch_lateral"] = y
            env.launch_plan = {name: torch.full((env.num_envs,), value, device=env.device,
                              dtype=torch.long if name == "object" else torch.float32)
                              for name, value in values.items()}
            env._throw(ids)
            state = env.objects[0].data.root_state_w[0]
            offset = state[:2] - env.robot.data.root_pos_w[0, :2]
            if y is None:
                assert abs(float(offset.norm()) - x) < 1e-5
            else:
                torch.testing.assert_close(offset, torch.tensor([x, y], device=env.device), atol=1e-5, rtol=0)
            flight = (env.robot.data.root_pos_w[0, 0] - state[0]) / state[7]
            impact_y = state[1] + state[8] * flight - env.robot.data.root_pos_w[0, 1]
            assert abs(float(impact_y) - values["target_lateral"]) < 1e-4
            impact_z = state[2] + state[9] * flight + .5 * env.cfg.sim.gravity[2] * flight**2
            assert abs(float(impact_z - env.scene.env_origins[0, 2]) - s.target_height) < 1e-4
            _, _, terminated, truncated, extras = env.step(torch.zeros_like(env.actions))
            assert not terminated[0] and not truncated[0] and not extras["shelf"]["dropped"][0], "边界投放被立即误判失败"
    finally:
        env.launch_plan = original_plan
    env.reset()
    prior_target = env.joint_target.clone()
    env._pre_physics_step(torch.ones_like(env.actions))
    assert (env.joint_target - prior_target).abs().max() <= shelf_cfg.joint_speed * env.step_dt + 1e-6
    limits = env.robot.data.soft_joint_pos_limits
    assert (env.joint_target >= limits[..., 0] - 1e-6).all() and (env.joint_target <= limits[..., 1] + 1e-6).all()
    shelf = env.scene.rigid_objects["ShelfTop"]
    expected = torch.tensor(shelf_cfg.position, device=env.device).expand(env.num_envs, -1).clone()
    expected[:, 2] -= shelf_cfg.size[2] / 2
    torch.testing.assert_close(shelf.data.root_pos_w - env.scene.env_origins, expected, atol=1e-5, rtol=0)
    for name in ("ShelfTop", "ShelfLower", "ShelfLeg0", "ShelfLeg1", "ShelfLeg2", "ShelfLeg3"):
        prim = get_current_stage().GetPrimAtPath(f"/World/envs/env_0/{name}")
        assert UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr().Get(), f"架子未固定: {name}"

    def put_box_above_shelf(caught):
        env.reset()
        env.next_throw[:] = float("inf")
        env._throw(ids)
        # 只在测试夹具中布置箱体；运行环境没有瞬移箱体到架面的逻辑。
        state = env.objects[0].data.default_root_state[ids].clone()
        state[:, :3] = task.goal + env.scene.env_origins[ids]
        state[:, 2] += .025
        state[:, 7:] = 0
        env.objects[0].write_root_state_to_sim(state, env_ids=ids)
        task.caught[0] = caught
        task.best_distance[0] = .025

    def fixture_step():
        # 这里隔离架面物理与任务判据，固定测试机器人的站姿；零动作尚无平衡策略。
        # 只布置机器人，箱体继续自由积分，承重和松手必须由真实接触传感器验证。
        state = env.robot.data.default_root_state.clone()
        state[:, :3] += env.scene.env_origins
        env.robot.write_root_state_to_sim(state)
        env.robot.write_joint_state_to_sim(env.robot.data.default_joint_pos, env.robot.data.default_joint_vel)
        return env.step(torch.zeros_like(env.actions))

    steps = int((shelf_cfg.settle_seconds + .35) / env.step_dt) + 2
    put_box_above_shelf(False)
    supported = False
    for _ in range(steps):
        _, _, terminated, _, extras = fixture_step()
        metrics = extras["shelf"]
        supported |= bool(metrics["supported"][0])
        assert not metrics["success"][0], "箱子直接落架被误判为接箱放架成功"
        assert not terminated[0], "放架夹具在接触验证前发生意外终止"
    assert supported, "架面没有检测到实际承重"
    bottom = env.objects[0].data.root_pos_w[0, 2] - task.half_size[2] - env.scene.env_origins[0, 2]
    assert abs(float(bottom) - shelf_cfg.position[2]) < shelf_cfg.height_tolerance, "箱体穿过架面"

    put_box_above_shelf(True)
    completed = False
    for _ in range(steps):
        obs, reward, terminated, truncated, extras = fixture_step()
        if terminated[0] or truncated[0]:
            metrics = extras["shelf"]
            assert metrics["success"][0] and terminated[0] and not truncated[0], "松手放稳未成功终止"
            assert metrics["supported"][0] and not metrics["robot_touch"][0]
            assert metrics["settle_time"][0] >= shelf_cfg.settle_seconds and reward[0] > 90
            assert not task.caught[0] and not task.success[0] and env.active_object[0] == -1
            if env.terminal_status_enabled:
                status = extras["terminal"][0]
                assert status["success"] and status["caught"] and status["supported"]
                assert status["active"] == 0 and not status["robot_touch"]
            assert torch.isfinite(obs["policy"]).all()
            if env.num_envs > 1:
                assert not (terminated[1:] | truncated[1:]).any(), "放架成功误重置其他环境"
                assert (env.episode_length_buf[1:] > 0).all()
            completed = True
            break
    assert completed, "放架成功未在预期时间内触发"

    put_box_above_shelf(True)
    state = env.objects[0].data.root_state_w[ids].clone()
    state[:, 0:2] = env.scene.env_origins[ids, :2] + torch.tensor([1.0, 0.0], device=env.device)
    state[:, 2] = env.scene.env_origins[ids, 2] + task.half_size[2] + .005
    state[:, 7:] = 0
    env.objects[0].write_root_state_to_sim(state, env_ids=ids)
    _, reward, terminated, _, extras = env.step(torch.zeros_like(env.actions))
    assert terminated[0] and extras["shelf"]["dropped"][0] and not extras["shelf"]["success"][0]
    assert reward[0] < -59 and env.active_object[0] == -1
    if env.terminal_status_enabled:
        status = extras["terminal"][0]
        assert status["dropped"] and status["terminated"] and status["active"] == 0
        assert status["bottom"] <= shelf_cfg.ground_tolerance
        assert status["box"][2] > 0, "终端快照错误地记录了复位后停放到地下的箱体"
    for _ in range(int(env.settings.first_throw_delay / env.step_dt) + 2):
        env.step(torch.zeros_like(env.actions))
    assert env.active_object[0] == 0, "掉落重置后未重新投放箱体"
    env.reset()
    env.next_throw[:] = float("inf")
    env.episode_length_buf[0] = env.max_episode_length - 2
    _, reward, terminated, truncated, extras = fixture_step()
    assert truncated[0] and not terminated[0] and not extras["shelf"]["success"][0]
    assert reward[0] <= -30, "超时空手站立未判为失败"
    env.reset()
    print("SHELF CHECK PASSED: 动作限幅限速/固定架体/跨环境位置/真实承重/直接落架不成功/放稳成功及局部复位/掉落失败再投放/超时失败；接住历史及机器人站姿为测试夹具", flush=True)
