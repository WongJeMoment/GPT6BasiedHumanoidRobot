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
    controller = env.catch_controller
    if controller is not None:
        # 写入独立任务状态，确认局部复位不串扰其他环境。
        controller.planner.elapsed[1:] = 0.123
        controller.planner.success[1:] = True
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
        speed = state[7:10].norm()
        assert torch.allclose(speed, env.launch_speed[0], atol=1e-5), "初速度大小不符"
        low, high = env.settings.speed_range
        assert low - 1e-5 <= float(speed) <= high + 1e-5
        target = env.robot.data.root_pos_w[0, :2]
        flight_time = (target - state[:2]).norm() / state[7:9].norm()
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
    if controller is not None:
        assert int(controller.planner.phase[0]) == 0
        assert not controller.planner.success[0]
        assert controller.planner.success[1:].all()
        assert (controller.planner.elapsed[1:] == 0.123).all()
        # 检查结束后清除测试注入的状态，避免污染实际成功统计。
        controller.reset(env.indices)
    print(f"LAUNCH CHECK PASSED: 覆盖 {len(seen)} 个物体，种类切换/速度/弹道/局部重置/STL 尺寸碰撞正常")
