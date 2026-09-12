"""在 PhysX 初始化前把 G1 的 14 个手指转动关节替换为固定关节。"""
import re

import isaaclab.sim as sim_utils
from isaaclab.sim.utils import clone, get_current_stage
from isaaclab_assets.robots.unitree import G1_CFG
from pxr import Usd, UsdPhysics, UsdShade

FINGER = re.compile(r"(left|right)_(zero|one|two|three|four|five|six)_joint$")


@clone
def spawn_fixed_hand_g1(prim_path, cfg, translation=None, orientation=None, **kwargs):
    root = sim_utils.spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)
    stage = get_current_stage()
    # 官方模型部分可视网格可能为 instance；解除实例才能覆盖手部材质。
    for prim in Usd.PrimRange(root):
        if prim.IsInstance():
            prim.SetInstanceable(False)
    joints = [p for p in Usd.PrimRange(root) if FINGER.fullmatch(p.GetName())]
    if len(joints) != 14:
        raise RuntimeError(f"预期官方 G1 的 14 个手指关节，实际找到 {len(joints)}；请检查 USD")
    hand_bodies = set()
    for prim in joints:
        joint = UsdPhysics.Joint(prim)
        hand_bodies.update(joint.GetBody0Rel().GetTargets())
        hand_bodies.update(joint.GetBody1Rel().GetTargets())
        # 包括手指根关节的父刚体（掌部），将掌部和全部指节一起着色。
        # 保留局部 joint frames，因此固定在 USD 的零角度手型。
        if prim.HasAPI(UsdPhysics.DriveAPI, "angular"):
            prim.RemoveAPI(UsdPhysics.DriveAPI, "angular")
        prim.SetTypeName("PhysicsFixedJoint")
    material_path = f"{prim_path}/BlackPlastic"
    material_cfg = sim_utils.PreviewSurfaceCfg(
        diffuse_color=cfg.hand_color, roughness=cfg.hand_roughness, metallic=0.0
    )
    material_cfg.func(material_path, material_cfg)
    material = UsdShade.Material(stage.GetPrimAtPath(material_path))
    for path in hand_bodies:
        body = stage.GetPrimAtPath(path)
        if body:
            UsdShade.MaterialBindingAPI.Apply(body).Bind(
                material, bindingStrength=UsdShade.Tokens.strongerThanDescendants
            )
    return root


def make_robot_cfg(settings):
    cfg = G1_CFG.copy()
    cfg.prim_path = "/World/envs/env_.*/Robot"
    cfg.spawn.func = spawn_fixed_hand_g1
    cfg.spawn.hand_color = settings.hand_color
    cfg.spawn.hand_roughness = settings.hand_roughness
    if settings.robot_usd:
        cfg.spawn.usd_path = settings.robot_usd
    cfg.init_state.joint_pos = {
        key: value for key, value in cfg.init_state.joint_pos.items() if not FINGER.fullmatch(key)
    }
    arms = cfg.actuators["arms"]
    arms.joint_names_expr = [".*_shoulder_.*", ".*_elbow_.*"]
    arms.armature = {".*_shoulder_.*": 0.01, ".*_elbow_.*": 0.01}
    if settings.controller == "hierarchical" or settings.shelf_task:
        # 来球飞行仅约 0.2 s，复位即采用预备抱接姿态，避免从垂臂开始追球。
        cfg.init_state.joint_pos.update({
            "left_shoulder_pitch_joint": -0.85,
            "right_shoulder_pitch_joint": -0.85,
            ".*_elbow_pitch_joint": 1.25,
            "left_shoulder_roll_joint": 0.25,
            "right_shoulder_roll_joint": -0.25,
        })
        # 保持官方力矩上限，增加接物时的位置跟踪与踝部支撑刚度。
        arms.stiffness = 100.0
        arms.damping = 8.0
        cfg.actuators["feet"].stiffness = 80.0
        cfg.actuators["feet"].damping = 6.0
    return cfg
