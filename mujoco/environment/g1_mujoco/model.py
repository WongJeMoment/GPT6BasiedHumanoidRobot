"""从固定版本的 G1 MJCF 构建场景；配置沿用 env_configs。"""
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ASSETS = Path(__file__).resolve().parents[1] / "assets" / "unitree_g1"
FINGER = re.compile(r"(left|right)_(zero|one|two|three|four|five|six)_joint$")

# 2026-09-16 在当前 Isaac Lab G1ThrowEnv 实测的关节顺序（不是 MJCF 深度优先顺序）。
JOINT_NAMES = (
    "left_hip_pitch_joint", "right_hip_pitch_joint", "torso_joint",
    "left_hip_roll_joint", "right_hip_roll_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_knee_joint", "right_knee_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_elbow_pitch_joint", "right_elbow_pitch_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_elbow_roll_joint", "right_elbow_roll_joint",
)


def numbers(values):
    return " ".join(f"{value:.10g}" for value in np.asarray(values).reshape(-1))


def default_pose(settings):
    result = np.zeros(23)
    for index, name in enumerate(JOINT_NAMES):
        if "hip_pitch" in name:
            result[index] = -.20
        elif "knee" in name:
            result[index] = .42
        elif "ankle_pitch" in name:
            result[index] = -.23
        elif "elbow_pitch" in name:
            result[index] = 1.25 if settings.shelf_task else .87
        elif "shoulder_pitch" in name:
            result[index] = -.85 if settings.shelf_task else .35
        elif "shoulder_roll" in name:
            result[index] = (.25 if settings.shelf_task else .16) * (1 if name.startswith("left") else -1)
    return result


def _actuator_parameters(name, shelf_task):
    if "ankle" in name:
        return (80., 6., 20.) if shelf_task else (20., 2., 20.)
    if "shoulder" in name or "elbow" in name:
        return (100., 8., 300.) if shelf_task else (40., 10., 300.)
    return (150. if "hip_roll" in name or "hip_yaw" in name else 200., 5., 300.)


def _add_ruler(world, asset, settings):
    from g1_throw.ground_ruler import ruler_geometry

    vertices, counts, indices, colors = ruler_geometry(settings)
    vertices = np.asarray(vertices)
    cursor = 0
    for index, (count, color) in enumerate(zip(counts, colors)):
        points = vertices[indices[cursor:cursor + count]]
        cursor += count
        edge, side = points[1] - points[0], points[-1] - points[0]
        center = points.mean(axis=0)
        attributes = dict(name=f"ruler_{index}", contype="0", conaffinity="0", group="1",
                          density="0", rgba=numbers((*color, 1)))
        if abs(np.dot(edge, side)) < 1e-8:
            ET.SubElement(world, "geom", **attributes, type="box", pos=numbers(center),
                          size=numbers((np.linalg.norm(edge) / 2, np.linalg.norm(side) / 2, .00015)),
                          euler=numbers((0, 0, np.arctan2(edge[1], edge[0]))))
        else:
            # 自定义扇形投放区域的四边形；极薄棱柱保证 MuJoCo 网格有非零体积。
            prism = np.concatenate((points - [0, 0, .00015], points + [0, 0, .00015]))
            name = f"ruler_mesh_{index}"
            ET.SubElement(asset, "mesh", name=name, vertex=numbers(prism))
            ET.SubElement(world, "geom", **attributes, type="mesh", mesh=name)


def build_model(settings):
    """返回模型和生成的 XML；构建过程中不创建日志、缓存或其他文件。"""
    root = ET.parse(ASSETS / "g1.xml").getroot()
    root.set("model", "G1_throw_catch_and_place")
    root.find("compiler").set("meshdir", str(ASSETS / "assets"))
    root.find("compiler").set("fusestatic", "false")
    # 去掉原始 37 关节 keyframe 和力矩驱动，重新定义固定手部的 23 维位置控制。
    for tag in ("keyframe", "actuator", "sensor", "contact"):
        element = root.find(tag)
        if element is not None:
            root.remove(element)
    ET.SubElement(root, "option", timestep=str(settings.physics_dt), gravity="0 0 -9.81",
                  integrator="implicitfast", cone="elliptic", iterations="80", tolerance="1e-9")
    root.find("default/default/joint").attrib.update(damping="0", frictionloss="0", armature="0.01")
    collision = root.find("default/default/default[@class='collision']/geom")
    collision.attrib.update(contype="1", conaffinity="6", friction="0.6 0.005 0.0001", condim="3")
    asset, world = root.find("asset"), root.find("worldbody")
    ET.SubElement(asset, "material", name="fixed_hand_black", rgba=numbers((*settings.hand_color, 1)),
                  specular="0.15", shininess="0.1")
    pelvis = world.find("body[@name='pelvis']")
    pelvis.set("pos", "0 0 0.74")
    removed = 0
    for body in pelvis.iter("body"):
        for joint in list(body.findall("joint")):
            if FINGER.fullmatch(joint.get("name", "")):
                body.remove(joint)
                removed += 1
        if re.fullmatch(r"(left|right)_(zero|one|two|three|four|five|six)_link", body.get("name", "")):
            for geom in body.findall("geom"):
                geom.set("material", "fixed_hand_black")
    if removed != 14:
        raise ValueError(f"G1 原始模型应有 14 个手指关节，实际 {removed}")
    # 原版 MJCF 把掌部网格放在肘部刚体中；分为固定子刚体方便逐部位报告接触。
    # 掌部惯量已经包含在原版肘部惯量中，因此这里不额外增加质量。
    for side in ("left", "right"):
        elbow = pelvis.find(f".//body[@name='{side}_elbow_roll_link']")
        palm = ET.SubElement(elbow, "body", name=f"{side}_palm_link", pos="0.12 0 0")
        for geom in list(elbow.findall("geom")):
            if geom.get("mesh") == f"{side}_palm_link":
                elbow.remove(geom)
                geom.set("pos", "0 0 0")
                geom.set("density", "0")
                geom.set("material", "fixed_hand_black")
                palm.append(geom)
    # 原始 MJCF 的头/标志网格也挂在 torso 下，分开命名以免把头部撞箱记作躯干接触。
    torso = pelvis.find(".//body[@name='torso_link']")
    for name in ("head_link", "logo_link"):
        fixed = ET.SubElement(torso, "body", name=name)
        for geom in list(torso.findall("geom")):
            if geom.get("mesh") == name:
                torso.remove(geom)
                geom.set("density", "0")
                fixed.append(geom)
    actuator = ET.SubElement(root, "actuator")
    for name in JOINT_NAMES:
        kp, kd, effort = _actuator_parameters(name, settings.shelf_task)
        joint = pelvis.find(f".//joint[@name='{name}']")
        if joint is None:
            raise ValueError(f"G1 模型缺少关节: {name}")
        ET.SubElement(actuator, "position", name=name, joint=name, kp=str(kp), kv=str(kd),
                      forcelimited="true", forcerange=numbers((-effort, effort)))
    ET.SubElement(world, "geom", name="ground", type="plane", size="200 200 .1", rgba=".08 .10 .12 1",
                  contype="2", conaffinity="5", friction="0.8 0.005 0.0001")
    ET.SubElement(world, "light", pos="3 -3 6", dir="-.4 .4 -1", diffuse=".7 .7 .7", directional="true")
    root.find("visual/global").attrib.update(offwidth="1280", offheight="960")
    if settings.shelf_task:
        cfg = settings.shelf
        x, y, z = cfg.position
        sx, sy, thickness = cfg.size
        parts = [("ShelfTop", cfg.size, (x, y, z - thickness / 2)),
                 ("ShelfLower", cfg.size, (x, y, cfg.lower_height - thickness / 2))]
        for i, (dx, dy) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1))):
            parts.append((f"ShelfLeg{i}", (cfg.leg_width, cfg.leg_width, z - thickness),
                          (x + dx * (sx - cfg.leg_width) / 2, y + dy * (sy - cfg.leg_width) / 2,
                           (z - thickness) / 2)))
        for name, size, position in parts:
            body = ET.SubElement(world, "body", name=name, pos=numbers(position))
            ET.SubElement(body, "geom", name=name, type="box", size=numbers(np.asarray(size) / 2),
                          contype="2", conaffinity="5", friction="0.8 0.005 0.0001",
                          rgba=".16 .18 .20 1" if "Leg" in name else ".55 .38 .20 1")
    for index, spec in enumerate(settings.objects):
        name = f"Object_{spec.name}"
        body = ET.SubElement(world, "body", name=name, pos=f"0 0 {-20 - 2 * index}")
        ET.SubElement(body, "freejoint", name=f"{name}_joint")
        attributes = dict(name=name, mass=str(spec.mass), rgba=numbers((*spec.color, 1)),
                          contype="4", conaffinity="3", friction="0.6 0.005 0.0001", condim="3")
        if spec.shape == "stl":
            import trimesh

            path = Path(spec.mesh_path).expanduser()
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            mesh = trimesh.load_mesh(path, process=True)
            vertices = (mesh.vertices - mesh.bounds.mean(axis=0)) * spec.size[0] / mesh.extents.max()
            # MuJoCo 用凸包碰撞，与当前 PhysX STL 配置的语义一致。
            ET.SubElement(asset, "mesh", name=name, vertex=numbers(vertices), face=numbers(mesh.faces))
            ET.SubElement(body, "geom", **attributes, type="mesh", mesh=name)
        else:
            shape = {"cuboid": "box", "sphere": "sphere", "cylinder": "cylinder"}[spec.shape]
            size = (np.asarray(spec.size) / 2 if shape == "box" else
                    (spec.size[0], spec.size[1] / 2) if shape == "cylinder" else spec.size)
            ET.SubElement(body, "geom", **attributes, type=shape, size=numbers(size))
    if settings.ground_ruler:
        _add_ruler(world, asset, settings)
    xml = ET.tostring(root, encoding="unicode")
    model = mujoco.MjModel.from_xml_string(xml)
    if model.nu != 23 or any(FINGER.fullmatch(model.joint(i).name) for i in range(model.njnt)):
        raise ValueError("MuJoCo G1 必须是 23 个身体关节、手部 0 DOF")
    return model, xml
