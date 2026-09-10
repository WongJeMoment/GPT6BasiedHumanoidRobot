"""STL → 居中、米制 USD 网格缓存；PhysX 使用凸包碰撞。"""
import hashlib
from pathlib import Path

import numpy as np
import trimesh
from pxr import Usd, UsdGeom, UsdPhysics, Vt

from isaaclab.sim.utils import clone, bind_physics_material
import isaaclab.sim as sim_utils

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@clone
def spawn_stl(prim_path, cfg, translation=None, orientation=None, **kwargs):
    prim = sim_utils.spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)
    material_path = prim_path + "/PhysicsMaterial"
    cfg.throw_physics_material.func(material_path, cfg.throw_physics_material)
    bind_physics_material(prim_path + "/geometry", material_path)
    return prim


def prepare_stl(spec):
    path = Path(spec.mesh_path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    digest = hashlib.sha256(path.read_bytes() + repr((spec.size, "stl-v2")).encode()).hexdigest()[:20]
    cache = PROJECT_ROOT / "assets" / "usd_cache"
    cache.mkdir(parents=True, exist_ok=True)
    output = cache / f"{digest}.usdc"
    if output.exists():
        return str(output)
    mesh = trimesh.load_mesh(path, process=True)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError(f"STL 没有有效三角网格: {path}")
    if not np.isfinite(mesh.vertices).all() or np.any(mesh.extents <= 0):
        raise ValueError(f"STL 尺寸或顶点无效: {path}")
    vertices = (mesh.vertices - mesh.bounds.mean(axis=0)) * (spec.size[0] / mesh.extents.max())
    stage = Usd.Stage.CreateNew(str(output))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Object")
    stage.SetDefaultPrim(root.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(root.GetPrim())
    UsdPhysics.MassAPI.Apply(root.GetPrim())
    geometry = UsdGeom.Mesh.Define(stage, "/Object/geometry")
    geometry.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(vertices.astype(np.float32)))
    geometry.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(mesh.faces), 3, dtype=np.int32)))
    geometry.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(mesh.faces.astype(np.int32).reshape(-1)))
    geometry.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    geometry.CreateDoubleSidedAttr(True)
    UsdPhysics.CollisionAPI.Apply(geometry.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(geometry.GetPrim()).CreateApproximationAttr("convexHull")
    stage.GetRootLayer().Save()
    print(f"STL 已转换: {path.name} → 最长边 {spec.size[0]:.3f} m，{len(mesh.faces)} 个三角面", flush=True)
    return str(output)
