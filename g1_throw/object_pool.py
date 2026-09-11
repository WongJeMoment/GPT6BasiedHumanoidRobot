"""未投放物体的隔离：避免地下刚体被地面的穿透恢复速度推回场景。"""


def park_inactive_objects(objects, origins, active_object):
    """每个物理步固定备用物体的位置；绝不写入当前投放物体的状态。

    仅在 reset 时移到地下并不能停用动态刚体：PhysX 会持续消除地面
    穿透，约十秒后物体重新冒出并撞到机器人。备用对象必须持续停放。
    """
    for index, obj in enumerate(objects):
        ids = (active_object != index).nonzero(as_tuple=False).flatten()
        if ids.numel() == 0:
            continue
        state = obj.data.default_root_state[ids].clone()
        state[:, :3] = origins[ids]
        state[:, 2] -= 20.0 + 2 * index
        state[:, 7:] = 0.0
        obj.write_root_state_to_sim(state, env_ids=ids)
