"""验证备用物体隔离不干预实际投放物体，包括已掉落的物体。"""
from types import SimpleNamespace
import unittest
import torch
from g1_throw.object_pool import park_inactive_objects


class FakeObject:
    def __init__(self):
        self.state = torch.randn(3, 13)
        self.data = SimpleNamespace(default_root_state=torch.zeros(3, 13))
        self.data.default_root_state[:, 3] = 1

    def write_root_state_to_sim(self, state, env_ids):
        self.state[env_ids] = state


class ObjectPoolTests(unittest.TestCase):
    def test_only_inactive_objects_are_reparked(self):
        objects = [FakeObject(), FakeObject()]
        original = [o.state.clone() for o in objects]
        origins = torch.tensor([[0., 0., 0.], [8., 0., 0.], [16., 0., 0.]])
        active = torch.tensor([0, 1, -1])
        park_inactive_objects(objects, origins, active)
        torch.testing.assert_close(objects[0].state[0], original[0][0])
        torch.testing.assert_close(objects[1].state[1], original[1][1])
        for index, obj in enumerate(objects):
            ids = active != index
            torch.testing.assert_close(obj.state[ids, :2], origins[ids, :2])
            self.assertTrue((obj.state[ids, 2] == -20 - 2 * index).all())
            self.assertTrue((obj.state[ids, 7:] == 0).all())


if __name__ == "__main__":
    unittest.main()
