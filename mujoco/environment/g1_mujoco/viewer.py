"""等待 MuJoCo passive viewer 完成 GL 清理后再退出 Python。"""
from contextlib import contextmanager
import threading


@contextmanager
def passive_viewer(model, data, **kwargs):
    import mujoco.viewer

    # MuJoCo 3.10 的 Handle.close 仅发出退出请求；CLI 独占此次 viewer 启动。
    # 保存它创建的 Python 线程并 join，避免解释器先调用 glfw.terminate 导致段错误。
    existing = set(threading.enumerate())
    viewer = mujoco.viewer.launch_passive(model, data, **kwargs)
    threads = set(threading.enumerate()) - existing
    try:
        yield viewer
    finally:
        viewer.close()
        for thread in threads:
            thread.join(timeout=5.)
            if thread.is_alive():
                raise RuntimeError("MuJoCo 可视化线程未在 5 秒内关闭")
