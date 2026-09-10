from copy import deepcopy
import importlib.util
from pathlib import Path


def load_settings(path):
    path = Path(path).expanduser().resolve()
    spec = importlib.util.spec_from_file_location("user_environment_config", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"无法加载配置: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    settings = deepcopy(module.CONFIG)
    settings.validate()
    return settings
