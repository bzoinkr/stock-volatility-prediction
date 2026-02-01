from pathlib import Path
import yaml

from common.paths import CONFIG_DIR


def load_config(name: str):
    """
    Load a YAML config from configs/.

    Example:
        cfg = load_config("run.yaml")
    """
    path = Path(CONFIG_DIR) / name

    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with open(path, "r") as f:
        return yaml.safe_load(f)
