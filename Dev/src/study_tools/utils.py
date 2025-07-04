from pathlib import Path

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"

_cached_cfg = None


def load_config(path: Path | str = DEFAULT_CONFIG_PATH):
    global _cached_cfg
    if _cached_cfg is None or path != DEFAULT_CONFIG_PATH:
        import yaml
        with open(path, "r", encoding="utf-8") as fh:
            _cached_cfg = yaml.safe_load(fh)
    return _cached_cfg
