"""Remove generated data."""

import shutil
from pathlib import Path

from .utils import load_config


def main():
    cfg = load_config()
    paths = cfg["paths"]
    for key in ("chroma_dir", "cache_dir"):
        p = Path(paths[key])
        if p.exists():
            shutil.rmtree(p)
            print(f"Deleted {p}")
    for f in ("summary.md", "summary.pdf", "study.apkg"):
        fp = Path(f)
        if fp.exists():
            fp.unlink()
            print(f"Deleted {fp}")


if __name__ == "__main__":
    main()
