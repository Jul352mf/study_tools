"""Simple document count utility."""

from pathlib import Path

from .utils import load_config


def main():
    from llama_index.core import SimpleDirectoryReader
    cfg = load_config()
    docs_dir = Path(cfg["paths"]["docs_dir"])
    docs = SimpleDirectoryReader(str(docs_dir)).load_data()
    print(f"Loaded {len(docs)} docs")


if __name__ == "__main__":
    main()
