import json, os, tiktoken, pathlib

def get_tokenizer(default="gpt-3.5-turbo"):
    cfg = pathlib.Path("config.json")
    model = os.getenv("MODEL_NAME", default)
    if cfg.exists():
        try:
            model = json.loads(cfg.read_text())["model_name"]
        except Exception:
            pass
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")