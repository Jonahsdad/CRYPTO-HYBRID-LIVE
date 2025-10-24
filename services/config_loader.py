import json, yaml, os
from jsonschema import validate, ValidationError

ROOT = os.path.dirname(os.path.dirname(__file__))

def _load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_config(name: str):
    cfg_path = os.path.join(ROOT, "config", f"{name}.yaml")
    schema_path = os.path.join(ROOT, "config", "schema", f"{name}.schema.json")
    cfg = _load_yaml(cfg_path)
    if os.path.exists(schema_path):
        schema = _load_json(schema_path)
        try:
            validate(instance=cfg, schema=schema)
        except ValidationError as e:
            raise ValueError(f"[CONFIG ERROR] {name}.yaml failed schema: {e.message}")
    return cfg

def load_all():
    return {
        "pdm_plus": load_config("pdm_plus"),
        "ctps": load_config("ctps"),
        "fir": load_config("fir"),
    }
