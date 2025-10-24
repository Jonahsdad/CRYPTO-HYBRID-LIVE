import re, os, yaml
ROOT = os.path.dirname(os.path.dirname(__file__))

def _yaml_path(name): return os.path.join(ROOT, "config", f"{name}.yaml")

def _load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _save_yaml(path, data):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)

def apply_ig_command(cmd: str):
    # Example: IG + ACTIVATE PDM+ [min_half_life=30d] [regime_awareness=on]
    m = re.match(r"\s*IG\s*\+\s*(ACTIVATE|UPDATE)\s+([A-Z0-9\+\-]+)\s*(.*)", cmd, re.I)
    if not m:
        raise ValueError("Unrecognized IG command format")
    action, module, rest = m.groups()
    module_key = {
        "PDM+": ("pdm_plus", "pdm_plus"),
        "CTPS": ("ctps", "ctps"),
        "FIR":  ("fir", "fir")
    }.get(module)
    if not module_key:
        raise ValueError(f"Unsupported module '{module}'")
    file_name, root_key = module_key
    cfg_path = _yaml_path(file_name)
    cfg = _load_yaml(cfg_path)
    # parse bracketed [key=value]
    for bracket in re.findall(r"\[(.*?)\]", rest):
        if "=" in bracket:
            k, v = bracket.split("=", 1)
            k = k.strip()
            v = v.strip()
            # normalize booleans and numbers; keep strings as-is
            if v.lower() in ("on", "true"): v_val = True
            elif v.lower() in ("off", "false"): v_val = False
            elif v.endswith(("d","m","h")): v_val = v  # keep duration-like strings
            else:
                try:
                    v_val = float(v) if "." in v else int(v)
                except ValueError:
                    v_val = v
            # place under the known root
            if root_key not in cfg: cfg[root_key] = {}
            cfg[root_key][k] = v_val
    _save_yaml(cfg_path, cfg)
    return f"Applied {action} to {module}: {rest}"
