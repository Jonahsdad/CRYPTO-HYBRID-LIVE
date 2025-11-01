import json, hashlib, os
from datetime import datetime

LOG_DIR = "app_v3/logs"
os.makedirs(LOG_DIR, exist_ok=True)

def log_jsonl(filename: str, record: dict):
    path = os.path.join(LOG_DIR, filename)
    with open(path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path

def hash_file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    digest = h.hexdigest()
    # Write/update a .sha256 alongside the file for tamper evidence
    with open(path + ".sha256", "w") as f:
        f.write(digest)
    return digest

def audit_event(source: str, details: dict):
    # Standard envelope
    rec = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "source": source,
        "details": details
    }
    path = log_jsonl("data_log.jsonl", rec)
    # Optionally re-hash daily (cheap)
    hash_file_sha256(path)
