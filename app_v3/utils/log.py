import os, json, hashlib
from datetime import datetime
from utils.settings import settings

os.makedirs(settings.logs_dir, exist_ok=True)
LOG_PATH = os.path.join(settings.logs_dir, "data_log.jsonl")

def audit_event(source: str, **details):
    rec = {"ts": datetime.utcnow().isoformat()+"Z", "source": source, "details": details}
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    h = hashlib.sha256()
    with open(LOG_PATH, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""): h.update(chunk)
    with open(LOG_PATH + ".sha256", "w") as f:
        f.write(h.hexdigest())
