"""
LIPE / HIS Bridge Script
Scans repo structure and creates audit_snapshot.json
"""

import os, json, hashlib, time

def scan_repo(root="."):
    audit_map = {}
    for dirpath, _, files in os.walk(root):
        if "/.git" in dirpath or "/__pycache__" in dirpath:
            continue
        for f in files:
            path = os.path.join(dirpath, f)
            if not path.endswith((".py", ".yaml", ".yml", ".md", ".toml", ".json")):
                continue
            try:
                with open(path, "r", errors="ignore") as fh:
                    content = fh.read()
            except Exception as e:
                content = f"Error reading file: {e}"
            audit_map[path] = {
                "hash": hashlib.md5(content.encode()).hexdigest(),
                "size": len(content),
                "lines": content.count("\n"),
                "preview": content[:400]
            }
    snap = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "files": audit_map
    }
    with open("audit_snapshot.json","w") as out:
        json.dump(snap, out, indent=2)
    print("✅  audit_snapshot.json created")

if __name__ == "__main__":
    scan_repo()
