"""One-off: add v0 display names (README/result row names) to catalog model
files as deployment-level `display:` fields, so `bench render` can reproduce
the v0 table rows exactly. Imports the name maps the two catalog extractors
already maintain. Run: .venv/Scripts/python.exe legacy/add_display_names.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable

# regenerate the catalog first so display enrichment is part of one pipeline
# (note: this pass reformats the generated files; they are script-owned data)
for script in ("extract_catalog_hosted.py", "extract_catalog_local.py"):
    subprocess.run([PY, str(ROOT / "legacy" / script)], check=True, cwd=ROOT)

sys.path.insert(0, str(ROOT / "legacy"))
import extract_catalog_hosted as hosted  # noqa: E402
import extract_catalog_local as local  # noqa: E402

changed = 0
skipped: list[str] = []

for module in (hosted, local):
    for entry in module.MODELS:
        mid = entry.get("id") or f"{entry['org']}/{entry['slug']}"
        names = entry.get("_v0_names") or entry.get("v0_names") or []
        if not names:
            continue
        path = ROOT / "catalog" / "models" / f"{mid}.yaml"
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        deployments = doc.get("deployments") or []
        if len(names) == len(deployments):
            for dep, name in zip(deployments, names):
                dep["display"] = name
        elif len(names) == 1 and deployments:
            # single v0 row for the file: name the benched/primary deployment
            target = next(
                (d for d in deployments if (d.get("v0") or {}).get("benched")),
                deployments[0],
            )
            target["display"] = names[0]
        else:
            skipped.append(f"{mid}: {len(names)} names vs {len(deployments)} deployments")
            continue
        path.write_text(
            yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=100),
            encoding="utf-8",
        )
        changed += 1

print(f"display names written into {changed} model files")
for s in skipped:
    print(f"SKIPPED {s}")
