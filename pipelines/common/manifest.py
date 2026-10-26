"""Manifeste d'ingestion (spec 4.1).

Stocké en JSONL append-only (`data/meta/ingest_manifest/<pipeline>.jsonl`) plutôt qu'en Parquet :
chaque unité est enregistrée dès qu'elle est finie, donc un arrêt brutal ne perd rien.
Lecture directe dans DuckDB : `SELECT * FROM read_json_auto('data/meta/ingest_manifest/*.jsonl')`.
Le statut courant d'une unité est sa dernière ligne.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from pipelines.common.storage import Layout, append_jsonl, read_jsonl

_PKG_DIR = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def code_version() -> str:
    """Hash git court, suffixé de '-dirty' si l'arbre est modifié ; sinon hash des sources."""
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=_PKG_DIR, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--", "."], cwd=_PKG_DIR, capture_output=True, text=True
        ).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        h = hashlib.sha1()
        for p in sorted(_PKG_DIR.rglob("*.py")):
            h.update(p.read_bytes())
        return "nogit-" + h.hexdigest()[:10]


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class Manifest:
    def __init__(self, layout: Layout, pipeline: str):
        self.pipeline = pipeline
        self.path = layout.meta("ingest_manifest", f"{pipeline}.jsonl")
        self._latest: dict[str, dict[str, Any]] = {}
        for rec in read_jsonl(self.path):
            self._latest[rec["unit_key"]] = rec

    def status(self, unit_key: str) -> str | None:
        rec = self._latest.get(unit_key)
        return rec["status"] if rec else None

    def latest(self) -> dict[str, dict[str, Any]]:
        return dict(self._latest)

    def record(
        self,
        unit_key: str,
        status: str,
        n_rows: int = 0,
        started_at: str | None = None,
        error: str | None = None,
        source: str | None = None,
        **extra: Any,
    ) -> None:
        rec = {
            "pipeline": self.pipeline,
            "unit_key": unit_key,
            "source": source,
            "status": status,
            "n_rows": n_rows,
            "started_at": started_at or _utcnow(),
            "finished_at": _utcnow(),
            "code_version": code_version(),
            "error": error,
            **extra,
        }
        append_jsonl(self.path, rec)
        self._latest[unit_key] = rec

    def record_many(self, records: list[dict[str, Any]]) -> None:
        """Écriture groupée (une seule ouverture de fichier) pour les statuts par marché."""
        if not records:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        now = _utcnow()
        version = code_version()
        with open(self.path, "a", encoding="utf-8") as f:
            for r in records:
                rec = {
                    "pipeline": self.pipeline,
                    "source": None,
                    "n_rows": 0,
                    "started_at": now,
                    "finished_at": now,
                    "code_version": version,
                    "error": None,
                    **r,
                }
                f.write(json.dumps(rec, default=str, ensure_ascii=False) + "\n")
                self._latest[rec["unit_key"]] = rec
            f.flush()
            os.fsync(f.fileno())
