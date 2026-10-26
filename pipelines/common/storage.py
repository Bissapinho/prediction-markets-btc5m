"""Écriture des couches raw / clean / features / meta (spec 5.3).

Toutes les écritures de fichiers complets sont atomiques (fichier temporaire + os.replace).
La couche raw n'est jamais réécrite : chaque récupération produit un nouveau fichier horodaté.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import polars as pl
import zstandard as zstd


class Layout:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def raw(self, *parts: str) -> Path:
        return self.root.joinpath("raw", *parts)

    def clean(self, *parts: str) -> Path:
        return self.root.joinpath("clean", *parts)

    def features(self, *parts: str) -> Path:
        return self.root.joinpath("features", *parts)

    def meta(self, *parts: str) -> Path:
        return self.root.joinpath("meta", *parts)


def write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_zst(path: Path, data: bytes, level: int = 10) -> None:
    write_bytes(path, zstd.ZstdCompressor(level=level).compress(data))


def read_zst(path: Path) -> bytes:
    with open(path, "rb") as f:
        # read_across_frames : un fichier peut contenir plusieurs frames zstd concaténées.
        return zstd.ZstdDecompressor().stream_reader(f, read_across_frames=True).read()


def read_json_zst(path: Path) -> Any:
    return json.loads(read_zst(path))


def write_parquet(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.write_parquet(tmp, compression="zstd", statistics=True)
    os.replace(tmp, path)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Ajout d'une ligne JSON, durable (fsync). Pour les journaux à faible débit (manifeste, trous)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # dernière ligne tronquée après un arrêt brutal
    return out
