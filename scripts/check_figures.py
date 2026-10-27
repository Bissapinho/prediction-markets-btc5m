"""Checks that every number written in README.md and docs/METHODS.md appears in results/key_figures.md.

Numbers are compared as written (sign and thousands separators removed), so a README value that no longer matches
the regenerated results fails the check. Code blocks, inline code, link targets and the "Reproduce" section
(environment, commands, run times) are skipped. Usage: python scripts/check_figures.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

DOCS = [Path("README.md"), Path("docs/METHODS.md")]
KEY = Path("results/key_figures.md")
NUM = re.compile(r"(?<![\w.])\d{1,3}(?:,\d{3})+(?:\.\d+)?|(?<![\w.])\d+(?:\.\d+)?(?:[eE]-?\d+)?")
# Labels, not results: figure / section numbers and hypothesis ids are matched with their prefix and dropped below.
LABELS = re.compile(r"(?:Figure|fig|§|H|A|P|L|S|p|q|t|k|x|τ|σ|θ)\s?\d+|\b\d+\.\s|^#+ \d+(?:\.\d+)*", re.M)


def strip(text: str) -> str:
    text = re.split(r"^## \d*\.?\s*Reproduce", text, flags=re.M)[0]
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"`[^`]*`", " ", text)
    text = re.sub(r"\]\([^)]*\)", "]", text)
    text = re.sub(r"<[^>]+>", " ", text)
    return text


def tokens(text: str) -> set[str]:
    return {t.replace(",", "") for t in NUM.findall(text)}


def main() -> int:
    key = KEY.read_text(encoding="utf-8")
    known = tokens(key)
    bad = []
    for doc in DOCS:
        text = strip(doc.read_text(encoding="utf-8"))
        for i, line in enumerate(text.splitlines(), 1):
            clean = LABELS.sub(" ", line)
            for tok in NUM.findall(clean):
                if tok.replace(",", "") not in known:
                    bad.append(f"{doc}:{i}: {tok}  | {line.strip()[:110]}")
    if bad:
        print(f"{len(bad)} number(s) not found in {KEY}:")
        print("\n".join(bad))
        return 1
    print(f"ok: every number of {', '.join(map(str, DOCS))} appears in {KEY}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
