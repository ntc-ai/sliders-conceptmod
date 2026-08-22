#!/usr/bin/env python3
"""Rewrite `lora_unet--x` / `lora_te--x` double-delimiter keys to single.

Checkpoints trained with `--targets full` before the lora.py naming fix carry a
doubled delimiter, because the root model class is one of TARGET_REPLACE_FULL and
its `named_modules()` name is the empty string. Those keys disagree with the
single-delimiter names `--targets attn` gives the same attention modules, and the
ComfyUI converter matches on `lora_unet-`.

    python scripts/normalize_lora_keys.py models/foo/foo_..._full_last.safetensors

Writes in place after a `.bak` copy, unless --out is given. A file that has no
doubled keys is left untouched.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from safetensors.torch import load_file, save_file


def normalize(path: Path, out: Path | None, delimiter: str = "-", backup: bool = True) -> int:
    tensors = load_file(str(path))
    doubled = delimiter * 2
    renames = {k: k.replace(doubled, delimiter, 1) for k in tensors if doubled in k}
    if not renames:
        print(f"{path.name}: no doubled keys, unchanged")
        return 0
    collisions = {new for new in renames.values() if new in tensors}
    if collisions:
        raise SystemExit(
            f"{path.name}: {len(collisions)} normalized keys already exist "
            f"(e.g. {sorted(collisions)[0]}) — this looks like a double-wrapped "
            "checkpoint; refusing to merge two modules into one key"
        )
    fixed = {renames.get(k, k): v for k, v in tensors.items()}
    target = out or path
    if backup and target == path:
        shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
    save_file(fixed, str(target))
    print(f"{path.name}: renamed {len(renames)} keys -> {target}")
    return len(renames)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("weights", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=None, help="only valid with a single input")
    parser.add_argument("--delimiter", default="-")
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args(argv)
    if args.out and len(args.weights) > 1:
        raise SystemExit("--out takes a single input file")
    total = 0
    for path in args.weights:
        if not path.exists():
            print(f"missing {path}", file=sys.stderr)
            continue
        total += normalize(path, args.out, args.delimiter, backup=not args.no_backup)
    return 0 if total or not args.weights else 0


if __name__ == "__main__":
    raise SystemExit(main())
