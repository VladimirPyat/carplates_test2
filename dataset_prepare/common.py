"""Общие утилиты: обход файлов, хэши, CSV, отчёты"""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Соглашение о папках сплитов: root/train, root/val, root/test. Файлы вне этих папок получают split=unknown.
TRAIN_DIR = "train"
VAL_DIR = "val"
TEST_DIR = "test"
SPLITS = (TRAIN_DIR, VAL_DIR, TEST_DIR)
UNKNOWN_SPLIT = "unknown"

def prepare_out(args, stage_file: str):
    """Готовит папки артефактов этапа: out_root/<имя модуля>.

    Возвращает (out_root, out). out_root нужен для skip при сканировании
    (чтобы не обходить артефакты и quarantine всех этапов), out — папка этапа.
    """
    out_root = Path(args.out).resolve()
    out = out_root / Path(stage_file).stem
    out.mkdir(parents=True, exist_ok=True)
    return out_root, out

def base_parser():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--root", required=True, help="корень датасета")
    p.add_argument("--out", default=None, help="папка артефактов")
    p.add_argument("--note", action="append", default=[], help="заметка в отчёт")
    return p


def iter_images(root: Path, skip=()):
    skip = [Path(s).resolve() for s in skip]
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in IMG_EXTS:
            continue
        if any(s == p or s in p.parents for s in skip):
            continue
        yield p


def md5_file(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()



def write_report(out_dir: Path, name: str, summary: dict, deferred=(), notes=()):
    """Отчёт per-команда: {name}.json + {name}.md, чтобы этапы не перезатирали друг друга."""
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"generated_at": datetime.now(timezone.utc).isoformat(),
               "summary": summary, "deferred": list(deferred), "notes": list(notes)}
    (out_dir / f"{name}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md = [f"# {name}", ""] + [f"- {k}: {v}" for k, v in summary.items()]
    if deferred:
        md += ["", "## Осознанно пропущено", ""] + [f"- {d}" for d in deferred]
    if notes:
        md += ["", "## Заметки", ""] + [f"- {n}" for n in notes]
    (out_dir / f"{name}.md").write_text("\n".join(md) + "\n", encoding="utf-8")