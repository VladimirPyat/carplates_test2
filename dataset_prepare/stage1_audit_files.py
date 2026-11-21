
"""Этап 1: аудит файлов — пары метка-файл, стемы, битые файлы, EXIF, размеры, режимы.

 метка ищется рядом с изображением,
либо по уникальному стему внутри того же сплита. Политики обработки — в константах.

Workflow:
    1. python3 dataset_prepare.py files-find --root ./_dataset
       # manifest.csv (все файлы) + flagged.csv (проблемные, action из констант)
    2. python3 dataset_prepare.py images-apply --root ./_dataset           # dry-run
       python dataset_prepare.py images-apply --root ./_dataset --apply
       # плохие изображения -> quarantine, non-RGB -> конвертация (оригинал в backup)
    3. смотрим quarantine, нормальные возвращаем обратно вручную
    4. повторяем files-find (состав и md5 изменились)
    5. python3 dataset_prepare.py labels-apply --root ./_dataset --apply
       # чистка меток-сирот отдельной командой
"""
import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from . import common
from .text_utils import append_csv, read_csv, write_csv

KIND_IMAGE, KIND_LABEL = "image", "label"

# Политики: какие коды флагов во что превращаются
IMAGE_REMOVE_CODES = {"IMAGE_READ", "EMPTY_FILE", "SMALL_IMAGE"}
IMAGE_FIX_CODES = {"MODE_NON_RGB"}
LABEL_REMOVE_CODES = {"ORPHAN_LABEL"}
MANUAL_CODES = {"STEM_DUP", "LABEL_AMBIG", "LABEL_EXT_DUP", "XML_PARSE"}
SEVERITY_ORDER = {"INFO": 0, "WARN": 1, "ERROR": 2}

# Пороги и параметры проверок/исправлений
DEFAULT_MIN_IMAGE_SIDE = 50
EXIF_ORIENTATION_TAG = 274
EXIF_ORIENTATION_NORMAL = 1
TARGET_MODE = "RGB"
WARN_MODES = {"RGBA", "CMYK", "P"}
JPEG_QUALITY = 95

# Имена артефактов — контракт между этапами
MANIFEST_NAME = "manifest.csv"
FLAGGED_NAME = "flagged.csv"
REMOVED_LOG_NAME = "removed_log.csv"
FIXED_LOG_NAME = "fixed_log.csv"
QUARANTINE_DIR = "quarantine"
BACKUP_DIR = "originals"

MANIFEST_FIELDS = ["rel_path", "kind", "split", "size_bytes", "md5", "format", "width", "height",
                   "mode", "exif_orientation", "txt_path", "xml_path", "label_status"]
FLAGGED_FIELDS = ["rel_path", "kind", "split", "md5", "issues", "severity", "action", "reason",
                  "details", "txt_path", "xml_path"]
REMOVED_LOG_FIELDS = ["ts", "rel_path", "quarantine_path", "md5", "kind", "reason"]
FIXED_LOG_FIELDS = ["ts", "rel_path", "backup_path", "md5_before", "md5_after", "kind", "reason"]
DEFERRED = [
    "Геометрия боксов, дубли боксов, min размер в px, маппинг классов — этап 2",
    "Похожие (не точные) дубли — этап 3",
    "Blur / экспозиция / цвет — пропущено осознанно, просмотр в CVAT",
    "EXIF: только флаг, нормализация не выполнялась",
    "RGBA: альфа отбрасывается без композита по допущению , перепроверить позже",
    "Сплит / стратификация / data.yaml — этап 3",
]


def register(sub):
    """Регистрирует команды этапа 1: files-find, images-apply, labels-apply."""
    p = sub.add_parser("files-find", parents=[common.base_parser()], help="аудит файлов: manifest + flagged")
    p.add_argument("--min-side", type=int, default=DEFAULT_MIN_IMAGE_SIDE,
                   help="warn если сторона изображения меньше")
    p.set_defaults(func=cmd_find)

    p = sub.add_parser("images-apply", parents=[common.base_parser()],
                       help="изображения: remove -> quarantine, fix -> конвертация")
    _add_apply_args(p)
    p.set_defaults(func=cmd_images_apply)

    p = sub.add_parser("labels-apply", parents=[common.base_parser()], help="метки: сироты -> quarantine")
    _add_apply_args(p)
    p.set_defaults(func=cmd_labels_apply)


def _add_apply_args(p):
    """Общие аргументы apply-команд: путь к flagged, quarantine, флаг реального исполнения."""
    p.add_argument("--flagged", default=None, help="csv проблемных (по умолчанию out/flagged.csv)")
    p.add_argument("--quarantine", default=None, help="по умолчанию out/quarantine")
    p.add_argument("--apply", action="store_true", help="без флага только dry-run")


def _flag(rel, severity, code, detail=""):
    """Собирает запись о проблеме (чистая функция, ничего не пишет)."""
    return {"rel_path": str(rel), "severity": severity, "code": code, "detail": detail}


def _collect_files(root: Path, skip=()):
    """Собирает изображения и метки под root, пропуская служебные папки."""
    skip = [Path(s).resolve() for s in skip]
    images, labels = [], []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or any(s == p or s in p.parents for s in skip):
            continue
        ext = p.suffix.lower()
        if ext in common.IMG_EXTS:
            images.append(p)
        elif ext in common.LABEL_EXTS:
            labels.append(p)
    return images, labels


def _build_label_index(root: Path, labels):
    """Индексирует метки по (dir, stem) и (split, stem); проверяет XML и дубли расширений.

    XML парсится только на well-formedness, боксы не читаются — это этап 2.
    """
    by_dir, by_split, flags = defaultdict(dict), defaultdict(list), []
    for p in labels:
        rel = p.relative_to(root)
        kind = rel.suffix.lower()[1:]
        key = (rel.parent, rel.stem)
        if kind in by_dir[key]:
            flags.append(_flag(rel, "ERROR", "LABEL_EXT_DUP", f"две .{kind} с одним стемом в папке"))
        by_dir[key][kind] = rel
        by_split[(common.split_of(rel), rel.stem)].append(rel)
        if kind == "xml":
            try:
                ET.parse(p)
            except Exception as e:
                flags.append(_flag(rel, "ERROR", "XML_PARSE", str(e)))
    return by_dir, by_split, flags


def _image_meta(p: Path, rel: Path):
    """md5 + метаданные PIL (format, size, mode, EXIF). Битые файлы -> флаг IMAGE_READ."""
    meta = {"size_bytes": p.stat().st_size, "md5": "", "format": "",
            "width": "", "height": "", "mode": "", "exif_orientation": ""}
    flags = []
    if meta["size_bytes"] == 0:
        flags.append(_flag(rel, "ERROR", "EMPTY_FILE", "0 байт"))
        return meta, flags
    meta["md5"] = common.md5_file(p)
    try:
        with Image.open(p) as im:
            meta["format"], meta["mode"] = im.format, im.mode
            meta["width"], meta["height"] = im.size
            try:
                meta["exif_orientation"] = im.getexif().get(EXIF_ORIENTATION_TAG, EXIF_ORIENTATION_NORMAL)
            except Exception:
                meta["exif_orientation"] = "bad_exif"
            im.load()
        with Image.open(p) as im:
            im.verify()
    except Exception as e:
        flags.append(_flag(rel, "ERROR", "IMAGE_READ", str(e)))
    return meta, flags


def _image_checks(rel: Path, meta: dict, min_side: int):
    """Файловые проверки: размер, EXIF, цветовой режим, имя файла. Только флаги."""
    flags = []
    if meta["width"] and meta["height"]:
        w, h = int(meta["width"]), int(meta["height"])
        if min(w, h) < min_side:
            flags.append(_flag(rel, "WARN", "SMALL_IMAGE", f"{w}x{h}, сторона < {min_side}"))
        if meta["exif_orientation"] not in (EXIF_ORIENTATION_NORMAL, str(EXIF_ORIENTATION_NORMAL), ""):
            flags.append(_flag(rel, "WARN", "EXIF_ROTATED", f"orientation={meta['exif_orientation']}"))
        if meta["mode"] in WARN_MODES:
            flags.append(_flag(rel, "WARN", "MODE_NON_RGB", meta["mode"]))
        elif meta["mode"] != TARGET_MODE:
            flags.append(_flag(rel, "INFO", "MODE_NON_RGB", meta["mode"]))
    name = rel.name
    if any(c.isspace() for c in name) or not name.isascii():
        flags.append(_flag(rel, "WARN", "FILENAME", "пробелы или non-ASCII в имени"))
    return flags


def _match_labels(rel: Path, by_dir: dict, by_split: dict):
    """Ищет txt/xml: сначала рядом, затем по стему в том же сплите.

    Несколько кандидатов -> флаг ambiguous + детерминированный выбор первого по сортировке.
    """
    found = dict(by_dir.get((rel.parent, rel.stem), {}))
    ambiguous = False
    for kind in common.LABEL_KINDS:
        if kind in found:
            continue
        cands = [r for r in by_split.get((common.split_of(rel), rel.stem), [])
                 if r.suffix.lower()[1:] == kind and r.parent != rel.parent]
        if len(cands) == 1:
            found[kind] = cands[0]
        elif len(cands) > 1:
            ambiguous = True
            found[kind] = sorted(cands, key=str)[0]
    return found, ambiguous


def _suggest(kind: str, codes):
    """Производит action из кодов флагов и констант-политик: remove > fix > skip > keep."""
    remove_codes = IMAGE_REMOVE_CODES if kind == KIND_IMAGE else LABEL_REMOVE_CODES
    hard = [c for c in codes if c in remove_codes]
    if hard:
        return common.ACTION_REMOVE, ";".join(hard)
    if kind == KIND_IMAGE:
        fix = [c for c in codes if c in IMAGE_FIX_CODES]
        if fix:
            return common.ACTION_FIX, ";".join(fix)
    manual = [c for c in codes if c in MANUAL_CODES]
    if manual:
        return common.ACTION_SKIP, "требует решения: " + ";".join(manual)
    return common.ACTION_KEEP, "no blocking issues"


def _max_severity(file_flags):
    """Максимальная severity по списку флагов файла."""
    return max((f["severity"] for f in file_flags), key=lambda s: SEVERITY_ORDER[s], default="INFO")


def _build_flagged(manifest_rows, flags):
    """Собирает flagged.csv: только файлы с проблемами, action производный от политик.

    Одна строка на файл: коды, максимальная severity, action, детали, пары для изображений.
    """
    by_path = defaultdict(list)
    for f in flags:
        by_path[f["rel_path"]].append(f)
    rows = []
    for m in manifest_rows:
        ff = by_path.get(m["rel_path"])
        if not ff:
            continue
        codes = [f["code"] for f in ff]
        action, reason = _suggest(m["kind"], codes)
        rows.append({"rel_path": m["rel_path"], "kind": m["kind"], "split": m["split"],
                     "md5": m["md5"], "issues": ";".join(codes), "severity": _max_severity(ff),
                     "action": action, "reason": reason,
                     "details": ";".join(f"{f['code']}:{f['detail']}" for f in ff),
                     "txt_path": m["txt_path"], "xml_path": m["xml_path"]})
    return rows


def cmd_find(args):
    """Аудит файлов: строит manifest.csv (все файлы) и flagged.csv (проблемные). Ничего не меняет.

    Паринг метка-файл, дубли стемов, битые файлы, EXIF, размеры, режимы.
    Сводка — в снапшот files_find.md/json и общий журнал out_root/journal.jsonl.
    """
    root = Path(args.root).resolve()
    out_root, out = common.prepare_out(args, __file__)
    images, labels = _collect_files(root, skip=[out_root])
    by_dir, by_split, flags = _build_label_index(root, labels)

    stem_groups = defaultdict(list)
    for p in images:
        rel = p.relative_to(root)
        stem_groups[(rel.parent, rel.stem)].append(rel)
    for group in stem_groups.values():
        if len(group) > 1:
            for rel in group:
                flags.append(_flag(rel, "ERROR", "STEM_DUP", f"{len(group)} изображений с одним стемом в папке"))

    manifest, claimed, coverage = [], set(), defaultdict(int)
    for p in sorted(images):
        rel = p.relative_to(root)
        meta, mflags = _image_meta(p, rel)
        flags += mflags + _image_checks(rel, meta, args.min_side)
        found, ambiguous = _match_labels(rel, by_dir, by_split)
        if ambiguous:
            flags.append(_flag(rel, "ERROR", "LABEL_AMBIG",
                               "несколько меток с одним стемом в сплите; взят первый по сортировке"))
        row = {"rel_path": str(rel), "kind": KIND_IMAGE, "split": common.split_of(rel), **meta,
               "txt_path": "", "xml_path": "", "label_status": "none"}
        for kind, lrel in found.items():
            row[f"{kind}_path"] = str(lrel)
            claimed.add(lrel)
            if kind == "txt" and (root / lrel).stat().st_size == 0:
                flags.append(_flag(lrel, "INFO", "EMPTY_TXT", "пустой txt = фоновое изображение"))
        status = "both" if len(found) == 2 else (next(iter(found)) + "_only" if found else "none")
        row["label_status"] = status
        coverage[status] += 1
        if not found:
            flags.append(_flag(rel, "INFO", "NO_LABEL", "кандидат в фоновые изображения"))
        manifest.append(row)

    for p in sorted(labels):
        rel = p.relative_to(root)
        manifest.append({"rel_path": str(rel), "kind": KIND_LABEL, "split": common.split_of(rel),
                         "size_bytes": p.stat().st_size, "md5": common.md5_file(p),
                         "format": "", "width": "", "height": "", "mode": "", "exif_orientation": "",
                         "txt_path": "", "xml_path": "", "label_status": ""})
        if rel not in claimed:
            flags.append(_flag(rel, "ERROR", "ORPHAN_LABEL", "нет изображения-пары"))

    flagged = _build_flagged(manifest, flags)
    write_csv(out / MANIFEST_NAME, manifest, MANIFEST_FIELDS)
    write_csv(out / FLAGGED_NAME, flagged, FLAGGED_FIELDS)

    by_sev = defaultdict(int)
    for f in flags:
        by_sev[f["severity"]] += 1
    summary = {"images": len(images), "labels": len(labels), "coverage": dict(coverage),
               "orphans": sum(1 for f in flags if f["code"] == "ORPHAN_LABEL"),
               "ERROR": by_sev["ERROR"], "WARN": by_sev["WARN"], "INFO": by_sev["INFO"],
               "flagged_files": len(flagged),
               "image_remove": sum(1 for r in flagged if r["kind"] == KIND_IMAGE and r["action"] == common.ACTION_REMOVE),
               "image_fix": sum(1 for r in flagged if r["kind"] == KIND_IMAGE and r["action"] == common.ACTION_FIX),
               "label_remove": sum(1 for r in flagged if r["kind"] == KIND_LABEL and r["action"] == common.ACTION_REMOVE)}
    common.write_report(out, "files_find", summary, DEFERRED, args.note)
    common.append_journal(out_root, "files-find", summary, args)
    print(f"images={len(images)}  labels={len(labels)}  coverage={dict(coverage)}")
    print(f"ERROR={by_sev['ERROR']}  WARN={by_sev['WARN']}  INFO={by_sev['INFO']}  flagged={len(flagged)}")
    print(f"image_remove={summary['image_remove']}  image_fix={summary['image_fix']}  label_remove={summary['label_remove']}")
    print(f"manifest: {out / MANIFEST_NAME}  flagged: {out / FLAGGED_NAME}")


def _plan_actions(root: Path, rows, quarantine: Path):
    """Готовит перемещения и конвертации по flagged с проверкой существования и md5.

    Возвращает (moves, fixes, skipped). md5 не совпал -> файл менялся после find, пропуск.
    """
    moves, fixes, skipped = [], [], []
    for r in rows:
        src = root / r["rel_path"]
        if not src.is_file():
            skipped.append((r["rel_path"], "missing"))
            continue
        if r["md5"] and common.md5_file(src) != r["md5"]:
            skipped.append((r["rel_path"], "md5 changed since find"))
            continue
        if r["action"] == common.ACTION_REMOVE:
            moves.append((src, quarantine / r["rel_path"], r))
        elif r["action"] == common.ACTION_FIX:
            fixes.append((src, r))
    return moves, fixes, skipped


def _convert_to_rgb(src: Path, rel: str, backup_dir: Path):
    """Конвертирует изображение в TARGET_MODE in-place, оригинал копирует в backup.

    EXIF сохраняется как есть (ориентация молча не меняется). JPEG — с JPEG_QUALITY.
    Альфа отбрасывается без композита (допущение: мусора под ней нет).
    Возвращает (backup_path, md5_before, md5_after) или None при неизвестном формате.
    """
    md5_before = common.md5_file(src)
    with Image.open(src) as im:
        fmt, exif = im.format, im.info.get("exif")
        rgb = im.convert(TARGET_MODE)
    if not fmt:
        return None
    backup = backup_dir / rel
    backup.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, backup)
    kwargs = {"exif": exif} if exif else {}
    if fmt == "JPEG":
        kwargs["quality"] = JPEG_QUALITY
    rgb.save(src, format=fmt, **kwargs)
    return str(backup), md5_before, common.md5_file(src)


def _run_apply(args, kind: str, command: str):
    """Общий исполнитель apply для изображений/меток: dry-run по умолчанию, логи append.

    remove -> quarantine (только сами файлы, парные метки не трогаем — их чистит labels-apply),
    fix -> конвертация с бэкапом (только изображения). Пишет removed_log/fixed_log, снапшот, журнал.
    """
    root = Path(args.root).resolve()
    out_root, out = common.prepare_out(args, __file__)
    flagged_path = Path(args.flagged) if args.flagged else out / FLAGGED_NAME
    quarantine = Path(args.quarantine) if args.quarantine else out / QUARANTINE_DIR
    allowed = {common.ACTION_REMOVE, common.ACTION_FIX} if kind == KIND_IMAGE else {common.ACTION_REMOVE}
    rows = [r for r in read_csv(flagged_path) if r["kind"] == kind and r["action"] in allowed]
    moves, fixes, skipped = _plan_actions(root, rows, quarantine)

    if not args.apply:
        summary = {"dry_run": True, "to_move": len(moves), "to_fix": len(fixes), "skipped": len(skipped)}
        common.append_journal(out_root, command, summary, args)
        print(f"DRY-RUN [{command}]: move={len(moves)}  fix={len(fixes)}  skipped={len(skipped)}. Запусти с --apply")
        return

    removed_log, fixed_log = [], []
    ts = datetime.now(timezone.utc).isoformat()
    for src, dst, r in moves:
        if dst.exists():
            skipped.append((r["rel_path"], "quarantine path exists"))
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        removed_log.append({"ts": ts, "rel_path": r["rel_path"], "quarantine_path": str(dst),
                            "md5": r["md5"], "kind": kind, "reason": r["reason"]})
    for src, r in fixes:
        result = _convert_to_rgb(src, r["rel_path"], out / BACKUP_DIR)
        if result is None:
            skipped.append((r["rel_path"], "unknown format, skip fix"))
            continue
        backup, md5_before, md5_after = result
        fixed_log.append({"ts": ts, "rel_path": r["rel_path"], "backup_path": backup,
                          "md5_before": md5_before, "md5_after": md5_after, "kind": kind, "reason": r["reason"]})

    append_csv(out / REMOVED_LOG_NAME, removed_log, REMOVED_LOG_FIELDS)
    append_csv(out / FIXED_LOG_NAME, fixed_log, FIXED_LOG_FIELDS)
    summary = {"moved": len(removed_log), "fixed": len(fixed_log), "skipped": len(skipped),
               "quarantine": str(quarantine)}
    common.write_report(out, command.replace("-", "_"), summary, DEFERRED, args.note)
    common.append_journal(out_root, command, summary, args)
    print(f"[{command}] moved={len(removed_log)}  fixed={len(fixed_log)}  skipped={len(skipped)}")


def cmd_images_apply(args):
    """Исполняет flagged для изображений: remove -> quarantine, fix -> конвертация в RGB."""
    _run_apply(args, KIND_IMAGE, "images-apply")


def cmd_labels_apply(args):
    """Исполняет flagged для меток: сироты -> quarantine. Парные метки не трогаются."""
    _run_apply(args, KIND_LABEL, "labels-apply")