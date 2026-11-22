"""Этап 2: боксы + справочник классов + статистика. Read-only, датасет не меняет.
Вход: manifest.csv этапа 1 (kind=image, размеры, парные txt/xml).

Запуск:
    1. python3 dataset_prepare.py boxes-stats --root ./_dataset
       # classes.json (справочник) + stats.csv (итоговая стата по классам)
    2. python3 dataset_prepare.py boxes-check --root ./_dataset
       # flags_boxes.csv — очередь правок для CVAT, с номерами кадров

После правок в CVAT: перезапускаем files-find, затем обе команды снова.

Принудительно вывести заново — boxes-stats --remap (только если XML свежий).

Канонический формат разметки — YOLO txt; XML — источник имён и кросс-проверка.
"""
import json
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

from . import common, stage1_audit_files as stage1
from .text_utils import read_csv, write_csv

# --- Пороги проверок боксов (ключевые переопределяются CLI) ---
COORD_EPS = 1e-6                  # допуск float-округления: raw YOLO cx,cy,w,h в [0,1]±eps
SIDE_EPS = 1e-6                   # допуск сравнения стороны с порогом (4.9999px не считаем < 5)
DEFAULT_MIN_BOX_SIDE_PX = 5       # CLI --min-box-side: min(сторона) в px, меньше -> SMALL_BOX
DEFAULT_BOUNDS_TOL_PX = 2         # CLI --bounds-tol: допуск выхода бокса за кадр, px
MAX_ASPECT_RATIO = 20.0           # max/min сторона; у символов норма ~0.2-2
DEFAULT_DUP_IOU = 0.8             # CLI --dup-iou: IoU пары одного класса -> DUP_BOX
DEFAULT_MATCH_IOU = 0.5           # CLI --match-iou: IoU матчинга YOLO<->VOC (мягкий)
CROSS_CHECK_IOU = 0.9             # сматченные пары ниже -> GEOMETRY_DRIFT
DEFAULT_MAPPING_PURITY = 0.9      # CLI --purity: доля голосов за лучший id
MAX_OBJECTS_WARN = 300            # объектов на кадр (ориентир max_det на инференсе)
SMALL_AREA_PX = 32 ** 2           # COCO small: площадь < 32²
MEDIUM_AREA_PX = 96 ** 2          # COCO medium: 32²..96², large: > 96²
VERIFY_MANIFEST_MD5 = True        # сверять md5 с манифестом (ловим устаревание)
CROSS_CHECK_PER_BOX = False       # True -> писать drift по боксам (отладка), иначе агрегат

# --- Нумерация кадров: должна совпадать с порядком в задании CVAT ---
FRAME_SORT_KEY = "name"          # "rel_path" | "name" — ключ сортировки списка изображений
FRAME_NUMBER_SCOPE = "split"     # "global" (один task на всё) | "split" (task на каждый сплит)

# --- Политики вывода ---
SPLIT_ALL = "all"
FORMAT_YOLO, FORMAT_VOC, FORMAT_BOTH = "yolo", "voc", "both"

# --- Имена артефактов (контракт между этапами) ---
COMBINATIONS_NAME = "combinations.csv"
COMBINATIONS_FIELDS = ["signature", "count_train", "count_test", "count_total", "splits"]
CLASSES_NAME = "classes.json"
STATS_NAME = "stats.csv"
FLAGS_NAME = "flags_boxes.csv"

FLAGS_FIELDS = ["rel_path", "frame_index", "frame_no", "format", "box_index",
                "severity", "code", "detail"]
STATS_FIELDS = ["class_id", "name", "split", "instances", "images", "share",
                "small", "medium", "large", "side_px_min", "side_px_p50", "rel_p50"]
DEFERRED = [
    "Исправление боксов/классов — вручную в CVAT, скрипт только флагует",
    "GEOMETRY_DRIFT/FORMAT_UNMATCHED — системный сдвиг ~1px между форматами, только агрегат в отчёте",
    "obj.names — генерируется на этапе упаковки под CVAT из classes.json",
    "Мелкие боксы не удаляем: удаление бокса при живом объекте = неразмеченный позитив",

]


def register(sub):
    """Регистрирует команды этапа 2: boxes-stats и boxes-check."""
    p = sub.add_parser("boxes-stats", parents=[common.base_parser()],
                       help="справочник классов + итоговая статистика по классам")
    p.add_argument("--remap", action="store_true", help="принудительно переголосовать маппинг из XML")
    p.add_argument("--match-iou", type=float, default=DEFAULT_MATCH_IOU, help="IoU матчинга при голосовании")
    p.add_argument("--purity", type=float, default=DEFAULT_MAPPING_PURITY, help="мин. доля голосов за id")
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("boxes-check", parents=[common.base_parser()],
                       help="проверка боксов -> очередь правок с номерами кадров")
    p.add_argument("--min-box-side", type=int, default=DEFAULT_MIN_BOX_SIDE_PX, help="warn если сторона меньше, px")
    p.add_argument("--dup-iou", type=float, default=DEFAULT_DUP_IOU, help="IoU дубля бокса одного класса")
    p.add_argument("--bounds-tol", type=float, default=DEFAULT_BOUNDS_TOL_PX, help="допуск выхода за границы, px")
    p.add_argument("--match-iou", type=float, default=DEFAULT_MATCH_IOU, help="IoU кросс-матчинга форматов")
    p.set_defaults(func=cmd_check)


def _flag(rel, severity, code, detail="", fmt="", idx=""):
    """Собирает запись о проблеме бокса/файла (номера кадров добавляются позже)."""
    return {"rel_path": str(rel), "frame_index": "", "frame_no": "", "format": fmt,
            "box_index": idx, "severity": severity, "code": code, "detail": detail}


def _load_manifest(out_root: Path):
    """Читает манифест этапа 1 (kind=image) по контракту имя-модуля/константа."""
    path = out_root / Path(stage1.__file__).stem / stage1.MANIFEST_NAME
    if not path.is_file():
        raise SystemExit(f"нет манифеста этапа 1: {path}. Сначала запусти files-find")
    return [r for r in read_csv(path) if r["kind"] == stage1.KIND_IMAGE]


def _verify_manifest(root: Path, rows):
    """Проверяет свежесть манифеста: файл на месте, md5 не изменился.

    Missing -> строка исключается; stale md5 -> WARN, строка остаётся.
    """
    ok, flags = [], []
    for r in rows:
        p = root / r["rel_path"]
        if not p.is_file():
            flags.append(_flag(r["rel_path"], "ERROR", "MANIFEST_MISSING", "файл исчез после files-find"))
            continue
        if VERIFY_MANIFEST_MD5 and r["md5"] and common.md5_file(p) != r["md5"]:
            flags.append(_flag(r["rel_path"], "WARN", "MANIFEST_STALE", "md5 изменился после files-find"))
        ok.append(r)
    return ok, flags


def _signature(boxes: list, names: list) -> str:
    """Подпись кадра = последовательность имён классов его боксов, отсортированная по центру X.

    Один и тот же номерной знак даёт одинаковую подпись независимо от порядка
    разметки в txt. Используется и здесь (частоты), и на этапе сплита (дедуп).
    Кадр без боксов возвращает пустую строку (фоновые не группируем).
    """
    if not boxes:
        return ""
    ordered = sorted(boxes, key=lambda b: b["cx"])
    return "".join(names[b["cls"]] if b["cls"] < len(names) else f"?{b['cls']}" for b in ordered)


def _combinations(anns, names):
    """Сквозные частоты подписей номеров: одна строка на уникальную подпись.

    Подпись = последовательность имён классов боксов, отсортированная по центру X.
    count_train / count_test / count_total дают сквозную картину:
    подпись в обоих сплитах = один номер снят и в train, и в test (утечка),
    подпись только в test = потенциально уникальный для test номер.
    Колонка splits — где встречается (train / test / train;test).
    Возвращает (rows, summary): rows для combinations.csv, summary для отчёта.
    """
    per_split = defaultdict(Counter)
    for e in anns.values():
        sig = _signature(e["yolo"], names)
        if sig:
            per_split[e["split"]][sig] += 1

    sigs = sorted({sig for c in per_split.values() for sig in c})
    rows, overlap_top = [], []
    n_both = test_total = test_shared = 0
    for sig in sigs:
        tr = per_split[common.TRAIN_DIR][sig]
        te = per_split[common.TEST_DIR][sig]
        total = sum(c[sig] for c in per_split.values())
        rows.append({"signature": sig, "count_train": tr, "count_test": te,
                     "count_total": total,
                     "splits": ";".join(s for s in sorted(per_split) if per_split[s][sig])})
        if tr and te:
            n_both += 1
            test_shared += te
            overlap_top.append((sig, tr, te))
        test_total += te
    overlap_top.sort(key=lambda t: t[2], reverse=True)

    summary = {"sigs_unique": len(sigs),
               "sigs_in_both": n_both,
               "sigs_train_only": sum(1 for r in rows if r["count_train"] and not r["count_test"]),
               "sigs_test_only": sum(1 for r in rows if r["count_test"] and not r["count_train"]),
               "test_images_total": test_total,
               "test_images_shared_sig": test_shared,
               "test_images_unique_sig": test_total - test_shared,
               "top_overlap": [f"{s} train={tr} test={te}" for s, tr, te in overlap_top[:5]]}
    return rows, summary


def _frame_map(rows):
    """Считает порядковые номера кадров в полном списке изображений датасета.

    Порядок задаётся FRAME_SORT_KEY и должен совпадать с порядком в задании CVAT.
    FRAME_NUMBER_SCOPE='split' нумерует внутри каждого сплита отдельно.
    Возвращает {rel_path: index (0-based)}.
    """
    key = (lambda r: r["rel_path"]) if FRAME_SORT_KEY == "rel_path" else (lambda r: Path(r["rel_path"]).name)
    buckets = defaultdict(list)
    for r in rows:
        buckets[r["split"] if FRAME_NUMBER_SCOPE == "split" else SPLIT_ALL].append(r)
    mapping = {}
    for group in buckets.values():
        for i, r in enumerate(sorted(group, key=key)):
            mapping[r["rel_path"]] = i
    return mapping


def _parse_yolo(path: Path, w: int, h: int):
    """Парсит YOLO txt в боксы с px-координатами. Структурные ошибки -> флаги."""
    boxes, flags = [], []
    for li, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 5:
            flags.append(_flag(path, "ERROR", "YOLO_FORMAT", f"строка {li}: {len(parts)} полей", FORMAT_YOLO, li))
            continue
        try:
            cls = int(float(parts[0]))
            cx, cy, bw, bh = map(float, parts[1:])
        except ValueError:
            flags.append(_flag(path, "ERROR", "YOLO_FORMAT", f"строка {li}: не числа", FORMAT_YOLO, li))
            continue
        boxes.append({"cls": cls, "name": None, "cx": cx, "cy": cy, "bw": bw, "bh": bh,
                      "x1": (cx - bw / 2) * w, "y1": (cy - bh / 2) * h,
                      "x2": (cx + bw / 2) * w, "y2": (cy + bh / 2) * h})
    return boxes, flags


def _parse_voc(path: Path, w: int, h: int):
    """Парсит Pascal VOC xml в боксы (px). difficult/truncated сохраняем для отчёта."""
    boxes, flags = [], []
    try:
        root = ET.parse(path).getroot()
    except Exception as e:
        return [], [_flag(path, "ERROR", "XML_PARSE", str(e), FORMAT_VOC)]
    for oi, obj in enumerate(root.findall("object")):
        name = (obj.findtext("name") or "").strip()
        bb = obj.find("bndbox")
        if bb is None or not name:
            flags.append(_flag(path, "ERROR", "VOC_FORMAT", f"object {oi}: нет name/bndbox", FORMAT_VOC, oi))
            continue
        try:
            x1, y1 = float(bb.findtext("xmin")), float(bb.findtext("ymin"))
            x2, y2 = float(bb.findtext("xmax")), float(bb.findtext("ymax"))
        except (TypeError, ValueError):
            flags.append(_flag(path, "ERROR", "VOC_FORMAT", f"object {oi}: координаты не числа", FORMAT_VOC, oi))
            continue
        boxes.append({"cls": None, "name": name, "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                      "difficult": int(obj.findtext("difficult") or 0),
                      "truncated": int(obj.findtext("truncated") or 0)})
    return boxes, flags


def _load_annotations(root: Path, rows):
    """Общий парсер этапа: боксы обоих форматов на изображение, один проход на команду."""
    anns, flags = {}, []
    for r in rows:
        rel = r["rel_path"]
        w, h = int(r["width"] or 0), int(r["height"] or 0)
        entry = {"w": w, "h": h, "split": r["split"], "yolo": [], "voc": [],
                 "has_txt": bool(r["txt_path"]), "has_xml": bool(r["xml_path"])}
        if not w or not h:
            flags.append(_flag(rel, "ERROR", "NO_IMAGE_SIZE", "нет размеров в манифесте"))
        else:
            for key, col, parser in (("yolo", "txt_path", _parse_yolo), ("voc", "xml_path", _parse_voc)):
                if r[col]:
                    lp = root / r[col]
                    if not lp.is_file():
                        flags.append(_flag(rel, "ERROR", "LABEL_MISSING", f"нет {r[col]}"))
                        continue
                    boxes, bf = parser(lp, w, h)
                    entry[key] = boxes
                    flags += bf
        anns[rel] = entry
    return anns, flags


def _iou(a: dict, b: dict) -> float:
    """IoU двух боксов в px-координатах."""
    ix1, iy1 = max(a["x1"], b["x1"]), max(a["y1"], b["y1"])
    ix2, iy2 = min(a["x2"], b["x2"]), min(a["y2"], b["y2"])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    aa = max(0.0, a["x2"] - a["x1"]) * max(0.0, a["y2"] - a["y1"])
    ab = max(0.0, b["x2"] - b["x1"]) * max(0.0, b["y2"] - b["y1"])
    union = aa + ab - inter
    return inter / union if union > 0 else 0.0


def _match_boxes(yolo, voc, thresh: float):
    """Жадный матчинг YOLO<->VOC по IoU. Возвращает (pairs, unmatched_yolo_idx, unmatched_voc_idx)."""
    cands = sorted(((_iou(y, v), i, j) for i, y in enumerate(yolo)
                    for j, v in enumerate(voc) if _iou(y, v) >= thresh), reverse=True)
    used_y, used_v, pairs = set(), set(), []
    for iou, i, j in cands:
        if i in used_y or j in used_v:
            continue
        used_y.add(i)
        used_v.add(j)
        pairs.append((i, j, iou))
    uy = [i for i in range(len(yolo)) if i not in used_y]
    uv = [j for j in range(len(voc)) if j not in used_v]
    return pairs, uy, uv


def _geometry_flags(rel, fmt, idx, box, w, h, min_side, bounds_tol):
    """Геометрия одного бокса: вырожденность, границы, мелкий размер, aspect ratio."""
    flags = []
    bw, bh = box["x2"] - box["x1"], box["y2"] - box["y1"]
    if bw <= 0 or bh <= 0:
        flags.append(_flag(rel, "ERROR", "DEGENERATE_BOX", f"стороны {bw:.1f}x{bh:.1f}", fmt, idx))
        return flags
    if (box["x1"] < -bounds_tol or box["y1"] < -bounds_tol
            or box["x2"] > w + bounds_tol or box["y2"] > h + bounds_tol):
        flags.append(_flag(rel, "WARN", "BOX_BOUNDS",
                           f"xyxy=({box['x1']:.1f},{box['y1']:.1f},{box['x2']:.1f},{box['y2']:.1f}) "
                           f"при кадре {w}x{h}", fmt, idx))
    side = min(bw, bh)
    if side < min_side - SIDE_EPS:
        flags.append(_flag(rel, "WARN", "SMALL_BOX", f"сторона {side:.2f}px < {min_side}", fmt, idx))
    if max(bw, bh) / side > MAX_ASPECT_RATIO:
        flags.append(_flag(rel, "WARN", "EXTREME_AR", f"AR={max(bw, bh) / side:.1f}", fmt, idx))
    return flags


def _check_boxes(anns, nc, min_side, dup_iou, bounds_tol):
    """Проверки боксов обоих форматов: диапазоны YOLO, классы, геометрия, дубли одного класса."""
    flags = []
    for rel, e in anns.items():
        w, h = e["w"], e["h"]
        for fmt in ("yolo", "voc"):
            boxes = e[fmt]
            if len(boxes) > MAX_OBJECTS_WARN:
                flags.append(_flag(rel, "INFO", "MANY_OBJECTS", f"{len(boxes)} объектов", fmt))
            for idx, b in enumerate(boxes):
                flags += _geometry_flags(rel, fmt, idx, b, w, h, min_side, bounds_tol)
                if fmt == "yolo":
                    if any(not (-COORD_EPS <= b[k] <= 1 + COORD_EPS) for k in ("cx", "cy", "bw", "bh")):
                        flags.append(_flag(rel, "ERROR", "YOLO_RANGE",
                                           f"raw=({b['cx']},{b['cy']},{b['bw']},{b['bh']})", fmt, idx))
                    if b["bw"] <= 0 or b["bh"] <= 0:
                        flags.append(_flag(rel, "ERROR", "YOLO_RANGE", "w/h <= 0", fmt, idx))
                    if nc is not None and not (0 <= b["cls"] < nc):
                        flags.append(_flag(rel, "ERROR", "CLASS_RANGE", f"id={b['cls']}, nc={nc}", fmt, idx))
            key = "cls" if fmt == "yolo" else "name"
            for i, j in combinations(range(len(boxes)), 2):
                if boxes[i][key] != boxes[j][key]:
                    continue
                iou = _iou(boxes[i], boxes[j])
                if iou > dup_iou:
                    flags.append(_flag(rel, "WARN", "DUP_BOX", f"боксы {i}/{j}, IoU={iou:.2f}", fmt, i))
    return flags


def _check_cross(anns, match_iou):
    """Кросс-проверка форматов: агрегаты по сдвигу геометрии + per-box флаги для отладки.

    Возвращает (flags, summary). Флаги пишутся только при CROSS_CHECK_PER_BOX,
    сводка попадает в отчёт всегда.
    """
    flags, ious, unmatched_sides = [], [], []
    images_both = images_unmatched = drift_pairs = unmatched_y = unmatched_v = 0
    for rel, e in anns.items():
        if not (e["has_txt"] and e["has_xml"]):
            continue
        images_both += 1
        pairs, uy, uv = _match_boxes(e["yolo"], e["voc"], match_iou)
        if uy or uv:
            images_unmatched += 1
            unmatched_y += len(uy)
            unmatched_v += len(uv)
            for i in uy:
                b = e["yolo"][i]
                unmatched_sides.append(min(b["x2"] - b["x1"], b["y2"] - b["y1"]))
            flags.append(_flag(rel, "WARN", "FORMAT_UNMATCHED", f"yolo={len(uy)}, voc={len(uv)}", FORMAT_BOTH))
        for i, j, iou in pairs:
            ious.append(iou)
            if iou < CROSS_CHECK_IOU:
                drift_pairs += 1
                flags.append(_flag(rel, "WARN", "GEOMETRY_DRIFT",
                                   f"yolo[{i}] vs voc[{j}], IoU={iou:.2f}", FORMAT_BOTH, i))
    summary = {"cross_images_both": images_both, "cross_images_unmatched": images_unmatched,
               "cross_drift_pairs": drift_pairs, "cross_matched_pairs": len(ious),
               "cross_iou_p50": round(_pct(ious, 50), 3), "cross_iou_min": round(min(ious), 3) if ious else None,
               "cross_unmatched_yolo": unmatched_y, "cross_unmatched_voc": unmatched_v,
               "cross_unmatched_side_p50": round(_pct(unmatched_sides, 50), 1)}
    return flags, summary


def _merge_format_duplicates(flags):
    """Схлопывает одинаковые флаги из yolo и voc в одну строку (format='yolo+voc').

    Ключ: (файл, индекс бокса, код). Индексы боксов в форматах обычно совпадают.
    """
    merged, order = {}, []
    for f in flags:
        key = (f["rel_path"], str(f["box_index"]), f["code"])
        if key not in merged:
            merged[key] = {**f, "_formats": [f["format"]] if f["format"] else []}
            order.append(key)
        elif f["format"] and f["format"] not in merged[key]["_formats"]:
            merged[key]["_formats"].append(f["format"])
    rows = []
    for k in order:
        row = merged[k]
        row["format"] = "+".join(row.pop("_formats"))
        rows.append(row)
    return rows


def _attach_frames(flags, frames):
    """Проставляет frame_index (0-based) и frame_no (1-based) по полному списку изображений."""
    for f in flags:
        idx = frames.get(f["rel_path"])
        if idx is not None:
            f["frame_index"], f["frame_no"] = idx, idx + 1
    return flags


def _load_classes(out: Path):
    """Читает names из classes.json, если он есть (иначе None)."""
    path = out / CLASSES_NAME
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8")).get("names")
    return None


def _resolve_classes(votes, all_ids, purity):
    """Собирает id->name из голосов: majority + проверка биекции. Возвращает (names, conflicts)."""
    by_id, conflicts = {}, []
    for name in sorted(votes):
        counter = votes[name]
        best_id, best_n = counter.most_common(1)[0]
        total = sum(counter.values())
        if best_n / total < purity:
            conflicts.append({"name": name, "votes": dict(counter), "reason": f"purity {best_n / total:.2f} < {purity}"})
        if best_id in by_id and by_id[best_id] != name:
            conflicts.append({"id": best_id, "names": [by_id[best_id], name], "reason": "id collision"})
        else:
            by_id[best_id] = name
    nc = max(set(all_ids) | set(by_id), default=-1) + 1
    names = [by_id.get(i, f"class_{i}") for i in range(nc)]
    for i in range(nc):
        if i not in by_id:
            conflicts.append({"id": i, "reason": "нет голосов из XML"})
    return names, conflicts


def _ensure_classes(anns, out: Path, remap: bool, match_iou: float, purity: float):
    """Гарантирует справочник классов: переиспользует classes.json или выводит голосованием.

    По умолчанию переиспользует — после правок в CVAT XML устаревает.
    Возвращает (names, info) с признаком source=reused|voted.
    """
    path = out / CLASSES_NAME
    if path.is_file() and not remap:
        data = json.loads(path.read_text(encoding="utf-8"))
        names = data.get("names", [])
        return names, {"classes_source": "reused", "nc": len(names),
                       "classes_conflicts": len(data.get("conflicts", []))}
    votes, all_ids = defaultdict(Counter), set()
    for e in anns.values():
        all_ids.update(b["cls"] for b in e["yolo"])
        if e["has_txt"] and e["has_xml"]:
            pairs, _, _ = _match_boxes(e["yolo"], e["voc"], match_iou)
            for i, j, _ in pairs:
                votes[e["voc"][j]["name"]][e["yolo"][i]["cls"]] += 1
    names, conflicts = _resolve_classes(votes, all_ids, purity)
    path.write_text(json.dumps({"nc": len(names), "names": names, "conflicts": conflicts},
                               ensure_ascii=False, indent=2), encoding="utf-8")
    return names, {"classes_source": "voted", "nc": len(names), "classes_conflicts": len(conflicts)}


def _pct(values, q):
    """Перцентиль списка (numpy), 0 для пустого."""
    return float(np.percentile(values, q)) if len(values) else 0.0


def cmd_stats(args):
    """Справочник классов + итоговая статистика + частоты комбинаций за один парсинг.

    Артефакты:
        classes.json      — справочник (nc, names, conflicts), по умолчанию переиспользуется
        stats.csv         — стата по классам x сплитам (включая all)
        combinations.csv  — частоты подписей номеров (последовательность классов) по сплитам
    Покрытие форматов, источник справочника и сводка комбинаций — в отчёте.
    """
    root = Path(args.root).resolve()
    out_root, out = common.prepare_out(args, __file__)
    rows, flags = _verify_manifest(root, _load_manifest(out_root))
    anns, pflags = _load_annotations(root, rows)
    names, cinfo = _ensure_classes(anns, out, args.remap, args.match_iou, args.purity)

    acc = defaultdict(lambda: {"inst": 0, "imgs": set(), "sides": [], "rels": [],
                               "small": 0, "medium": 0, "large": 0})
    per_image, ars = [], []
    difficult = truncated = 0
    for rel, e in anns.items():
        per_image.append(len(e["yolo"]))
        for b in e["yolo"]:
            bw, bh = b["x2"] - b["x1"], b["y2"] - b["y1"]
            area, side = max(0.0, bw * bh), min(bw, bh)
            rel_size = (area ** 0.5) / ((e["w"] * e["h"]) ** 0.5) if e["w"] and e["h"] else 0.0
            if side > 0:
                ars.append(max(bw, bh) / side)
            bucket = "small" if area < SMALL_AREA_PX else "medium" if area < MEDIUM_AREA_PX else "large"
            for split in (e["split"], SPLIT_ALL):
                a = acc[(b["cls"], split)]
                a["inst"] += 1
                a["imgs"].add(rel)
                a["sides"].append(side)
                a["rels"].append(rel_size)
                a[bucket] += 1
        for b in e["voc"]:
            difficult += b.get("difficult", 0)
            truncated += b.get("truncated", 0)

    total = sum(a["inst"] for (c, s), a in acc.items() if s == SPLIT_ALL)
    stats = []
    for (cls, split), a in sorted(acc.items()):
        stats.append({"class_id": cls, "name": names[cls] if cls < len(names) else f"class_{cls}",
                      "split": split, "instances": a["inst"], "images": len(a["imgs"]),
                      "share": round(a["inst"] / total, 4) if total else 0,
                      "small": a["small"], "medium": a["medium"], "large": a["large"],
                      "side_px_min": round(min(a["sides"]), 1) if a["sides"] else 0,
                      "side_px_p50": round(_pct(a["sides"], 50), 1),
                      "rel_p50": round(_pct(a["rels"], 50), 4)})
    write_csv(out / STATS_NAME, stats, STATS_FIELDS)

    comb_rows, comb_summary = _combinations(anns, names)
    write_csv(out / COMBINATIONS_NAME, comb_rows, COMBINATIONS_FIELDS)

    cov = Counter(("both" if e["has_txt"] and e["has_xml"] else
                   "txt_only" if e["has_txt"] else "xml_only" if e["has_xml"] else "none")
                  for e in anns.values())
    inst_all = [a["inst"] for (c, s), a in acc.items() if s == SPLIT_ALL]
    all_sides = [s for (c, sp), a in acc.items() if sp == SPLIT_ALL for s in a["sides"]]
    summary = {"images": len(anns), "instances": total, **cinfo, "coverage": dict(cov),
               "objects_per_image_min": min(per_image, default=0),
               "objects_per_image_p50": _pct(per_image, 50),
               "objects_per_image_max": max(per_image, default=0),
               "zero_object_images": sum(1 for n in per_image if n == 0),
               "imbalance_ratio": round(max(inst_all) / min(inst_all), 2) if inst_all and min(inst_all) else None,
               "side_px_p50": round(_pct(all_sides, 50), 1), "ar_p95": round(_pct(ars, 95), 2),
               "difficult": difficult, "truncated": truncated,
               **comb_summary,
               "parse_flags": len(flags) + len(pflags)}
    common.write_report(out, "boxes_stats", summary, DEFERRED, args.note)
    common.append_journal(out_root, "boxes-stats", summary, args)

    print(f"images={len(anns)}  instances={total}  nc={cinfo['nc']}  classes={cinfo['classes_source']}")
    print(f"coverage={dict(cov)}  imbalance={summary['imbalance_ratio']}")

    # Глобальные метрики по подписям (сквозная статистика)
    print(f"sigs: unique={comb_summary['sigs_unique']}  in_both={comb_summary['sigs_in_both']}  "
          f"train_only={comb_summary['sigs_train_only']}  test_only={comb_summary['sigs_test_only']}")
    print(f"test images: total={comb_summary['test_images_total']}  "
          f"shared_sig={comb_summary['test_images_shared_sig']}  "
          f"unique_sig={comb_summary['test_images_unique_sig']}")
    if comb_summary.get('top_overlap'):
        print(f"top overlap (train & test): {', '.join(comb_summary['top_overlap'][:3])}")

    print(f"classes: {out / CLASSES_NAME}  stats: {out / STATS_NAME}  combinations: {out / COMBINATIONS_NAME}")


def cmd_check(args):
    """Проверка боксов -> flags_boxes.csv: очередь правок с номерами кадров.

    Кросс-форматные расхождения (drift/unmatched) уходят агрегатом в отчёт,
    одинаковые флаги yolo/voc схлопываются, строки отсортированы по кадру.
    """
    root = Path(args.root).resolve()
    out_root, out = common.prepare_out(args, __file__)
    rows, flags = _verify_manifest(root, _load_manifest(out_root))
    anns, pflags = _load_annotations(root, rows)
    flags += pflags
    names = _load_classes(out)
    nc = len(names) if names else (max((b["cls"] for e in anns.values() for b in e["yolo"]), default=-1) + 1 or None)
    flags += _check_boxes(anns, nc, args.min_box_side, args.dup_iou, args.bounds_tol)
    cross_flags, cross_summary = _check_cross(anns, args.match_iou)
    if CROSS_CHECK_PER_BOX:
        flags += cross_flags
    flags = _attach_frames(_merge_format_duplicates(flags), _frame_map(rows))
    flags.sort(key=lambda f: (f["frame_index"] if f["frame_index"] != "" else 10 ** 9,
                              int(f["box_index"]) if str(f["box_index"]).isdigit() else -1, f["code"]))
    write_csv(out / FLAGS_NAME, flags, FLAGS_FIELDS)

    by_sev, by_code = Counter(f["severity"] for f in flags), Counter(f["code"] for f in flags)
    summary = {"images": len(anns), "nc": nc, "classes_json": bool(names), "flag_rows": len(flags),
               "ERROR": by_sev["ERROR"], "WARN": by_sev["WARN"], "INFO": by_sev["INFO"],
               "by_code": dict(by_code), **cross_summary}
    common.write_report(out, "boxes_check", summary, DEFERRED, args.note)
    common.append_journal(out_root, "boxes-check", summary, args)
    print(f"images={len(anns)}  nc={nc}  flag_rows={len(flags)}  by_code={dict(by_code)}")
    print(f"cross: both={cross_summary['cross_images_both']}  drift_pairs={cross_summary['cross_drift_pairs']}  "
          f"iou_p50={cross_summary['cross_iou_p50']}  unmatched_imgs={cross_summary['cross_images_unmatched']}")
    print(f"флаги: {out / FLAGS_NAME}")