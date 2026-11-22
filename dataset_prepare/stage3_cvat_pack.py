"""Этап 3: упаковка датасета для просмотра и правки в CVAT. Датасет не меняет.

Готовит два архива — CVAT принимает их в разных операциях:
images.zip       — плоский набор изображений, грузится при создании task
annotations.zip  — разметка YOLO 1.1, грузится через Upload annotation

Структура annotations.zip (по docs.cvat.ai, формат YOLO 1.1):
obj.data                    classes / names / train
obj.names                   имена классов, порядок строк = id
obj_train_data/<stem>.txt   метки; имя соответствует имени изображения
train.txt                   список путей вида obj_train_data/<image>

Порядок кадров берётся из констант stage2 (FRAME_SORT_KEY / FRAME_NUMBER_SCOPE),
поэтому frame_no из flags_boxes.csv соответствует кадрам задания.
Архив плоский: имя файла = путь в архиве, сортировка однозначна.

Запуск:
python3 dataset_prepare.py cvat-pack --root ./_dataset

Дальше в CVAT:
1. Create task -> images.zip (метод сортировки лексикографический)
2. Actions -> Import annotations -> YOLO 1.1 -> annotations.zip
3. Сверить имя первого кадра с first_frame из отчёта (и понять, 0-based или 1-based)
"""
import json
import zipfile
from collections import defaultdict
from pathlib import Path

from . import common
from . import stage1_audit_files as stage1
from . import stage2_boxes as stage2
from .text_utils import read_csv

# --- Структура формата CVAT YOLO 1.1 ---
OBJ_DATA_NAME = "obj.data"
OBJ_NAMES_NAME = "obj.names"
LABELS_DIR = "obj_train_data"        # валидные subset в формате: train, valid
TRAIN_LIST_NAME = "train.txt"

# --- Имена артефактов ---
IMAGES_ARC_STEM = "images"
ANNOTATIONS_ARC_STEM = "annotations"
ZIP_SUFFIX = ".zip"
ALL_KEY = "all"

# --- Упаковка ---
IMAGES_COMPRESSION = zipfile.ZIP_STORED      # jpeg/png уже сжаты
LABELS_COMPRESSION = zipfile.ZIP_DEFLATED    # текст жмётся хорошо

DEFERRED = [
    "Обратный импорт правок из CVAT в датасет — отдельная команда (имя -> путь по манифесту)",
    "Поиск похожих (DINO) и стратификация val — следующие этапы, после правок",
    "Мини-task только по flagged-кадрам — не делаем, ходим по frame_no в общем задании",
]


def register(sub):
    """Регистрирует команду cvat-pack: архивы изображений и разметки для CVAT."""
    p = sub.add_parser("cvat-pack", parents=[common.base_parser()],
                       help="подготовить images.zip + annotations.zip для CVAT")
    p.set_defaults(func=cmd_pack)


def _stage_dir(module) -> str:
    """Имя папки этапа = имя его модуля (контракт между этапами).

    Берём из __name__, а не __file__: __name__ есть у любого импортированного
    модуля, __file__ может быть None (namespace-пакеты, сбои импорта).
    """
    return module.__name__.rsplit(".", 1)[-1]


def _load_images(out_root: Path):
    """Читает строки изображений (kind=image) из манифеста этапа 1."""
    path = out_root / _stage_dir(stage1) / stage1.MANIFEST_NAME
    if not path.is_file():
        raise SystemExit(f"нет манифеста этапа 1: {path}\nСначала запусти files-find")
    return [r for r in read_csv(path) if r["kind"] == stage1.KIND_IMAGE]


def _load_names(out_root: Path):
    """Берёт упорядоченный список имён классов из classes.json этапа 2.

    Ключи и имена нормализуются strip(): пробелы в именах классов ломают
    сопоставление меток в CVAT.
    """
    path = out_root / _stage_dir(stage2) / stage2.CLASSES_NAME
    if not path.is_file():
        raise SystemExit(f"нет справочника классов: {path}\nСначала запусти boxes-stats")
    data = {k.strip(): v for k, v in json.loads(path.read_text(encoding="utf-8")).items()}
    names = [str(n).strip() for n in data.get("names", [])]
    if not names:
        raise SystemExit(f"в {path} пустой список names")
    return names


def _existing(root: Path, rows):
    """Отбрасывает строки манифеста, чьи файлы исчезли. Возвращает (rows, n_missing)."""
    ok = [r for r in rows if (root / r["rel_path"]).is_file()]
    return ok, len(rows) - len(ok)


def _check_unique_names(rows):
    """Проверяет уникальность имён файлов: архив плоский, коллизии недопустимы."""
    seen = defaultdict(list)
    for r in rows:
        seen[Path(r["rel_path"]).name].append(r["rel_path"])
    dupes = {k: v for k, v in seen.items() if len(v) > 1}
    if dupes:
        raise SystemExit(f"неуникальные имена файлов, плоский архив невозможен: {list(dupes.items())[:3]}")


def _sorted_rows(rows):
    """Сортирует кадры так же, как stage2 нумерует frame_no (единый источник констант)."""
    if stage2.FRAME_SORT_KEY == "name":
        return sorted(rows, key=lambda r: Path(r["rel_path"]).name)
    return sorted(rows, key=lambda r: r["rel_path"])


def _write_images_zip(root: Path, rows, arc_path: Path):
    """Пишет плоский архив изображений в порядке кадров. Возвращает число файлов."""
    with zipfile.ZipFile(arc_path, "w", compression=IMAGES_COMPRESSION) as z:
        for r in rows:
            z.write(root / r["rel_path"], arcname=Path(r["rel_path"]).name)
    return len(rows)


def _write_annotations_zip(root: Path, rows, names, arc_path: Path):
    """Пишет архив разметки YOLO 1.1: obj.data, obj.names, obj_train_data/*.txt, train.txt.

    Метки берутся из txt_path манифеста (канонический формат YOLO).
    Кадры без меток попадают в train.txt, но без txt — в CVAT это кадр без аннотаций.
    Возвращает (число меток, число кадров без меток).
    """
    n_labels = n_empty = 0
    lines = []
    with zipfile.ZipFile(arc_path, "w", compression=LABELS_COMPRESSION) as z:
        z.writestr(OBJ_DATA_NAME,
                   f"classes = {len(names)}\n"
                   f"names = {OBJ_NAMES_NAME}\n"
                   f"train = {TRAIN_LIST_NAME}\n")
        z.writestr(OBJ_NAMES_NAME, "\n".join(names) + "\n")
        for r in rows:
            name = Path(r["rel_path"]).name
            lines.append(f"{LABELS_DIR}/{name}")
            lp = root / r["txt_path"] if r["txt_path"] else None
            if lp is not None and lp.is_file():
                z.write(lp, arcname=f"{LABELS_DIR}/{Path(name).stem}.txt")
                n_labels += 1
            else:
                n_empty += 1
        z.writestr(TRAIN_LIST_NAME, "\n".join(lines) + "\n")
    return n_labels, n_empty


def cmd_pack(args):
    """Готовит архивы для CVAT: изображения + разметка YOLO 1.1, в порядке кадров.

    FRAME_NUMBER_SCOPE='global' -> один комплект на весь датасет,
    'split' -> отдельный комплект на каждый сплит (под отдельные задания CVAT).
    В отчёт пишется first_frame для сверки нумерации с заданием.
    """
    root = Path(args.root).resolve()
    out_root, out = common.prepare_out(args, __file__)
    rows, missing = _existing(root, _load_images(out_root))
    if not rows:
        raise SystemExit("в манифесте нет доступных изображений")
    names = _load_names(out_root)
    _check_unique_names(rows)

    if stage2.FRAME_NUMBER_SCOPE == "split":
        groups = defaultdict(list)
        for r in rows:
            groups[r["split"]].append(r)
    else:
        groups = {ALL_KEY: rows}

    summary = {"images_total": len(rows), "missing": missing, "nc": len(names),
               "sort_key": stage2.FRAME_SORT_KEY, "scope": stage2.FRAME_NUMBER_SCOPE}
    for key in sorted(groups):
        ordered = _sorted_rows(groups[key])
        tag = "" if key == ALL_KEY else f"_{key}"
        img_arc = out / f"{IMAGES_ARC_STEM}{tag}{ZIP_SUFFIX}"
        ann_arc = out / f"{ANNOTATIONS_ARC_STEM}{tag}{ZIP_SUFFIX}"
        n_img = _write_images_zip(root, ordered, img_arc)
        n_labels, n_empty = _write_annotations_zip(root, ordered, names, ann_arc)
        summary[f"{key}_frames"] = n_img
        summary[f"{key}_labels"] = n_labels
        summary[f"{key}_no_label"] = n_empty
        summary[f"{key}_first_frame"] = Path(ordered[0]["rel_path"]).name
        print(f"[{key}] frames={n_img}  labels={n_labels}  no_label={n_empty}  "
              f"first={summary[f'{key}_first_frame']}")
        print(f"  {img_arc}")
        print(f"  {ann_arc}")

    common.write_report(out, "cvat_pack", summary, DEFERRED, args.note)
    common.append_journal(out_root, "cvat-pack", summary, args)