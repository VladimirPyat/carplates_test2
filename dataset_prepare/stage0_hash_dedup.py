"""Этап 0: точные дубли по md5. Только изображения, метки не трогаем.


Запуск проверки:
    python3 dataset_prepare.py hash-find --root ./_dataset --copy-to ./dataset_prepare/dupes_review
    # смотрим dupes_review/group_0001/...,
    # отчет в <out>/dupes_exact.csv (keep / remove / skip)

Удаление найденных дублей (перемещение в quarantine, не delete):
    python3 dataset_prepare.py hash-clean --root ./_dataset           # dry-run, только план
    python3 dataset_prepare.py hash-clean --root ./_dataset --apply   # переместить + лог
"""
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from . import common
from .text_utils import append_csv, read_csv, write_csv

PLAN_FIELDS = ["group_id", "md5", "rel_path", "split", "action", "reason"]
# test не трогаем: при cross-split дубле остаётся он, копия из train уходит
SPLIT_PRIORITY = {common.TEST_DIR: 0, common.VAL_DIR: 1,
                  common.UNKNOWN_SPLIT: 2, common.TRAIN_DIR: 3}

# то что осознанно пропустили
DEFERRED = [
]


def register(sub):
    """Регистрирует команды этапа 0 в общем CLI (dataset_prepare.py).

    Создаёт субпарсеры hash-find и hash-clean, привязывает обработчики
    через set_defaults(func=...). Главный файл вызывает только args.func(args).
    """
    p = sub.add_parser("hash-find", parents=[common.base_parser()],
                       help="найти точные дубли по md5, составить план очистки")
    p.add_argument("--copy-to", default=None, help="скопировать группы дублей сюда для просмотра")
    p.set_defaults(func=cmd_find)

    p = sub.add_parser("hash-clean", parents=[common.base_parser()],
                       help="убрать дубли по плану в quarantine (по умолчанию dry-run)")
    p.add_argument("--plan", default=None, help="csv плана (по умолчанию out/dupes_exact.csv)")
    p.add_argument("--quarantine", default=None, help="по умолчанию out/quarantine")
    p.add_argument("--apply", action="store_true", help="без флага только dry-run")
    p.set_defaults(func=cmd_clean)


def cmd_find(args):
    """Ищет точные дубли изображений по md5 и составляет план очистки.

    Сканирует root (пропуская out и --copy-to), группирует по хэшу.
    В группе >1 выбирает keep по приоритету сплитов
    (test > val > unknown > train, внутри сплита — первый по пути),
    остальным ставит action=remove.
    Артефакты: out/dupes_exact.csv (редактируемый план),
    out/hash_find.md/json (сводка + DEFERRED + notes).
    При --copy-to копирует группы для просмотра глазами.
    В датасете ничего не меняет.
    """
    root = Path(args.root).resolve()
    out_root, out = common.prepare_out(args, __file__)
    skip = [out_root] + ([Path(args.copy_to)] if args.copy_to else [])

    groups, n = defaultdict(list), 0
    for p in common.iter_images(root, skip=skip):
        groups[common.md5_file(p)].append(p.relative_to(root))
        n += 1

    plan, gid, cross = [], 0, 0
    for h, members in sorted(groups.items()):
        if len(members) < 2:
            continue
        gid += 1
        ordered = sorted(members, key=lambda r: (SPLIT_PRIORITY.get(common.split_of(r), 2), str(r)))
        keep = ordered[0]
        splits = {common.split_of(r) for r in members}
        cross += len(splits - {common.UNKNOWN_SPLIT}) > 1
        for rel in ordered:
            plan.append({"group_id": gid, "md5": h, "rel_path": str(rel),
                         "split": common.split_of(rel),
                         "action": "keep" if rel == keep else "remove",
                         "reason": "keep by split priority" if rel == keep else f"duplicate of {keep}"})

    write_csv(out / "dupes_exact.csv", plan, PLAN_FIELDS)
    if args.copy_to:
        _copy_groups(root, plan, Path(args.copy_to))

    summary = {"images_scanned": n, "dup_groups": gid, "cross_split_groups": cross,
               "to_remove": sum(r["action"] == "remove" for r in plan)}
    common.write_report(out, "hash_find", summary, DEFERRED, args.note)
    common.append_journal(out_root, "hash-find", summary, args)
    print(f"scanned={n}  dup_groups={gid}  cross_split={cross}  to_remove={summary['to_remove']}")
    print(f"план: {out / 'dupes_exact.csv'}")


def _copy_groups(root: Path, plan, dst: Path):
    """Копирует группы дублей в dst/group_0001/... для визуального просмотра.

    Все члены одной группы — в одной папке. Только копии (copy2),
    исходники не трогаются.
    """
    by_group = defaultdict(list)
    for r in plan:
        by_group[r["group_id"]].append(r["rel_path"])
    for gid, paths in sorted(by_group.items()):
        gdir = dst / f"group_{int(gid):04d}"
        gdir.mkdir(parents=True, exist_ok=True)
        for rel in paths:
            shutil.copy2(root / rel, gdir / Path(rel).name)


def cmd_clean(args):
    """Исполняет план: перемещает файлы с action=remove в quarantine.

    Перед перемещением проверяет существование файла и совпадение md5
    с планом (не менялся ли после find). Без --apply — только dry-run.
    Перемещение, не удаление, с сохранением относительного пути.

    Артефакты: out/removed_log.csv (append: что, куда, когда, md5),
    out/hash_clean.md/json. Метки не трогает — сирот поймает этап 1.
    """
    root = Path(args.root).resolve()
    out_root, out = common.prepare_out(args, __file__)
    plan_path = Path(args.plan) if args.plan else out / "dupes_exact.csv"
    quarantine = Path(args.quarantine) if args.quarantine else out / "quarantine"

    moves, skipped = [], []
    for r in read_csv(plan_path):
        if r["action"] != "remove":
            continue
        src = root / r["rel_path"]
        if not src.is_file():
            skipped.append((r["rel_path"], "missing"))
        elif common.md5_file(src) != r["md5"]:
            skipped.append((r["rel_path"], "md5 changed since find"))
        else:
            moves.append((src, quarantine / r["rel_path"], r))

    if not args.apply:
        print(f"DRY-RUN: к перемещению {len(moves)}, пропущено {len(skipped)}. Запусти с --apply")
        return

    log = []
    for src, dst, r in moves:
        if dst.exists():
            skipped.append((r["rel_path"], "quarantine path exists"))
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        log.append({"ts": datetime.now(timezone.utc).isoformat(), "rel_path": r["rel_path"],
                    "quarantine_path": str(dst), "md5": r["md5"], "group_id": r["group_id"]})

    append_csv(out / "removed_log.csv", log, ["ts", "rel_path", "quarantine_path", "md5", "group_id"])
    summary = {"moved": len(log), "skipped": len(skipped), "quarantine": str(quarantine)}
    common.write_report(out, "hash_clean", summary, DEFERRED, args.note)
    common.append_journal(out_root, "hash-clean", summary, args)
    print(f"moved={len(log)}  skipped={len(skipped)}  лог: {out / 'removed_log.csv'}")