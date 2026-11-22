"""Этап 4: пересборка сплитов по уникальности номеров (plan -> apply). Редактируем ТОЛЬКО train.

Правила:
    - подписи 1;1 (один кадр в train, один в test): TRAIN-копия уходит в quarantine,
      её test-пример становится уникальным (val чище); test не трогаем;
    - test_new = TEST_SIZE кадров из train-одиночек (count_train==1, count_test==0),
      отбор жадным покрытием классов (максимум непокрытых, приоритет редким);
      покрытие не закрылось -> отчёт рекомендует другой --seed или --allow-extra;
    - старый test остаётся как есть: пользователь сам переименует его в val позже,
      test_new позже переименовывается в test вручную;
    - кандидаты атомарны по построению: одиночные кадры своей подписи.

Подпись = имена классов боксов, отсортированные по центру X; кадр без боксов -> None,
остаётся в train (фоновые в test_new не попадают).

Запуск:
    python dataset_prepare.py split-plan --root ./_dataset    # план + покрытие, без правок
    python dataset_prepare.py split-apply --root ./_dataset   # переносы + split_log.csv
После apply: files-find, boxes-stats. Позже вручную: test -> val, test_new -> test.
"""
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from . import common, stage1_audit_files as stage1, stage2_boxes as stage2
from .text_utils import append_csv, read_csv, write_csv

# --- Параметры отбора ---
TEST_SIZE = 30            # лимит кадров в test_new: бережём train
EXTRA_CAP = 8             # максимум дозабора сверх лимита при --allow-extra
RANDOM_SEED = 42

# --- Артефакты и действия ---
PLAN_NAME = "split_plan.csv"
PLAN_FIELDS = ["rel_path", "signature", "cur_split", "target_split", "action", "reason"]
SPLIT_LOG_NAME = "split_log.csv"
SPLIT_LOG_FIELDS = ["ts", "action", "rel_path", "dst_path", "signature"]
ACT_QUARANTINE = "quarantine_train_twin"
ACT_TO_TEST_NEW = "move_to_test_new"
ACT_KEEP = "keep"

DEFERRED = [
    "val = старый test, переименовывается вручную позже; 15 общих подписей в нём остаются утечкой осознанно",
    "10 кадров val уникальны после карантина train-близнецов подписей 1;1",
    "Финальная честная оценка — только по бывшему test_new, трогать его до конца нельзя",
    "DINO-дедуп пропущен: близкие дубли внутри train не искажают holdout-оценку",
]


def register(sub):
    """Регистрирует split-plan (план и покрытие) и split-apply (переносы по плану)."""
    p = sub.add_parser("split-plan", parents=[common.base_parser()],
                       help="план: карантин train-близнецов 1;1 + отбор test_new из train-одиночек")
    p.add_argument("--test-size", type=int, default=TEST_SIZE, help="лимит кадров в test_new")
    p.add_argument("--seed", type=int, default=RANDOM_SEED, help="сид tie-break отбора")
    p.add_argument("--allow-extra", action="store_true", help="дозабор сверх лимита для покрытия классов")
    p.set_defaults(func=cmd_plan)
    sub.add_parser("split-apply", parents=[common.base_parser()],
                   help="исполнить split_plan.csv: переносы из train в quarantine и test_new").set_defaults(func=cmd_apply)


def _load_images(out_root: Path):
    """Читает строки изображений (kind=image) из manifest этапа 1."""
    path = out_root / Path(stage1.__file__).stem / stage1.MANIFEST_NAME
    if not path.is_file():
        raise SystemExit(f"нет manifest этапа 1: {path}\nСначала запусти files-find")
    return [r for r in read_csv(path) if r["kind"] == stage1.KIND_IMAGE]


def _load_names(out_root: Path):
    """Берёт имена классов из classes.json этапа 2 (порядок = id), strip от пробелов."""
    import json
    path = out_root / Path(stage2.__file__).stem / stage2.CLASSES_NAME
    if not path.is_file():
        raise SystemExit(f"нет справочника классов: {path}\nСначала запусти boxes-stats")
    data = {k.strip(): v for k, v in json.loads(path.read_text(encoding="utf-8")).items()}
    return [str(n).strip() for n in data.get("names", [])]


def _signatures(root: Path, rows, names):
    """Подпись и счётчики классов каждого кадра из YOLO txt. Кадр без боксов -> подпись None."""
    sigs, class_counts = {}, {}
    for r in rows:
        rel = r["rel_path"]
        counts, boxes = Counter(), []
        if r["txt_path"]:
            lp = root / r["txt_path"]
            if lp.is_file():
                boxes, _ = stage2._parse_yolo(lp, int(r["width"] or 0), int(r["height"] or 0))
        for b in boxes:
            counts[b["cls"]] += 1
        sigs[rel] = stage2._signature(boxes, names) or None
        class_counts[rel] = counts
    return sigs, class_counts


def _select_test(pool, class_counts, totals, test_size, allow_extra, seed):
    """Жадный отбор: каждый шаг берём кадр с максимумом непокрытых классов (редкие важнее).

    После закрытия покрытия дозаполняем до test_size кадрами с редкими классами.
    Если покрытие не закрылось и allow_extra — добираем до EXTRA_CAP сверх лимита.
    Возвращает (selected, sel_cls, missing).
    """
    import random
    rng = random.Random(seed)
    tie = {rel: rng.random() for rel in pool}
    uncovered = {c for c in totals if totals[c] > 0}
    selected, sel_cls, rest = [], Counter(), list(pool)

    def cover_score(rel):
        gain = uncovered & set(class_counts[rel])
        return (len(gain), sum(1.0 / max(1, totals[c]) for c in gain), tie[rel])

    while rest and len(selected) < test_size and uncovered:
        rest.sort(key=cover_score, reverse=True)
        rel = rest.pop(0)
        selected.append(rel)
        sel_cls.update(class_counts[rel])
        uncovered -= set(class_counts[rel])
    while rest and len(selected) < test_size:
        rest.sort(key=lambda r: (sum(1.0 / max(1, totals[c]) for c in class_counts[r]), tie[r]), reverse=True)
        rel = rest.pop(0)
        selected.append(rel)
        sel_cls.update(class_counts[rel])
    if uncovered and allow_extra:
        while rest and uncovered and len(selected) < test_size + EXTRA_CAP:
            rest.sort(key=cover_score, reverse=True)
            rel = rest.pop(0)
            selected.append(rel)
            sel_cls.update(class_counts[rel])
            uncovered -= set(class_counts[rel])
    return selected, sel_cls, uncovered


def cmd_plan(args):
    """Строит split_plan.csv: карантин train-близнецов 1;1 + отбор test_new. Датасет не меняет.

    Подписи пересчитываются здесь из живых txt (функция _signatures); combinations.csv
    не читается — это снапшот этапа 2. Для синхронизации combinations.csv перезапусти
    boxes-stats после любых движений файлов.
    """
    root = Path(args.root).resolve()
    out_root, out = common.prepare_out(args, __file__)
    rows = [r for r in _load_images(out_root) if (root / r["rel_path"]).is_file()]
    names = _load_names(out_root)
    sigs, class_counts = _signatures(root, rows, names)
    splits = {r["rel_path"]: r["split"] for r in rows}

    # сквозные счётчики подписей по сплитам
    train_cnt, test_cnt = Counter(), Counter()
    for rel, sig in sigs.items():
        if sig is None:
            continue
        (train_cnt if splits[rel] == common.TRAIN_DIR else test_cnt)[sig] += 1

    # val = старый test: состав сейчас + что станет уникальным после карантина близнецов
    plan = []
    n_val = n_val_unique = n_val_nosig = 0
    for rel in sorted(sigs):
        sig, cur = sigs[rel], splits[rel]
        if cur == common.TEST_DIR:
            n_val += 1
            if sig is None:
                n_val_nosig += 1
            elif train_cnt[sig] == 0:
                n_val_unique += 1
            continue
        if sig is None:
            continue
        if train_cnt[sig] == 1 and test_cnt[sig] == 1:
            plan.append({"rel_path": rel, "signature": sig, "cur_split": common.TRAIN_DIR,
                         "target_split": common.UNKNOWN_SPLIT, "action": ACT_QUARANTINE,
                         "reason": "train-близнец 1;1: убираем, test-пример станет уникальным (val)"})

    # кандидаты в test_new: train-одиночки, отсутствующие в test
    candidates = [rel for rel in sorted(sigs)
                  if sigs[rel] is not None and splits[rel] == common.TRAIN_DIR
                  and train_cnt[sigs[rel]] == 1 and test_cnt[sigs[rel]] == 0]
    totals, n_train = Counter(), 0
    for rel in splits:
        if splits[rel] == common.TRAIN_DIR:
            n_train += 1
            totals.update(class_counts[rel])
    selected, sel_cls, missing = _select_test(candidates, class_counts, totals,
                                              args.test_size, args.allow_extra, args.seed)
    for i, rel in enumerate(selected):
        plan.append({"rel_path": rel, "signature": sigs[rel], "cur_split": common.TRAIN_DIR,
                     "target_split": common.TEST_NEW_DIR, "action": ACT_TO_TEST_NEW,
                     "reason": "extra_for_coverage" if i >= args.test_size else "train_only_singleton"})

    # сводка: все числа про val — ПОСЛЕ будущего apply
    n_quar = sum(1 for p in plan if p["action"] == ACT_QUARANTINE)
    val_unique_after = n_val_unique + n_quar
    summary = {"pool": len(candidates), "test_new": len(selected), "quarantine": n_quar,
               "train_after": n_train - len(selected) - n_quar,
               "val_unchanged": n_val,
               "val_unique_after": val_unique_after,
               "val_shared_leak_after": n_val - n_val_nosig - val_unique_after,
               "val_no_signature": n_val_nosig,
               "classes_covered": sum(1 for c in totals if sel_cls[c] > 0),
               "classes_total": len(totals),
               "classes_missing": sorted(names[c] if c < len(names) else str(c) for c in missing),
               "test_new_min_class_instances": min((sel_cls[c] for c in totals), default=0)}
    if missing:
        summary["recommendation"] = ("покрытие не закрылось: перезапусти split-plan с другим --seed "
                                     "или добавь --allow-extra")
    write_csv(out / PLAN_NAME, plan, PLAN_FIELDS)
    common.write_report(out, "split_plan", summary, DEFERRED, args.note)
    common.append_journal(out_root, "split-plan", summary, args)

    print(f"pool={len(candidates)}  test_new={len(selected)}  quarantine={n_quar}  "
          f"train_after={summary['train_after']}")
    print(f"val: всего={n_val}, уникальных после apply={val_unique_after}, "
          f"останется общих (утечка, осознанно)={summary['val_shared_leak_after']}, "
          f"без подписи={n_val_nosig}")
    print(f"coverage test_new: {summary['classes_covered']}/{summary['classes_total']} классов, "
          f"min instances={summary['test_new_min_class_instances']}")
    if missing:
        print(f"ПРОПУЩЕНЫ классы: {summary['classes_missing']} -> {summary['recommendation']}")
    print(f"план: {out / PLAN_NAME} — проверь и запусти split-apply")


def cmd_apply(args):
    """Исполняет split_plan.csv: train-близнецы 1;1 в quarantine, отобранные кадры в test_new/."""
    root = Path(args.root).resolve()
    out_root, out = common.prepare_out(args, __file__)
    plan_path = out / PLAN_NAME
    if not plan_path.is_file():
        raise SystemExit(f"нет плана: {plan_path}\nСначала запусти split-plan")
    rows_by_rel = {r["rel_path"]: r for r in _load_images(out_root)}
    quarantine = out / stage1.QUARANTINE_DIR
    test_new = root / common.TEST_NEW_DIR

    log, ts, skipped = [], datetime.now(timezone.utc).isoformat(), 0
    for p in read_csv(plan_path):
        if p["action"] == ACT_KEEP:
            continue
        row = rows_by_rel.get(p["rel_path"])
        if row is None or not (root / p["rel_path"]).is_file():
            skipped += 1
            continue
        for rel in (p["rel_path"], row["txt_path"]):
            if not rel:
                continue
            src = root / rel
            if not src.is_file():
                continue
            dst = (quarantine / rel) if p["action"] == ACT_QUARANTINE else (test_new / Path(rel).name)
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                shutil.move(str(src), str(dst))
        log.append({"ts": ts, "action": p["action"], "rel_path": p["rel_path"],
                    "dst_path": str((quarantine / p["rel_path"]) if p["action"] == ACT_QUARANTINE
                                    else (test_new / Path(p["rel_path"]).name)),
                    "signature": p["signature"]})
    append_csv(out / SPLIT_LOG_NAME, log, SPLIT_LOG_FIELDS)
    summary = {"moved": len(log), "skipped": skipped}
    common.write_report(out, "split_apply", summary, DEFERRED, args.note)
    common.append_journal(out_root, "split-apply", summary, args)
    print(f"moved={len(log)}  skipped={skipped}  log: {out / SPLIT_LOG_NAME}")
    print("Дальше: files-find + boxes-stats; затем вручную test -> val и test_new -> test")