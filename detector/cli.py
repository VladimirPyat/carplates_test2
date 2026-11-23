"""CLI-точка входа: python plates_detect.py --image path.jpg"""
import argparse
import sys
from pathlib import Path

from .config import DetectorConfig
from .visualize import draw


def _resolve_config(explicit: Path | None, default_name: str) -> Path:
    if explicit:
        if not explicit.is_file():
            raise FileNotFoundError(f"конфиг не найден: {explicit}")
        return explicit
    candidates = [Path.cwd() / default_name, Path(__file__).resolve().parent.parent / default_name]
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError(f"конфиг не найден: {default_name}")


def cli_main(argv: list[str] | None = None) -> int:
    """CLI-точка входа. Возвращает 0 при успехе, >0 при ошибках."""
    # импортируем здесь, чтобы избежать циклических зависимостей
    from plates_detect import Pipeline

    p = argparse.ArgumentParser(description="CLI-детектор номерных знаков (YOLOv8 ONNX)")
    p.add_argument("--image", type=Path, help="путь к изображению")
    p.add_argument("--dir", type=Path, help="папка с изображениями (батч)")
    p.add_argument("--config", type=Path, default=None, help="путь к config.yaml")
    p.add_argument("--no-show", action="store_true", help="не открывать matplotlib окно")
    p.add_argument("--quiet", action="store_true", help="только signature, без списка боксов")
    args = p.parse_args(argv)

    if not args.image and not args.dir:
        p.error("укажите --image или --dir")

    try:
        config_path = _resolve_config(args.config, "config.yaml")
        pipeline = Pipeline.from_yaml(config_path)
        if args.no_show:
            pipeline.config.show = False
        if args.quiet:
            pipeline.config.verbose = False
    except (FileNotFoundError, ValueError, Exception) as e:
        print(f"[FATAL] {e}", file=sys.stderr)
        return 2

    cfg = pipeline.config

    # Краткий вывод настроек (только если не quiet)
    if cfg.verbose:
        print(f"модель: {cfg.model}  классов: {len(pipeline.detector.names)}  imgsz={cfg.imgsz}")
        print(f"conf={cfg.conf_threshold}  iou={cfg.iou_threshold}  device={cfg.device}")

    images: list[Path] = []
    if args.image:
        if not args.image.is_file():
            print(f"[FATAL] файл не найден: {args.image}", file=sys.stderr)
            return 2
        images = [args.image]
    elif args.dir:
        if not args.dir.is_dir():
            print(f"[FATAL] папка не найдена: {args.dir}", file=sys.stderr)
            return 2
        images = sorted(p for p in args.dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if not images:
            print(f"[WARN] в {args.dir} нет изображений", file=sys.stderr)
            return 0

    errors = 0
    for img in images:
        try:
            result = pipeline.run(img)
        except Exception as e:
            print(f"[ERROR] {img.name}: {e}", file=sys.stderr)
            errors += 1
            continue
            # import traceback
            # traceback.print_exc()
            # print(f"[ERROR] {img.name}: {e}", file=sys.stderr)
            # errors += 1
            # continue

        # Основной вывод: signature всегда
        status = f"signature='{result.signature}'" if result.signature else "(no detection)"
        print(f"\n[{img.name}] {status}  boxes={len(result.boxes)}")

        # Детальный вывод: только если verbose
        if cfg.verbose:
            for box in result.boxes:
                print(f"  {pipeline.detector.names[box.class_id]:>2}  {box.confidence:.3f}  "
                      f"[{box.x1:.0f},{box.y1:.0f},{box.x2:.0f},{box.y2:.0f}]")

        save_path = cfg.output_dir / f"{img.stem}_pred.jpg" if cfg.save else None
        try:
            draw(img, result, pipeline.detector.names, cfg, save_path=save_path, show=cfg.show)
        except Exception as e:
            print(f"[WARN] отрисовка {img.name} не удалась: {e}", file=sys.stderr)

    if errors:
        print(f"\n{errors} из {len(images)} файлов завершились с ошибкой", file=sys.stderr)
    return errors