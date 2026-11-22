"""CLI-точка входа: python -m detector.cli --image path.jpg"""
import argparse
import sys
from pathlib import Path

from .config import DetectorConfig
from .detector import Detector, DetectorError
from .visualize import draw

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


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


def main(argv: list[str] | None = None) -> int:
    """Возвращает 0 при успехе, >0 при ошибках. Код = число неудачных файлов."""
    p = argparse.ArgumentParser(description="CLI-детектор номерных знаков (YOLOv8 ONNX)")
    p.add_argument("--image", type=Path, help="путь к изображению")
    p.add_argument("--dir", type=Path, help="папка с изображениями (батч)")
    p.add_argument("--config", type=Path, default=None, help="путь к config.yaml")
    p.add_argument("--no-show", action="store_true", help="не открывать matplotlib окно")
    args = p.parse_args(argv)

    if not args.image and not args.dir:
        p.error("укажите --image или --dir")

    try:
        config_path = _resolve_config(args.config, "config.yaml")
        cfg = DetectorConfig.from_yaml(config_path)
        if args.no_show:
            cfg.show = False
        detector = Detector(cfg)
    except (FileNotFoundError, ValueError, DetectorError) as e:
        print(f"[FATAL] {e}", file=sys.stderr)
        return 2

    print(f"модель: {cfg.model}  классов: {len(detector.names)}  imgsz={cfg.imgsz}")
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
        images = sorted(p for p in args.dir.iterdir() if p.suffix.lower() in IMG_EXTS)
        if not images:
            print(f"[WARN] в {args.dir} нет изображений ({', '.join(sorted(IMG_EXTS))})", file=sys.stderr)
            return 0

    errors = 0
    for img in images:
        try:
            result = detector.predict(img)
        except DetectorError as e:
            print(f"[ERROR] {img.name}: {e}", file=sys.stderr)
            errors += 1
            continue

        status = f"signature='{result.signature}'" if result.signature else "(no detection)"
        print(f"\n[{img.name}] {status}  boxes={len(result.boxes)}")
        for (x1, y1, x2, y2), label, score in zip(result.boxes, result.labels, result.scores):
            print(f"  {detector.names[label]:>2}  {score:.3f}  [{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]")

        save_path = cfg.output_dir / f"{img.stem}_pred.jpg" if cfg.save else None
        try:
            draw(img, result, detector.names, save_path=save_path, show=cfg.show)
        except DetectorError as e:
            print(f"[WARN] отрисовка {img.name} не удалась: {e}", file=sys.stderr)
            # отрисовка — не критична, инференс уже прошёл

    if errors:
        print(f"\n{errors} из {len(images)} файлов завершились с ошибкой", file=sys.stderr)
    return errors


if __name__ == "__main__":
    sys.exit(main())