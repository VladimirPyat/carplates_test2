"""CLI-точка входа: python plates_detect.py --image path.jpg"""
import argparse
import sys
from pathlib import Path
from typing import Optional

from .config import DetectorConfig
from .detector import Detector
from .visualize import draw


def _resolve_config(explicit: Optional[Path], default_name: str) -> Path:
    if explicit:
        return explicit
    candidates = [Path.cwd() / default_name, Path(__file__).resolve().parent.parent / default_name]
    for c in candidates:
        if c.is_file():
            return c
    raise FileNotFoundError(f"конфиг не найден: {default_name}")


def main(argv: Optional[list[str]] = None):
    p = argparse.ArgumentParser(description="CLI-детектор номерных знаков (YOLOv8 ONNX)")
    p.add_argument("--image", type=Path, help="путь к изображению")
    p.add_argument("--dir", type=Path, help="папка с изображениями (батч)")
    p.add_argument("--config", type=Path, default=None, help="путь к config.yaml")
    p.add_argument("--no-show", action="store_true", help="не открывать matplotlib окно")
    p.add_argument("--quiet", action="store_true", help="только signature, без списка боксов")
    args = p.parse_args(argv)

    if not args.image and not args.dir:
        p.error("укажите --image или --dir")

    config_path = _resolve_config(args.config, "config.yaml")
    cfg = DetectorConfig.from_yaml(config_path)
    if args.no_show:
        cfg.show = False
    if args.quiet:
        cfg.verbose = False

    detector = Detector(cfg)

    # Краткий вывод настроек (только если не quiet)
    if cfg.verbose:
        print(f"модель: {cfg.model}  классов: {len(detector.names)}  imgsz={cfg.imgsz}")
        print(f"conf={cfg.conf_threshold}  iou={cfg.iou_threshold}  device={cfg.device}")

    images: list[Path] = []
    if args.image:
        images = [args.image]
    elif args.dir:
        images = sorted(p for p in args.dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})

    for img in images:
        try:
            result = detector.predict(img)
        except Exception as e:
            print(f"[ERROR] {img.name}: {e}", file=sys.stderr)
            continue

        # Основной вывод: signature всегда
        print(f"\n[{img.name}] signature='{result.signature}'  boxes={len(result.boxes)}")

        # Детальный вывод: только если verbose
        if cfg.verbose:
            for (x1, y1, x2, y2), label, score in zip(result.boxes, result.labels, result.scores):
                print(f"  {detector.names[label]:>2}  {score:.3f}  [{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]")

        save_path = cfg.output_dir / f"{img.stem}_pred.jpg" if cfg.save else None
        draw(img, result, detector.names, cfg, save_path=save_path, show=cfg.show)


if __name__ == "__main__":
    main()