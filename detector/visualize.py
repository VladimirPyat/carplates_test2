"""Отрисовка боксов и распознанной строки."""
from pathlib import Path
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np

from .detector import DetectionResult, DetectorError


def _palette(n: int) -> list[tuple[int, int, int]]:
    import colorsys
    return [
        tuple(int(c * 255) for c in colorsys.hsv_to_rgb(i / max(n, 1), 0.7, 0.95))
        for i in range(n)
    ]


def draw(image_path: Path, result: DetectionResult, names: list[str],
         save_path: Optional[Path] = None, show: bool = True) -> None:
    """Отрисовывает результат. При ошибке чтения/сохранения выбрасывает DetectorError."""
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise DetectorError(f"не удалось прочитать для отрисовки: {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    colors = _palette(len(names))
    for (x1, y1, x2, y2), label, score in zip(result.boxes, result.labels, result.scores):
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        color = colors[label % len(colors)]
        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), color, 2)
        text = f"{names[label]} {score:.2f}"
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img_rgb, (x1, y1 - th - baseline - 2), (x1 + tw, y1), color, -1)
        cv2.putText(img_rgb, text, (x1, y1 - baseline - 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    if result.signature:
        cv2.putText(img_rgb, result.signature, (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    else:
        cv2.putText(img_rgb, "(no detection)", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    if save_path:
        try:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_path), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
        except Exception as e:
            raise DetectorError(f"не удалось сохранить {save_path}: {e}") from e
    if show:
        plt.figure(figsize=(10, 8))
        plt.imshow(img_rgb)
        plt.title(f"signature: {result.signature or '(empty)'}  boxes: {len(result.boxes)}")
        plt.axis("off")
        plt.tight_layout()
        plt.show()