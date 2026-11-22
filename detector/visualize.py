"""Отрисовка боксов и распознанной строки."""
from pathlib import Path
from typing import Optional

import cv2
import matplotlib.pyplot as plt


from .config import DetectorConfig
from .detector import DetectionResult


# палитра по классам (HSV-равномерно)
def _palette(n: int) -> list[tuple[int, int, int]]:
    import colorsys
    return [
        tuple(int(c * 255) for c in colorsys.hsv_to_rgb(i / max(n, 1), 0.7, 0.95))
        for i in range(n)
    ]


def _save_yolo_txt(txt_path: Path, result: DetectionResult):
    """Сохраняет детекции в формате YOLO: class_id cx cy w h (нормализованные)."""
    w, h = result.image_size
    with open(txt_path, "w", encoding="utf-8") as f:
        for (x1, y1, x2, y2), label in zip(result.boxes, result.labels):
            cx = ((x1 + x2) / 2) / w
            cy = ((y1 + y2) / 2) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            f.write(f"{label} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")


def draw(image_path: Path, result: DetectionResult, names: list[str],
         config: DetectorConfig, save_path: Optional[Path] = None, show: bool = True):
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise FileNotFoundError(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    colors = _palette(len(names))

    # Рисуем боксы только если show_boxes=True
    if config.show_boxes:
        for (x1, y1, x2, y2), label, score in zip(result.boxes, result.labels, result.scores):
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            color = colors[label % len(colors)]
            cv2.rectangle(img_rgb, (x1, y1), (x2, y2), color, 2)

            # Показываем текст только если show_labels=True И conf >= порога
            if config.show_labels and score >= config.min_confidence_label:
                text = f"{names[label]} {score:.2f}"
                (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(img_rgb, (x1, y1 - th - baseline - 2), (x1 + tw, y1), color, -1)
                cv2.putText(img_rgb, text, (x1, y1 - baseline - 1),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Показываем подпись номера только если show_signature=True
    if config.show_signature:
        if result.signature:
            cv2.putText(img_rgb, result.signature, (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        else:
            cv2.putText(img_rgb, "(no detection)", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

        # Сохраняем txt рядом с картинкой
        if config.save_txt:
            txt_path = save_path.with_suffix(".txt")
            _save_yolo_txt(txt_path, result)

    if show:
        plt.figure(figsize=(10, 8))
        plt.imshow(img_rgb)
        title = f"signature: {result.signature or '(empty)'}"
        if config.verbose:
            title += f"  boxes: {len(result.boxes)}"
        plt.title(title)
        plt.axis("off")
        plt.tight_layout()
        plt.show()