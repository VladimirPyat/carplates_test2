"""ONNX-инференс YOLOv8 + постпроцессинг.
Отсутствие детекций — НЕ ошибка: predict() возвращает DetectionResult
с пустыми массивами и signature=''. Ошибки (битый файл, неверный ONNX)
выбрасываются как исключения с понятным сообщением.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import onnxruntime as ort

from .config import DetectorConfig


@dataclass
class DetectionResult:
    """Результат детекции одного кадра.

    При отсутствии детекций: boxes.shape == (0, 4), labels/scores пустые,
    signature == ''. image_size всегда корректный (w, h) исходного кадра.
    """
    boxes: np.ndarray          # (N, 4) xyxy в пикселях исходного изображения
    labels: list[int]
    scores: list[float]
    signature: str
    image_size: tuple[int, int]  # (w, h)


class DetectorError(RuntimeError):
    """Базовое исключение детектора (файл/модель/инференс)."""


class Detector:
    """Минималистичный детектор на базе YOLOv8 ONNX."""

    def __init__(self, config: DetectorConfig):
        self.cfg = config
        try:
            self.names = config.load_names()
        except (FileNotFoundError, ValueError) as e:
            raise DetectorError(str(e)) from e

        if not config.model.is_file():
            raise DetectorError(f"файл модели не найден: {config.model}")

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if config.device == "cuda"
            else ["CPUExecutionProvider"]
        )
        try:
            self.session = ort.InferenceSession(str(config.model), providers=providers)
        except Exception as e:
            raise DetectorError(f"не удалось загрузить ONNX модель {config.model}: {e}") from e

        inputs = self.session.get_inputs()
        if not inputs:
            raise DetectorError(f"модель {config.model} не имеет входных тензоров")
        self.input_name = inputs[0].name

        # проверяем, что вход — 4D тензор NCHW (стандарт YOLOv8)
        shape = inputs[0].shape
        if not (isinstance(shape, list) and len(shape) == 4):
            raise DetectorError(f"ожидается 4D вход (NCHW), получил shape={shape}")

    # ---------- публичный API ----------

    def predict(self, image_path: Path) -> DetectionResult:
        """Запускает инференс. Возвращает пустой результат при отсутствии детекций."""
        if not image_path.is_file():
            raise DetectorError(f"файл не найден: {image_path}")

        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            raise DetectorError(f"не удалось прочитать изображение (битый файл?): {image_path}")

        h, w = img_bgr.shape[:2]
        if h == 0 or w == 0:
            raise DetectorError(f"изображение имеет нулевой размер: {image_path}")

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        letterbox, ratio, (pad_x, pad_y) = self._letterbox(img_rgb, self.cfg.imgsz)
        blob = np.expand_dims(letterbox.transpose(2, 0, 1), 0).astype(np.float32) / 255.0

        try:
            outputs = self.session.run(None, {self.input_name: blob})
        except Exception as e:
            raise DetectorError(f"ошибка инференса на {image_path.name}: {e}") from e

        if not outputs or outputs[0].ndim != 3:
            raise DetectorError(f"неожиданный выход модели: shape={outputs[0].shape if outputs else 'empty'}")

        boxes, scores, labels = self._postprocess(outputs[0][0], ratio, pad_x, pad_y, w, h)
        signature = self._build_signature(boxes, labels)
        return DetectionResult(
            boxes=boxes, labels=labels, scores=scores,
            signature=signature, image_size=(w, h),
        )

    # ---------- internals ----------

    @staticmethod
    def _letterbox(img: np.ndarray, target: int):
        h, w = img.shape[:2]
        scale = min(target / h, target / w)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((target, target, 3), 114, dtype=np.uint8)
        y_off = (target - new_h) // 2
        x_off = (target - new_w) // 2
        canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
        return canvas, scale, (x_off, y_off)

    def _postprocess(self, raw: np.ndarray, ratio: float,
                     pad_x: float, pad_y: float, orig_w: int, orig_h: int):
        """raw shape: (4+nc, N). Возвращает отфильтрованные и денормализованные боксы.

        Если детекций нет после фильтрации по confidence — возвращает пустые массивы.
        """
        raw = raw.T  # (N, 4+nc)
        if raw.shape[1] < 5:
            raise DetectorError(f"модель вернула тензор без классов: shape={raw.shape}")

        boxes_xywh = raw[:, :4]
        class_scores = raw[:, 4:]

        cx, cy, bw, bh = boxes_xywh[:, 0], boxes_xywh[:, 1], boxes_xywh[:, 2], boxes_xywh[:, 3]
        x1 = cx - bw / 2
        y1 = cy - bh / 2
        x2 = cx + bw / 2
        y2 = cy + bh / 2

        best_cls = class_scores.argmax(axis=1)
        best_score = class_scores[np.arange(len(best_cls)), best_cls]

        keep = best_score >= self.cfg.conf_threshold
        if not keep.any():
            return np.zeros((0, 4), dtype=np.float32), [], []

        x1, y1, x2, y2 = x1[keep], y1[keep], x2[keep], y2[keep]
        best_cls = best_cls[keep]
        best_score = best_score[keep]

        indices = cv2.dnn.NMSBoxes(
            bboxes=np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist(),
            scores=best_score.tolist(),
            score_threshold=self.cfg.conf_threshold,
            nms_threshold=self.cfg.iou_threshold,
        )
        if len(indices) == 0:
            return np.zeros((0, 4), dtype=np.float32), [], []
        indices = np.array(indices).flatten()

        def undo(v, pad, ratio):
            return ((v - pad) / ratio).clip(0, None)

        x1 = undo(x1[indices], pad_x, ratio)
        y1 = undo(y1[indices], pad_y, ratio)
        x2 = undo(x2[indices], pad_x, ratio)
        y2 = undo(y2[indices], pad_y, ratio)

        boxes = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, orig_w)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, orig_h)
        return boxes, best_score[indices].tolist(), best_cls[indices].tolist()

    def _build_signature(self, boxes: np.ndarray, labels: list[int]) -> str:
        """Собирает строку номера. Пустые боксы -> '' (не ошибка)."""
        if len(boxes) == 0:
            return ""
        order = np.argsort(boxes[:, 0] + boxes[:, 2])
        parts = []
        for i in order:
            lbl = labels[i]
            if lbl < 0 or lbl >= len(self.names):
                raise DetectorError(f"класс {lbl} вне диапазона [0, {len(self.names)})")
            parts.append(self.names[lbl])
        return "".join(parts)