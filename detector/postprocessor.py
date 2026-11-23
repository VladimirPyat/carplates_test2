"""Постпроцессинг детектора: абстрактный класс + реализация для YOLOv8.

    - разбор сырого выхода ONNX-модели (модель-специфичный формат)
    - конвертация xywh -> xyxy
    - денормализация из координат letterbox в пиксели оригинала
    - NMS и фильтры по confidence/классу/размеру
    - возврат list[BBox] в пикселях оригинала (наш контракт)
"""
from abc import ABC, abstractmethod

import cv2
import numpy as np

from .entities import BBox, PreprocessMeta


class Postprocessor(ABC):
    """Абстрактный постпроцессор. Каждая модель детекции имеет свою реализацию."""

    @abstractmethod
    def process(self, raw_output: np.ndarray, meta: PreprocessMeta) -> list[BBox]:
        """Разбирает сырой выход модели в список боксов.

        Args:
            raw_output: тензор выхода модели (формат зависит от архитектуры)
            meta: метаинформация о препроцессинге для денормализации

        Returns:
            list[BBox] в пикселях оригинального изображения
        """


class YOLOv8Postprocessor(Postprocessor):
    """Постпроцессор для YOLOv8.

    Формат выхода модели: (1, 4+nc, N)
        4 = cx, cy, w, h в пикселях letterbox
        nc = количество классов
        N = количество candidate-детекций (обычно 8400)
    """

    def __init__(
            self,
            conf_threshold: float = 0.35,
            iou_threshold: float = 0.45,
            class_filter: list[int] | None = None,
            min_box_size: int = 0,
            max_boxes: int | None = None,
    ):
        if not (0.0 < conf_threshold < 1.0):
            raise ValueError(f"conf_threshold должен быть в (0, 1), получил {conf_threshold}")
        if not (0.0 < iou_threshold < 1.0):
            raise ValueError(f"iou_threshold должен быть в (0, 1), получил {iou_threshold}")
        if min_box_size < 0:
            raise ValueError(f"min_box_size не может быть отрицательным: {min_box_size}")

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.class_filter = class_filter
        self.min_box_size = min_box_size
        self.max_boxes = max_boxes

    def process(self, raw_output: np.ndarray, meta: PreprocessMeta) -> list[BBox]:
        if raw_output.ndim != 2:
            raise ValueError(f"ожидается 2D тензор (4+nc, N), получил shape={raw_output.shape}")

        # (4+nc, N) -> (N, 4+nc)
        raw = raw_output.T
        if raw.shape[1] < 5:
            raise ValueError(f"тензор без классов: shape={raw.shape}")

        boxes_xywh = raw[:, :4]
        class_scores = raw[:, 4:]

        # xywh -> xyxy (в координатах letterbox)
        cx, cy, bw, bh = boxes_xywh[:, 0], boxes_xywh[:, 1], boxes_xywh[:, 2], boxes_xywh[:, 3]
        x1 = cx - bw / 2
        y1 = cy - bh / 2
        x2 = cx + bw / 2
        y2 = cy + bh / 2

        # лучший класс и его score
        best_cls = class_scores.argmax(axis=1)
        best_score = class_scores[np.arange(len(best_cls)), best_cls]

        # фильтр по confidence
        keep = best_score >= self.conf_threshold
        if not keep.any():
            return []

        x1, y1, x2, y2 = x1[keep], y1[keep], x2[keep], y2[keep]
        best_cls = best_cls[keep]
        best_score = best_score[keep]

        # фильтр по классу
        if self.class_filter is not None:
            allowed = np.isin(best_cls, self.class_filter)
            if not allowed.any():
                return []
            x1, y1, x2, y2 = x1[allowed], y1[allowed], x2[allowed], y2[allowed]
            best_cls = best_cls[allowed]
            best_score = best_score[allowed]

        # NMS
        indices = cv2.dnn.NMSBoxes(
            bboxes=np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist(),
            scores=best_score.tolist(),
            score_threshold=self.conf_threshold,
            nms_threshold=self.iou_threshold,
        )
        if len(indices) == 0:
            return []
        indices = np.array(indices).flatten()

        # ПРИМЕНИТЬ INDICES К МАССИВАМ КЛАССОВ И УВЕРЕННОСТИ!
        best_cls = best_cls[indices]
        best_score = best_score[indices]

        # денормализация: из letterbox в оригинал
        def undo(v: np.ndarray, pad: float, ratio: float) -> np.ndarray:
            return ((v - pad) / ratio).clip(0, None)

        x1 = undo(x1[indices], meta.pad_x, meta.ratio)
        y1 = undo(y1[indices], meta.pad_y, meta.ratio)
        x2 = undo(x2[indices], meta.pad_x, meta.ratio)
        y2 = undo(y2[indices], meta.pad_y, meta.ratio)

        # клип в границы оригинала
        x1 = x1.clip(0, meta.orig_w)
        y1 = y1.clip(0, meta.orig_h)
        x2 = x2.clip(0, meta.orig_w)
        y2 = y2.clip(0, meta.orig_h)

        # фильтр по минимальному размеру
        widths = x2 - x1
        heights = y2 - y1
        if self.min_box_size > 0:
            size_ok = (widths >= self.min_box_size) & (heights >= self.min_box_size)
            if not size_ok.any():
                return []
            x1, y1, x2, y2 = x1[size_ok], y1[size_ok], x2[size_ok], y2[size_ok]
            widths, heights = widths[size_ok], heights[size_ok]
            best_cls = best_cls[size_ok]
            best_score = best_score[size_ok]

        # top-k по уверенности
        if self.max_boxes is not None and len(best_score) > self.max_boxes:
            top_idx = np.argsort(best_score)[::-1][: self.max_boxes]
            x1, y1, x2, y2 = x1[top_idx], y1[top_idx], x2[top_idx], y2[top_idx]
            best_cls = best_cls[top_idx]
            best_score = best_score[top_idx]

        # #ОТЛАДКА: проверяем размеры всех массивов
        # print(f"[DEBUG] После всех фильтров:")
        # print(f"  x1.shape={x1.shape}, y1.shape={y1.shape}, x2.shape={x2.shape}, y2.shape={y2.shape}")
        # print(f"  best_cls.shape={best_cls.shape}, best_score.shape={best_score.shape}")
        # print(f"  widths.shape={widths.shape}, heights.shape={heights.shape}")


        # собираем BBox
        image_size = (meta.orig_w, meta.orig_h)
        return [
            BBox(
                x1=float(x1[i]),
                y1=float(y1[i]),
                x2=float(x2[i]),
                y2=float(y2[i]),
                class_id=int(best_cls[i]),
                confidence=float(best_score[i]),
                image_size=image_size,
            )
            for i in range(len(best_cls))
        ]