"""ONNX-инференс YOLOv8 + приватный препроцессинг.

Postprocess вынесен в отдельный класс Postprocessor (модель-специфичный).
Preprocess оставлен приватным методом (стандартный для 90% моделей).
"""
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from .config import DetectorConfig
from .entities import BBox, DetectionResult, PreprocessMeta
from .postprocessor import Postprocessor


class DetectorError(RuntimeError):
    """Базовое исключение детектора."""


class Detector:
    """ONNX-детектор: загрузка модели, препроцессинг, инференс."""

    def __init__(self, config: DetectorConfig, postprocessor: Postprocessor):
        """Инициализирует детектор.

        Args:
            config: конфигурация (пути, device, imgsz)
            postprocessor: постпроцессор для конкретной модели (YOLOv8, YOLOv5, etc.)
        """
        self.config = config
        self.postprocessor = postprocessor
        self.names = config.load_names()
        if not self.names:
            raise DetectorError(f"пустой список классов в {config.classes}")

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

        shape = inputs[0].shape
        if not (isinstance(shape, list) and len(shape) == 4):
            raise DetectorError(f"ожидается 4D вход (NCHW), получил shape={shape}")

    def detect(self, image: np.ndarray) -> DetectionResult:
        """Запускает полный пайплайн: preprocess → inference → postprocess.

        Args:
            image: np.ndarray shape=(H, W, 3), dtype=uint8, BGR

        Returns:
            DetectionResult с боксами и signature
        """
        if image is None:
            raise DetectorError("изображение - None")
        if image.ndim != 3 or image.shape[2] != 3:
            raise DetectorError(f"ожидается 3-канальное изображение, получил shape={image.shape}")

        h, w = image.shape[:2]
        blob, meta = self._preprocess(image)

        try:
            outputs = self.session.run(None, {self.input_name: blob})
        except Exception as e:
            raise DetectorError(f"ошибка инференса: {e}") from e

        if not outputs or outputs[0].ndim != 3:
            raise DetectorError(f"неожиданный выход модели: shape={outputs[0].shape if outputs else 'empty'}")

        boxes = self.postprocessor.process(outputs[0][0], meta)
        return DetectionResult.from_boxes(boxes, (w, h), self.names)

    def _preprocess(self, image: np.ndarray) -> tuple[np.ndarray, PreprocessMeta]:
        """Приватный препроцессинг: letterbox + normalize + transpose.

        Стандартный для 90% моделей, не выносим в отдельный класс.
        """
        h, w = image.shape[:2]
        target = self.config.imgsz

        # letterbox: сохраняем пропорции, pad серым (114)
        scale = min(target / h, target / w)
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((target, target, 3), 114, dtype=np.uint8)
        y_off = (target - new_h) // 2
        x_off = (target - new_w) // 2
        canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized

        # BGR → RGB
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

        # normalize [0, 1]
        blob = rgb.astype(np.float32) / 255.0

        # transpose HWC → CHW
        blob = blob.transpose(2, 0, 1)

        # add batch dimension: CHW → NCHW
        blob = np.expand_dims(blob, 0)

        meta = PreprocessMeta(
            ratio=scale,
            pad_x=float(x_off),
            pad_y=float(y_off),
            orig_w=w,
            orig_h=h,
        )
        return blob, meta