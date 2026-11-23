#!/usr/bin/env python3
"""Оркестратор детектора: класс Pipeline, собирающий все компоненты.

Примеры использования:
    # CLI
    python plates_detect.py --image photo.jpg
    python plates_detect.py --dir _dataset/test/ --no-show

    # Программно
    from plates_detect import Pipeline
    pipeline = Pipeline.from_yaml("config.yaml")
    result = pipeline.run("photo.jpg")
    print(result.signature)
"""
import sys
from pathlib import Path

from detector.config import DetectorConfig
from detector.detector import Detector
from detector.image_reader import ImageReader
from detector.postprocessor import YOLOv8Postprocessor
from detector.entities import DetectionResult
from detector.visualize import draw


class Pipeline:
    """Оркестратор детектора: собирает все компоненты и запускает пайплайн.

    Компоненты:
        - ImageReader: чтение изображения из любого источника
        - Detector: загрузка ONNX, препроцессинг, инференс
        - Postprocessor: разбор выхода модели, NMS, фильтры
        - DetectionResult: финальный результат с signature

    Модель загружается один раз при создании Pipeline, переиспользуется для всех вызовов.
    """

    def __init__(self, config: DetectorConfig):
        """Инициализирует пайплайн с заданной конфигурацией.

        Args:
            config: конфигурация детектора (пути, пороги, фильтры)
        """
        self.config = config
        postprocessor = YOLOv8Postprocessor(
            conf_threshold=config.conf_threshold,
            iou_threshold=config.iou_threshold,
            class_filter=config.class_filter,
            min_box_size=config.min_box_size,
            max_boxes=config.max_boxes,
        )
        self.detector = Detector(config, postprocessor)

    @classmethod
    def from_yaml(cls, config_path: Path) -> "Pipeline":
        """Создаёт Pipeline из YAML-конфига.

        Args:
            config_path: путь к config.yaml

        Returns:
            Pipeline с загруженной моделью
        """
        config = DetectorConfig.from_yaml(config_path)
        return cls(config)

    def run(self, source: Path | bytes | object) -> DetectionResult:
        """Запускает полный пайплайн детекции.

        Args:
            source: источник изображения (Path, bytes или np.ndarray)

        Returns:
            DetectionResult с боксами и signature

        Raises:
            ImageReadError: если не удалось прочитать изображение
            DetectorError: если ошибка инференса
        """
        image = ImageReader.read(source)
        return self.detector.detect(image)


def main():
    """CLI-точка входа. Импортирует и запускает cli.main()."""
    from detector.cli import cli_main
    sys.exit(cli_main())


if __name__ == "__main__":
    main()