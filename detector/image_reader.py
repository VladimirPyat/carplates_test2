"""Чтение изображения из различных источников.

Поддерживает:
    - Path: путь к файлу на диске
    - bytes: сырые байты изображения (например, из HTTP-запроса)
    - np.ndarray: уже загруженное изображение (для тестов и pipeline)

Всегда возвращает np.ndarray в формате BGR, HxWx3, uint8.
При ошибке чтения выбрасывает ImageReadError.
"""
from pathlib import Path


import cv2
import numpy as np


class ImageReadError(Exception):
    """Ошибка чтения изображения из источника."""


class ImageReader:
    """Читает изображение из различных источников."""

    @staticmethod
    def read(source: Path | bytes | np.ndarray) -> np.ndarray:
        """Читает изображение из источника.

        Args:
            source: Path (файл), bytes (сырые данные) или np.ndarray (уже загружено)

        Returns:
            np.ndarray shape=(H, W, 3), dtype=uint8, каналы BGR

        Raises:
            ImageReadError: если не удалось прочитать или неверный формат
        """
        if isinstance(source, np.ndarray):
            if source.ndim != 3 or source.shape[2] != 3:
                raise ImageReadError(
                    f"ожидается 3-канальное изображение (H, W, 3), получил shape={source.shape}"
                )
            if source.dtype != np.uint8:
                raise ImageReadError(f"ожидается dtype=uint8, получил {source.dtype}")
            return source

        if isinstance(source, bytes):
            arr = np.frombuffer(source, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                raise ImageReadError("не удалось декодировать изображение из bytes")
            return img

        if isinstance(source, Path):
            if not source.is_file():
                raise ImageReadError(f"файл не найден: {source}")
            img = cv2.imread(str(source))
            if img is None:
                raise ImageReadError(f"не удалось прочитать файл (битый файл?): {source}")
            return img

        raise ImageReadError(f"неподдерживаемый тип источника: {type(source)}")