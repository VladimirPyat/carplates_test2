"""Датаклассы для детектора: BBox, DetectionResult, PreprocessMeta."""
from dataclasses import dataclass


@dataclass
class PreprocessMeta:
    """Метаинформация о препроцессинге изображения. Нужна для денормализации координат."""
    ratio: float          # масштаб letterbox (min(imgsz/w, imgsz/h))
    pad_x: float          # горизонтальный отступ в letterbox (пиксели)
    pad_y: float          # вертикальный отступ в letterbox (пиксели)
    orig_w: int           # ширина оригинального изображения
    orig_h: int           # высота оригинального изображения


@dataclass
class BBox:
    """Bounding box в пикселях оригинального изображения.
    Формат: xyxy (верхний левый угол → нижний правый угол).
    """
    x1: float
    y1: float
    x2: float
    y2: float
    class_id: int
    confidence: float
    image_size: tuple[int, int] | None = None # (width, height) оригинала

    @property
    def cx(self) -> float:
        """Центр X бокса."""
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        """Центр Y бокса."""
        return (self.y1 + self.y2) / 2

    @property
    def width(self) -> float:
        """Ширина бокса."""
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        """Высота бокса."""
        return self.y2 - self.y1

    def to_yolo(self) -> tuple[float, float, float, float]:
        """Конвертирует в YOLO-формат (относительные координаты cx, cy, w, h).

        Требует image_size. Возвращает (cx, cy, w, h) в диапазоне [0, 1].
        """
        if self.image_size is None:
            raise ValueError("image_size не задан, невозможно конвертировать в YOLO-формат")
        w, h = self.image_size
        cx = self.cx / w
        cy = self.cy / h
        bw = self.width / w
        bh = self.height / h
        return cx, cy, bw, bh


@dataclass
class DetectionResult:
    """Результат детекции одного изображения."""
    boxes: list[BBox]
    signature: str
    image_size: tuple[int, int]  # (width, height)

    @classmethod
    def from_boxes(cls, boxes: list[BBox], image_size: tuple[int, int], names: list[str]) -> "DetectionResult":
        """Собирает DetectionResult из списка боксов.
        Сортирует боксы по центру X и собирает signature из имён классов.
        Если боксов нет — signature пустая строка.
        """
        if not boxes:
            return cls(boxes=[], signature="", image_size=image_size)

        # сортировка по cx
        sorted_boxes = sorted(boxes, key=lambda b: b.cx)

        # сборка signature
        parts = []
        for box in sorted_boxes:
            if box.class_id < 0 or box.class_id >= len(names):
                raise ValueError(f"class_id {box.class_id} вне диапазона [0, {len(names)})")
            parts.append(names[box.class_id])
        signature = "".join(parts)

        return cls(boxes=sorted_boxes, signature=signature, image_size=image_size)