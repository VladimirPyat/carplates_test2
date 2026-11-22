"""Загрузка и валидация конфига."""
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class DetectorConfig:
    model: Path
    classes: Path
    imgsz: int = 640
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    device: str = "cpu"
    # Вывод в терминал
    verbose: bool = True
    # Визуализация
    output_dir: Path = Path("./predictions")
    show: bool = True
    save: bool = True
    save_txt: bool = True
    show_boxes: bool = True
    show_labels: bool = True
    show_signature: bool = True
    min_confidence_label: float = 0.5

    @classmethod
    def from_yaml(cls, path: Path) -> "DetectorConfig":
        """Загружает конфиг, резолвит относительные пути относительно файла конфига."""
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        data = {k: v for k, v in data.items() if v is not None}

        # преобразуем пути: если относительные — относительно конфига, если абсолютные — как есть
        config_dir = path.resolve().parent
        for key in ("model", "classes", "output_dir"):
            if key in data:
                p = Path(data[key])
                if not p.is_absolute():
                    p = config_dir / p
                data[key] = p

        return cls(**data)

    def load_names(self) -> list[str]:
        import json
        with open(self.classes, encoding="utf-8") as f:
            data = json.load(f)
        return [str(n).strip() for n in data.get("names", [])]