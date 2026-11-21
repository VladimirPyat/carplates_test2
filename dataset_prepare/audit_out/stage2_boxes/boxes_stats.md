# boxes_stats

- images: 530
- instances: 3530
- classes_source: voted
- nc: 27
- classes_conflicts: 0
- coverage: {'both': 530}
- objects_per_image_min: 1
- objects_per_image_p50: 7.0
- objects_per_image_max: 8
- zero_object_images: 0
- imbalance_ratio: 7.59
- side_px_p50: 15.0
- ar_p95: 5.38
- difficult: 0
- truncated: 0
- parse_flags: 0

## Осознанно пропущено

- Исправление боксов/классов — вручную в CVAT, скрипт только флагует
- GEOMETRY_DRIFT/FORMAT_UNMATCHED — системный сдвиг ~1px между форматами, только агрегат в отчёте
- obj.names — генерируется на этапе упаковки под CVAT из classes.json
- Мелкие боксы не удаляем: удаление бокса при живом объекте = неразмеченный позитив
- Решение по imgsz / тайлингу / P2 — после статистики размеров
