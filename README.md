

# Детекция символов номерных знаков

Проект детектирует отдельные символы номерных знаков с помощью YOLOv8 ONNX и собирает распознанную строку слева направо.



## Обзор архитектуры

```text
Изображение
  -> letterbox до 640 px
  -> YOLOv8 ONNX (models/best.onnx)
  -> confidence-фильтрация и NMS
  -> сортировка боксов по горизонтали
  -> строка номера + визуализация
```

- plates_detect.py — CLI-точка входа.
- detector — ONNX-инференс, постпроцессинг, конфигурация и отрисовка.
- config.yaml — модель, пороги, размер входа, устройство и параметры вывода.
- dataset_prepare.py, dataset_prepare — очистка датасета, аудит разметки и формирование сплитов.
- _dataset — данные в формате YOLO.
- best.onnx — модель с 27 классами символов.
- predictions — сохранённые результаты инференса.

## Запуск

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python3 plates_detect.py --image _dataset/test/example.jpeg --no-show
python3 plates_detect.py --dir _dataset/test --no-show --quiet
```

Настройки задаются в config.yaml.

## Подготовка датасета

Конвейер: дедупликация → аудит файлов → проверка боксов → упаковка для CVAT → перераспределение сплитов.

```bash
python3 dataset_prepare.py hash-find --root ./_dataset
python3 dataset_prepare.py files-find --root ./_dataset
python3 dataset_prepare.py boxes-stats --root ./_dataset
python3 dataset_prepare.py boxes-check --root ./_dataset
python3 dataset_prepare.py cvat-pack --root ./_dataset
python3 dataset_prepare.py split-plan --root ./_dataset
```

Отчеты и промежуточные артефакты сохраняются в audit_out.

Подробный разбор датасета: ./dataset_prepare/dataset_report.md
Обучение и графики https://colab.research.google.com/drive/1b0kl3Cqx0RCN_BOG7MouCJ1pYsL0u-Ea?usp=sharing
