"""CLI-детектор номерных знаков.

Примеры:
    python3 plates_detect.py --image _dataset/test/ckicup6z1000g246a7fphj2ry.jpeg
    python3 plates_detect.py --dir _dataset/test/ --no-show

"""
from detector.cli import main

if __name__ == "__main__":
    main()