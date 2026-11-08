import sys
from pathlib import Path

# корень репозитория (пакет contour) и папка тестов (synth.py) в путях импорта
HERE = Path(__file__).resolve().parent
for p in (HERE.parent, HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
