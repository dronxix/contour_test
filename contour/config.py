"""Настройки запуска из файла .env.

Строки файла имеют вид КЛЮЧ=ЗНАЧЕНИЕ; пустые строки и строки, начинающиеся
с #, пропускаются; значение можно взять в одинарные или двойные кавычки.
Переменная, уже заданная в окружении, важнее файла: так одну настройку можно
поменять на один запуск, не правя файл.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV = ROOT / ".env"


def load_env(path: Path | None) -> list[str]:
    """Загрузить файл в os.environ; вернуть ключи, взятые из файла."""
    if path is None:
        path = DEFAULT_ENV
        if not path.exists():
            return []
    if not path.exists():
        raise FileNotFoundError(f"env file not found: {path}")
    taken = []
    for n, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"{path}:{n}: expected KEY=VALUE, got {raw!r}")
        key, val = (s.strip() for s in line.split("=", 1))
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        elif val.startswith("#"):
            val = ""                                    # КЛЮЧ=   # комментарий  (пустое значение)
        elif " #" in val:
            val = val.split(" #", 1)[0].rstrip()       # КЛЮЧ=значение   # комментарий
        if key not in os.environ:
            os.environ[key] = val
            taken.append(key)
    return taken


def setting(key: str, default: str = "") -> str:
    """Значение настройки; пустое значение (КЛЮЧ= в файле) означает значение по умолчанию."""
    return os.environ.get(key, "").strip() or default


def int_setting(key: str, default: int, lo: int, hi: int) -> int:
    """Целая настройка в пределах lo..hi; иначе понятная ошибка до начала обработки."""
    raw = setting(key, str(default))
    try:
        v = int(raw)
    except ValueError:
        raise ValueError(f"{key} must be a whole number, got {raw!r}") from None
    if not lo <= v <= hi:
        raise ValueError(f"{key} must be {lo}..{hi}, got {v}")
    return v


def float_setting(key: str, default: float, lo: float, hi: float) -> float:
    """Дробная настройка в пределах lo..hi; иначе понятная ошибка до начала обработки."""
    raw = setting(key, str(default))
    try:
        v = float(raw)
    except ValueError:
        raise ValueError(f"{key} must be a number, got {raw!r}") from None
    if not lo <= v <= hi:
        raise ValueError(f"{key} must be {lo:g}..{hi:g}, got {v:g}")
    return v


def resolve(path: str) -> Path:
    """Относительные пути в настройках считаются от папки проекта."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p
