"""Синтетический лист A4 для тестов: рамка, штамп, деталь, размерная линия.

Толщины как в CAD при 300 dpi: основная линия 0,7 мм (8 px), тонкая 0,25 мм
(3 px). Деталь — прямоугольник 80 x 40 мм с отверстием, левый верхний угол
в (60, 80) мм от угла листа.
"""
from pathlib import Path

import cv2
import numpy as np

DPI = 300
PPM = DPI / 25.4
A4 = (210, 297)
PART = (60, 80, 140, 120)          # x0, y0, x1, y1 в мм


def px(mm: float) -> int:
    return int(round(mm * PPM))


def draw_sheet(stamp_h_mm: float = 55, part: bool = True) -> np.ndarray:
    w, h = px(A4[0]), px(A4[1])
    img = np.full((h, w), 255, np.uint8)
    thick, thin = 8, 3
    # рамка по ГОСТ 2.301: слева 20 мм, остальные 5 мм
    x0, y0, x1, y1 = px(20), px(5), w - px(5), h - px(5)
    cv2.rectangle(img, (x0, y0), (x1, y1), 0, thick)
    # штамп 185 мм в правом нижнем углу рамки, с внутренними графами
    sx = x1 - px(185)
    sy = y1 - px(stamp_h_mm)
    cv2.rectangle(img, (sx, sy), (x1, y1), 0, thick)
    for k in range(1, int(stamp_h_mm // 5)):
        y = sy + px(5 * k)
        cv2.line(img, (sx, y), (sx + px(65), y), 0, thin)
    cv2.line(img, (sx + px(65), sy), (sx + px(65), y1), 0, thick)
    if part:
        a, b, c, d = (px(v) for v in PART)
        cv2.rectangle(img, (a, b), (c, d), 0, thick)
        cv2.circle(img, (px(100), px(100)), px(8), 0, thick)
        # размерная линия тонкая, под деталью
        cv2.line(img, (a, d + px(12)), (c, d + px(12)), 0, thin)
        cv2.line(img, (a, d + px(3)), (a, d + px(15)), 0, thin)
        cv2.line(img, (c, d + px(3)), (c, d + px(15)), 0, thin)
        # осевая штрихпунктирная
        for x in range(a - px(5), c + px(5), px(10)):
            cv2.line(img, (x, px(100)), (x + px(6), px(100)), 0, thin)
    return img


def save(img: np.ndarray, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(path.suffix, img)[1].tofile(str(path))
    return path
