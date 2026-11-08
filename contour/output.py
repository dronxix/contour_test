"""Картинки и contour.json для каждого листа: overlay.png, preview.png,
крупные планы видов, contour.json (verdict.txt и log.txt пишет pipeline.py)."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

CONTOUR_BGR = (255, 0, 200)      # малиновый: хорошо виден и на чёрном, и на белом
DROPPED_BGR = (0, 170, 255)      # оранжевый для выброшенного; в overlay не рисуется (по ТЗ там только контур)


def write_overlay(path: Path, gray: np.ndarray, segs_px, dropped_px, line_px: int) -> None:
    """overlay.png: осветлённый чертёж и контур поверх.

    dropped_px не рисуется: по ТЗ на картинке только чертёж и контур, а что
    выброшено и почему, перечислено в log.txt.
    """
    base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    # чертёж осветляется, чтобы контур выделялся
    base = (255 - (255 - base.astype(np.int16)) * 0.45).clip(0, 255).astype(np.uint8)
    for x1, y1, x2, y2 in segs_px:
        cv2.line(base, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))),
                 CONTOUR_BGR, line_px, cv2.LINE_AA)
    ok, buf = cv2.imencode(".png", base, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    if not ok:
        raise IOError("png encode failed")
    buf.tofile(str(path))


def write_failure_image(path: Path, message: str) -> None:
    """overlay.png для листа, который не удалось даже прочитать: картинка с
    надписью об этом (шрифты OpenCV только латинские, текст короткий)."""
    img = np.full((240, 1200, 3), 255, np.uint8)
    cv2.rectangle(img, (4, 4), (1195, 235), (0, 0, 220), 4)
    cv2.putText(img, "SHEET NOT PROCESSED - no contour", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 220), 3)
    text = message.encode("ascii", "replace").decode()
    for i in range(0, min(len(text), 240), 80):
        cv2.putText(img, text[i:i + 80], (30, 140 + 35 * (i // 80)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    ok, buf = cv2.imencode(".png", img)
    buf.tofile(str(path))


def write_preview(path: Path, gray: np.ndarray, segs_px, max_side: int = 1024, line_px: int = 3) -> None:
    """Картинка всего листа для модели. Сначала лист уменьшается, потом поверх
    рисуется контур: так он остаётся ярким (если уменьшить готовый overlay,
    тонкие малиновые линии превращаются в бледно-серые)."""
    h, w = gray.shape[:2]
    s = min(1.0, max_side / max(h, w))
    small = cv2.resize(gray, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else gray
    img = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
    img = (255 - (255 - img.astype(np.int16)) * 0.55).clip(0, 255).astype(np.uint8)
    for x1, y1, x2, y2 in segs_px:
        cv2.line(img, (int(round(x1 * s)), int(round(y1 * s))), (int(round(x2 * s)), int(round(y2 * s))),
                 CONTOUR_BGR, line_px, cv2.LINE_AA)
    ok, buf = cv2.imencode(".png", img)
    buf.tofile(str(path))


def view_boxes(segs_px, shape, ppm: float, gap_mm: float = 5.0, pad_mm: float = 4.0,
               min_mm: float = 30.0) -> list[tuple[int, int, int, int]]:
    """Рамки (x0, y0, x1, y1 в пикселях) вокруг групп линий контура — видов детали.

    Линии контура ближе gap_mm друг к другу — одна группа. К рамке
    добавляются поля pad_mm, чтобы были видны размеры и стрелки рядом с
    видом, и она не меньше min_mm по каждой стороне, чтобы у маленького
    выносного вида было видно окружение. Порядок рамок постоянный: сверху
    вниз, слева направо.
    """
    h, w = shape[:2]
    q = max(1, int(round(ppm / 2)))                 # ячейки примерно 0,5 мм
    m = np.zeros((h // q + 1, w // q + 1), np.uint8)
    for x1, y1, x2, y2 in segs_px:
        cv2.line(m, (int(x1 / q), int(y1 / q)), (int(x2 / q), int(y2 / q)), 1, 1)
    k = max(3, int(round(gap_mm * ppm / q)) | 1)
    m = cv2.dilate(m, np.ones((k, k), np.uint8))
    n, _, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    pad, mn = pad_mm * ppm, min_mm * ppm
    boxes = []
    for i in range(1, n):
        x, y, bw, bh, _ = st[i]
        # убрать расширение, потом добавить поля
        cx0, cy0 = x * q + k * q / 2, y * q + k * q / 2
        cx1, cy1 = (x + bw) * q - k * q / 2, (y + bh) * q - k * q / 2
        cx0, cy0, cx1, cy1 = cx0 - pad, cy0 - pad, cx1 + pad, cy1 + pad
        if cx1 - cx0 < mn:
            c = (cx0 + cx1) / 2
            cx0, cx1 = c - mn / 2, c + mn / 2
        if cy1 - cy0 < mn:
            c = (cy0 + cy1) / 2
            cy0, cy1 = c - mn / 2, c + mn / 2
        boxes.append((max(0, int(cx0)), max(0, int(cy0)), min(w, int(cx1)), min(h, int(cy1))))
    return sorted(boxes, key=lambda b: (b[1] // int(20 * ppm), b[0]))


def write_zoom(path: Path, gray: np.ndarray, segs_px, box, max_side: int = 1024, line_px: int = 3) -> None:
    """Крупный план одного вида: вырезка в полном разрешении (или уменьшенная
    до max_side), контур рисуется поверх после масштабирования."""
    x0, y0, x1, y1 = box
    crop = gray[y0:y1, x0:x1]
    s = min(1.0, max_side / max(crop.shape))
    small = cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else crop
    img = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
    img = (255 - (255 - img.astype(np.int16)) * 0.55).clip(0, 255).astype(np.uint8)
    for a, b, c, d in segs_px:
        cv2.line(img, (int(round((a - x0) * s)), int(round((b - y0) * s))),
                 (int(round((c - x0) * s)), int(round((d - y0) * s))), CONTOUR_BGR, line_px, cv2.LINE_AA)
    ok, buf = cv2.imencode(".png", img)
    buf.tofile(str(path))


def contour_hash(segs: list[dict]) -> str:
    """Отпечаток (SHA-1) координат контура: привязывает вердикт ревью именно к этому контуру."""
    import hashlib
    return hashlib.sha1(json.dumps(segs, sort_keys=True).encode()).hexdigest()


def write_contour_json(path: Path, source: str, segs_px, px_per_mm: float, metrics: dict) -> str:
    """contour.json: отрезки в мм (до 0,001 мм); возвращает отпечаток контура."""
    segs = [
        {"x1": round(x1 / px_per_mm, 3), "y1": round(y1 / px_per_mm, 3),
         "x2": round(x2 / px_per_mm, 3), "y2": round(y2 / px_per_mm, 3)}
        for x1, y1, x2, y2 in segs_px
    ]
    data = {
        "source": source,
        "units": "mm",
        "origin": "top-left corner of the sheet image, x to the right, y down",
        "sheet_format": metrics.get("format"),
        "px_per_mm": metrics.get("px_per_mm"),
        "segment_count": len(segs),
        "segments": segs,
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    return contour_hash(segs)


def write_empty_contour(path: Path, source: str, error: str) -> None:
    """Пустой contour.json для неразобранного листа, с текстом ошибки."""
    data = {"source": source, "units": "mm", "segment_count": 0, "segments": [], "error": error}
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
