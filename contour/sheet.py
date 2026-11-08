"""Чтение листа (PDF или картинка), выравнивание, формат листа и масштаб в мм."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .config import setting

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
PDF_EXT = {".pdf"}
SUPPORTED_EXT = IMAGE_EXT | PDF_EXT

DEFAULT_DPI = 300


def render_dpi() -> float:
    """CONTOUR_DPI (по умолчанию 300): разрешение, в котором рендерится
    страница PDF. Для картинки оно же принимается, только если в файле нет
    DPI и штамп не найден. Картинки никогда не пересчитываются: берутся их
    настоящие пиксели, а масштаб в мм — по штампу."""
    raw = setting("CONTOUR_DPI", str(DEFAULT_DPI))
    try:
        dpi = float(raw)
    except ValueError:
        raise ValueError(f"CONTOUR_DPI must be a number, got {raw!r}") from None
    if not 72 <= dpi <= 1200:
        raise ValueError(f"CONTOUR_DPI must be 72..1200, got {dpi:g}")
    return dpi

# форматы листов по ГОСТ 2.301: (короткая сторона, длинная сторона) в мм
FORMATS = {
    "A4": (210, 297), "A3": (297, 420), "A2": (420, 594), "A1": (594, 841), "A0": (841, 1189),
    "A4x3": (297, 630), "A4x4": (297, 841), "A4x5": (297, 1051),
    "A3x3": (420, 891), "A3x4": (420, 1189), "A2x3": (594, 1261), "A1x3": (841, 1783),
}

# Основная надпись (штамп) по ГОСТ 2.104 всегда 185 мм в ширину; высота
# зависит от формы: 55 мм (форма 1, первый лист чертежа), 40 мм (форма 2,
# текстовые документы), 15 мм (форма 2а, мелкий штамп последующих листов).
# Формы ищутся в этом порядке: сначала форма 1, чтобы внутренние строки её
# штампа не приняли за штамп поменьше.
STAMP_W_MM = 185.0
STAMP_H_MM = 55.0
STAMP_FORMS = ((55.0, "form 1"), (40.0, "form 2"), (15.0, "form 2a"))


@dataclass
class Sheet:
    """Прочитанный лист и то, что о нём известно."""
    path: Path
    gray: np.ndarray                 # uint8, бумага светлая, линии тёмные
    source: str                      # "image" или "pdf"
    pages: int = 1
    file_dpi: float | None = None    # dpi рендера PDF или из метаданных картинки
    notes: list[str] = field(default_factory=list)
    upscale: int = 1                 # во сколько раз gray увеличен для обработки
    deskew_deg: float = 0.0          # на какой угол gray повёрнут, чтобы стать ровным
    geometry_changed: bool = False   # перспектива изменила масштаб листа: dpi страницы уже не точен


def _imread_unicode(path: Path) -> np.ndarray:
    """Прочитать картинку в оттенках серого (путь может содержать кириллицу)."""
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        raise ValueError("file is empty")
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("image could not be decoded")
    return img


def _image_dpi(path: Path) -> float | None:
    """DPI из метаданных картинки или None (значения 50 и меньше считаются мусором)."""
    try:
        from PIL import Image
        with Image.open(path) as im:
            dpi = im.info.get("dpi")
        if dpi and dpi[0] and float(dpi[0]) > 50:
            return float(dpi[0])
    except Exception:
        pass
    return None


DESKEW_MIN_DEG = 0.1     # меньший наклон не трогаем: по всему листу это доли пикселя
DESKEW_MAX_DEG = 10.0    # больший «наклон» — уже не кривой скан, угадывать не беремся


DESKEW_MODES = ("off", "rotate", "perspective")


def deskew_mode() -> str:
    """CONTOUR_DESKEW: off (по умолчанию) | rotate | perspective; "on" = rotate."""
    v = setting("CONTOUR_DESKEW", "off").lower()
    if v == "on":
        v = "rotate"
    if v not in DESKEW_MODES:
        raise ValueError(f"CONTOUR_DESKEW must be off|rotate|perspective, got {v!r}")
    return v


def _profile_sharpness(xs: np.ndarray, ys: np.ndarray, deg: float, n: int) -> float:
    """Насколько «остры» гистограммы чернил по строкам и столбцам, если
    повернуть лист на deg: горизонтальные и вертикальные линии попадают
    каждая в одну строку (столбец) только при верном угле."""
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    r = np.round(ys * c - xs * s).astype(np.int64)     # строка каждого пикселя после поворота
    k = np.round(xs * c + ys * s).astype(np.int64)     # столбец
    hr = np.bincount(r - r.min(), minlength=1).astype(np.float64)
    hc = np.bincount(k - k.min(), minlength=1).astype(np.float64)
    return float((hr ** 2).sum() + (hc ** 2).sum()) / n ** 2


def skew_angle(gray: np.ndarray) -> tuple[float | None, str]:
    """Наклон листа в градусах (метод проекционного профиля).

    Рамка, штамп, линии таблиц и строки текста на чертеже горизонтальны или
    вертикальны. При повороте пикселей чернил на верный угол каждая такая
    линия ложится в одну строку (или столбец) гистограммы, и гистограмма
    становится самой «острой» (наибольшая сумма квадратов). Перебор: грубо
    ±10° с шагом 0,5°, затем шагом 0,02° вокруг лучшего. Наклонные линии
    детали (конусы, фаски) — малая доля чернил и пик не сдвигают.
    Детерминировано: выборка пикселей с постоянным шагом, без случайности.
    """
    _, t = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ys, xs = np.nonzero(t)
    if len(xs) < 1000:
        return None, "almost no ink"
    step = max(1, len(xs) // 300_000)                  # пикселей достаточно, время ограничено
    xs = xs[::step].astype(np.float64)
    ys = ys[::step].astype(np.float64)
    n = len(xs)

    def best(angles):
        sc = [_profile_sharpness(xs, ys, a, n) for a in angles]
        i = int(np.argmax(sc))
        return float(angles[i]), sc

    coarse, _ = best(np.arange(-DESKEW_MAX_DEG, DESKEW_MAX_DEG + 1e-9, 0.5))
    fine, sc = best(np.round(np.arange(coarse - 0.5, coarse + 0.5 + 1e-9, 0.02), 2))
    # лист без прямых линий даёт плоскую кривую: надёжного угла нет
    contrast = max(sc) / (np.median(sc) + 1e-12)
    if contrast < 1.05:
        return None, f"no dominant straight lines (contrast {contrast:.3f})"
    # знак: тот угол, который нужно передать в cv2.getRotationMatrix2D, чтобы лист стал ровным
    return fine, f"projection profile, contrast {contrast:.2f}"


def deskew(sheet: "Sheet") -> "Sheet":
    """Повернуть наклонный лист ровно (размер картинки тот же, углы заливаются белым)."""
    a, why = skew_angle(sheet.gray)
    if a is None:
        sheet.notes.append(f"deskew: angle not measured ({why}), sheet used as is")
        return sheet
    if abs(a) < DESKEW_MIN_DEG:
        sheet.notes.append(f"deskew: tilt {a:+.2f} deg ({why}), below {DESKEW_MIN_DEG} deg, not rotated")
        return sheet
    h, w = sheet.gray.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), a, 1.0)
    sheet.gray = cv2.warpAffine(sheet.gray, M, (w, h), flags=cv2.INTER_CUBIC,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    sheet.deskew_deg = a
    sheet.notes.append(f"deskew: sheet tilted {a:+.2f} deg ({why}), rotated upright; "
                       "contour coordinates are in the upright sheet")
    return sheet


PERSPECTIVE_MIN_PX = 2.0     # углы рамки ближе этого к ровному прямоугольнику: лист не трогаем
FRAME_SIDE_SUPPORT = 0.6     # у стороны рамки не меньше этой доли голосов самой длинной линии своего направления
FRAME_MAX_TILT_DEG = 20.0    # стороны рамки отклоняются от осей не больше чем на столько (фото под углом)


def _frame_component(t: np.ndarray) -> tuple[np.ndarray | None, str]:
    """Маска рамки со всем, что к ней присоединено (штамп, графы): связный
    кусок чернил с наибольшим числом пикселей, занимающий не меньше половины
    картинки по обеим осям."""
    h, w = t.shape
    n, lab, st, _ = cv2.connectedComponentsWithStats(t, connectivity=8)
    best = None
    for i in range(1, n):
        x, y, bw, bh, area = st[i]
        if bw >= 0.5 * w and bh >= 0.5 * h and (best is None or area > st[best][4]):
            best = i
    if best is None:
        return None, "no frame-sized piece of ink"
    x, y, bw, bh, _ = st[best]
    if x <= 0 or y <= 0 or x + bw >= w or y + bh >= h:
        return None, "frame touches the image edge (cut off)"
    return (lab == best).astype(np.uint8), "ok"


def _side_lines(mask: np.ndarray, horizontal: bool) -> list[tuple[float, float, int]]:
    """Сильные прямые одного направления в маске: (rho, theta, голоса) по Хафу."""
    tilt = np.radians(FRAME_MAX_TILT_DEG)
    step = np.pi / 1800                                   # 0,1°
    thr = int(0.3 * min(mask.shape))
    ranges = [(np.pi / 2 - tilt, np.pi / 2 + tilt)] if horizontal else [(0.0, tilt), (np.pi - tilt, np.pi)]
    out = []
    for lo, hi in ranges:
        res = cv2.HoughLinesWithAccumulator(mask, 1, step, thr, min_theta=lo, max_theta=hi)
        if res is not None:
            out += [(float(r), float(t), int(v)) for r, t, v in res.reshape(-1, 3)]
    return out


def _refine_side(pts: np.ndarray, line: tuple[float, float]) -> tuple[float, float] | None:
    """Средняя линия штриха рамки около прямой Хафа, методом наименьших квадратов.

    У толстого штриха прямая Хафа «гуляет» внутри него на доли градуса;
    прямая, проведённая через все пиксели штриха рядом с ней, даёт его
    среднюю линию с точностью до долей пикселя. Два прохода с сужающейся
    полосой; веса Huber ослабляют влияние коротких поперечных линий
    (столбцы штампа), касающихся стороны."""
    r, th = line
    for band in (15.0, 8.0):
        n = np.array([np.cos(th), np.sin(th)])
        near = pts[np.abs(pts @ n - r) <= band]
        if len(near) < 50:
            return None
        vx, vy, x0, y0 = cv2.fitLine(near.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        n = np.array([-vy, vx], dtype=np.float64)
        th = float(np.arctan2(n[1], n[0]))
        r = float(n @ np.array([x0, y0]))
    return r, th


def _intersect(a, b) -> np.ndarray:
    """Точка пересечения двух прямых, заданных как (rho, theta)."""
    (r1, t1), (r2, t2) = a, b
    A = np.array([[np.cos(t1), np.sin(t1)], [np.cos(t2), np.sin(t2)]])
    return np.linalg.solve(A, np.array([r1, r2]))


def frame_quad(gray: np.ndarray) -> tuple[np.ndarray | None, str]:
    """Четыре угла рамки (левый верхний, правый верхний, правый нижний, левый нижний) в пикселях.

    Перспектива сохраняет прямые, поэтому стороны рамки остаются самыми
    длинными прямыми куска рамки. Прямые ищутся преобразованием Хафа в двух
    группах (почти горизонтальные и почти вертикальные); в каждой группе
    учитываются только прямые с не меньше чем 60% голосов самой сильной, и
    две крайние из них — стороны рамки. Короткие линии, примыкающие к рамке,
    отсекаются: дополнительные графы на левом поле (примерно в половину
    высоты рамки), верхняя линия штампа. Углы — пересечения сторон; они
    лежат на внешнем краю линии рамки (маска — во всю толщину линии).
    """
    _, t = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask, why = _frame_component(t)
    if mask is None:
        return None, why
    h, w = mask.shape
    sides = []
    for horizontal in (True, False):
        lines = _side_lines(mask, horizontal)
        if not lines:
            return None, f"no long {'horizontal' if horizontal else 'vertical'} frame lines"
        top = max(v for _, _, v in lines)
        strong = [(r, th) for r, th, v in lines if v >= FRAME_SIDE_SUPPORT * top]

        # положение каждой прямой поперёк листа: y посередине ширины или x посередине высоты
        def pos(l):
            r, th = l
            return (r - (w / 2) * np.cos(th)) / np.sin(th) if horizontal else (r - (h / 2) * np.sin(th)) / np.cos(th)
        strong.sort(key=pos)
        if len(strong) < 2 or pos(strong[-1]) - pos(strong[0]) < 0.5 * (h if horizontal else w):
            return None, f"only one {'horizontal' if horizontal else 'vertical'} frame side found"
        sides.append((strong[0], strong[-1]))
    ys, xs = np.nonzero(mask)
    pts = np.column_stack([xs, ys]).astype(np.float64)
    refined = []
    for pair in sides:
        for line in pair:
            r = _refine_side(pts, line)
            if r is None:
                return None, "frame side could not be fitted"
            refined.append(r)
    top_l, bot_l, left_l, right_l = refined
    q = np.array([_intersect(top_l, left_l), _intersect(top_l, right_l),
                  _intersect(bot_l, right_l), _intersect(bot_l, left_l)])
    if not (np.all(q[:, 0] > -0.05 * w) and np.all(q[:, 0] < 1.05 * w)
            and np.all(q[:, 1] > -0.05 * h) and np.all(q[:, 1] < 1.05 * h)):
        return None, "frame corners fall outside the image"
    return q, "ok"


def _stamp_stretch(gray: np.ndarray) -> tuple[float | None, str]:
    """Растяжение по высоте, при котором штамп снова получает размер по ГОСТ.

    Четыре угла задают форму рамки с точностью до соотношения сторон (оно при
    перспективе теряется). Штамп — прямоугольник с известным соотношением
    (185:55, 185:40 или 185:15 по форме), поэтому после первой правки его
    измеренное соотношение показывает, насколько лист ещё растянут по вертикали.
    """
    from .extract import measure_stamp      # импорт здесь: extract сам импортирует этот модуль
    _, t = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask, why = _frame_component(t)
    if mask is None:
        return None, f"frame lost after correction ({why})"
    stamp = measure_stamp(mask)
    if stamp is None:
        return None, "title block not measured"
    k = (stamp["w_px"] / STAMP_W_MM) / (stamp["h_px"] / stamp["h_mm"])
    return k, f"title block {stamp['w_px']:.0f}x{stamp['h_px']:.0f} px -> height x{k:.4f}"


def unwarp(sheet: "Sheet") -> "Sheet":
    """Убрать перспективу (фото под углом) и поворот.

    Шаг 1: четыре угла рамки переносятся на ровный прямоугольник со
    средними длинами сторон, с центром там же, где была рамка. Шаг 2:
    соотношение сторон штампа уточняет высоту этого прямоугольника; затем
    исходная картинка перерисовывается один раз по окончательным углам, так
    что пересчитывается только однажды. Изогнутая или мятая бумага не
    исправляется (стороны рамки тогда не прямые).
    """
    q, why = frame_quad(sheet.gray)
    if q is None:
        sheet.notes.append(f"perspective: frame corners not found ({why}), sheet used as is")
        return sheet
    tl, tr, br, bl = q
    # рамка уже ровный прямоугольник (экспорт из CAD, ровный скан): не перерисовываем
    off_axis = max(abs(tl[0] - bl[0]), abs(tr[0] - br[0]), abs(tl[1] - tr[1]), abs(bl[1] - br[1]))
    if off_axis < PERSPECTIVE_MIN_PX:
        sheet.notes.append(f"perspective: frame is already an upright rectangle "
                           f"(corners off axis by {off_axis:.1f} px), not changed")
        return sheet
    width = (np.hypot(*(tr - tl)) + np.hypot(*(br - bl))) / 2
    height = (np.hypot(*(bl - tl)) + np.hypot(*(br - tr))) / 2
    h, w = sheet.gray.shape
    src = q.astype(np.float32)

    def target(W, H):
        cx = float(np.clip(q[:, 0].mean(), W / 2, w - W / 2)) if W < w else w / 2
        cy = float(np.clip(q[:, 1].mean(), H / 2, h - H / 2)) if H < h else h / 2
        return np.array([[cx - W / 2, cy - H / 2], [cx + W / 2, cy - H / 2],
                         [cx + W / 2, cy + H / 2], [cx - W / 2, cy + H / 2]], np.float32)

    def warp(dst):
        M = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(sheet.gray, M, (w, h), flags=cv2.INTER_CUBIC,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=255)

    first = warp(target(width, height))
    k, how = _stamp_stretch(first)
    if k is None or not 0.8 <= k <= 1.25:
        sheet.gray = first
        W, H = width, height
        how = f"aspect from the frame corners only, mm may be off by a few % ({how})"
    else:
        W, H = width, height * k
        sheet.gray = warp(target(W, H))
    sheet.geometry_changed = True
    c = ", ".join(f"({x:.0f},{y:.0f})" for x, y in q)
    sheet.notes.append(f"perspective: frame corners {c} mapped onto an upright {W:.0f}x{H:.0f} px rectangle; "
                       f"{how}; contour coordinates are in the corrected sheet")
    return sheet


def pdf_page_count(path: Path) -> int:
    """Число страниц PDF; 1 для всего остального и для PDF, который не
    открывается (ошибка тогда проявится при чтении листа, как red)."""
    if path.suffix.lower() not in PDF_EXT:
        return 1
    try:
        import pymupdf
        with pymupdf.open(str(path)) as doc:
            return max(1, doc.page_count)
    except Exception:
        return 1


def load_sheet(path: Path, page: int = 0) -> Sheet:
    """Прочитать лист и при необходимости выровнять (CONTOUR_DESKEW).
    page — страница PDF с 0 (каждая страница PDF — отдельный лист)."""
    sheet = _read(path, page)
    m = deskew_mode()
    if m == "rotate":
        deskew(sheet)
    elif m == "perspective":
        unwarp(sheet)
    else:
        sheet.notes.append("deskew: off (CONTOUR_DESKEW=off)")
    return sheet


def _read(path: Path, page: int = 0) -> Sheet:
    """PDF: страница рендерится в оттенках серого в CONTOUR_DPI.
    Картинка: читается как есть, DPI берётся из метаданных файла."""
    ext = path.suffix.lower()
    if ext in PDF_EXT:
        import pymupdf
        with pymupdf.open(str(path)) as doc:
            if doc.page_count == 0:
                raise ValueError("PDF has no pages")
            if not 0 <= page < doc.page_count:
                raise ValueError(f"PDF has {doc.page_count} pages, page {page + 1} requested")
            dpi = render_dpi()
            pix = doc[page].get_pixmap(dpi=int(round(dpi)), colorspace=pymupdf.csGRAY, alpha=False)
            gray = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width).copy()
            sh = Sheet(path, gray, "pdf", pages=doc.page_count, file_dpi=float(int(round(dpi))))
            if doc.page_count > 1:
                sh.notes.append(f"PDF page {page + 1} of {doc.page_count} (each page is a separate sheet)")
            return sh
    if page:
        raise ValueError(f"{path.name} is not a PDF, it has no page {page + 1}")
    gray = _imread_unicode(path)
    return Sheet(path, gray, "image", file_dpi=_image_dpi(path))


@dataclass
class Scale:
    fmt: str
    orientation: str        # "portrait" / "landscape"
    px_per_mm: float
    method: str             # как определён масштаб (пишется в log.txt)
    candidates: list[str] = field(default_factory=list)


def _format_candidates(h: int, w: int, tol: float = 0.03):
    """Форматы, чьи пропорции совпадают с картинкой (в обеих ориентациях)."""
    out = []
    for name, (a, b) in FORMATS.items():
        for orient, (W, H) in (("portrait", (a, b)), ("landscape", (b, a))):
            sx, sy = w / W, h / H
            err = abs(sx - sy) / ((sx + sy) / 2)
            if err <= tol:
                out.append((err, name, orient, (sx + sy) / 2))
    out.sort()
    return out


def decide_scale(sheet: Sheet, stamp_w_px: float | None) -> Scale:
    """Выбрать формат листа и число пикселей на мм.

    PDF: по размеру страницы (точно). Картинка: по ширине штампа (185 мм по
    ГОСТ) -> по DPI из файла -> по CONTOUR_DPI. Пропорции картинки всегда
    ограничивают выбор формата.
    """
    h, w = sheet.gray.shape
    cands = _format_candidates(h, w)
    names = [f"{n}-{o}" for _, n, o, _ in cands]
    if sheet.source == "pdf" and sheet.file_dpi and not sheet.geometry_changed:
        # у страницы PDF точный физический размер, и при известном dpi рендера
        # масштаб точен. Формат ГОСТ здесь только метка: подгонка под него
        # исказила бы нестандартную страницу (250 мм превратились бы в 297 мм).
        ppm = sheet.file_dpi / 25.4
        near = sorted((abs(np.log(c[3] / ppm)), c) for c in cands)
        fmt, orient = (near[0][1][1], near[0][1][2]) if near and near[0][0] < 0.02 else \
            ("unknown", "landscape" if w > h else "portrait")
        how = (f"PDF page size, rendered at {sheet.file_dpi / sheet.upscale:g} dpi "
               f"(page {w / ppm:.1f} x {h / ppm:.1f} mm)")
        if stamp_w_px:
            how += f"; title block {stamp_w_px / ppm:.1f} mm wide (GOST 185)"
        return Scale(fmt, orient, ppm, how, names)
    if not cands:
        # пропорции картинки не совпали ни с одним форматом ГОСТ (скан с полями,
        # обрезанный лист): масштаб всё равно даёт штамп
        if stamp_w_px:
            return Scale("unknown", "landscape" if w > h else "portrait", stamp_w_px / STAMP_W_MM,
                         f"title block width {stamp_w_px:.0f}px = 185 mm (image aspect {w}x{h} matches no GOST format)",
                         names)
        dpi = sheet.file_dpi or render_dpi() * sheet.upscale
        return Scale("unknown", "landscape" if w > h else "portrait", dpi / 25.4,
                     f"no GOST format matches aspect {w}x{h}; assumed {dpi:.0f} dpi", names)

    def pick(target_ppm: float, how: str) -> Scale:
        best = min(cands, key=lambda c: abs(np.log(c[3] / target_ppm)))
        return Scale(best[1], best[2], best[3], how, names)

    if stamp_w_px:
        s = pick(stamp_w_px / STAMP_W_MM, f"title block width {stamp_w_px:.0f}px = 185 mm")
        # принимается, только если масштаб по штампу и по выбранному формату расходятся меньше чем на 6%
        if abs(np.log(s.px_per_mm / (stamp_w_px / STAMP_W_MM))) < 0.06:
            return s
    if sheet.file_dpi:
        return pick(sheet.file_dpi / 25.4, f"file dpi {sheet.file_dpi / sheet.upscale:.0f}")
    dpi = render_dpi()
    return pick(dpi * sheet.upscale / 25.4, f"assumed {dpi:g} dpi (CONTOUR_DPI)")
