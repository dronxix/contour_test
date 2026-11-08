"""Построение контура: сплошные основные (толстые) линии детали без рамки,
штампа, размеров, осевых, штриховки и текста.

Идея: по ГОСТ 2.303 основная линия в 2–3 раза толще любой тонкой
(размерной, выносной, осевой, штриховки). Полуширины линий меряются на
самом листе, остаются только толстые штрихи, рамка удаляется вместе со
всем, что к ней присоединено (штамп, графы), оставшееся переводится в
отрезки, и выбрасывается толстое, что не является контуром (наконечники
стрелок, жирный текст, знак шероховатости, обозначения разрезов).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .sheet import STAMP_FORMS, STAMP_W_MM, Scale, Sheet, decide_scale
from .vectorize import Edge, Graph, polyline_segments, simplify, skeleton_graph

# --- пороги (в миллиметрах листа, если не сказано иначе) --------------------
SPUR_MM = 3.0            # висящая ветка короче этого — заусенец
ARROW_MAX_MM = 8.0       # висящая ветка короче этого и шире основной линии — наконечник стрелки
BIG_LEN_MM = 40.0        # кусок с меньшей общей длиной линий — «мелкий» (у букв и надписей ~10–30 мм)
BIG_SIZE_MM = 15.0       # ... или с меньшим габаритом
INSIDE_PAD_MM = 2.0      # щели до этого размера заклеиваются при построении силуэта вида
SMALL_OPEN_KEEP_MM = 15.0  # незамкнутый мелкий кусок внутри вида остаётся, если он не короче этого
RINGS_MIN_MM = 10.0      # замкнутая фигура (петля есть, хвостов нет) не меньше этого — элемент детали, не текст
LETTER_MAX_MM = 8.0     # петля с хвостами внутри вида не больше этого — буква
                         # (буквы на чертеже 5–7 мм; отверстие с прилипшей стрелкой больше)
TINY_LEN_MM = 1.5        # висящие цепочки короче этого удаляются сразу (шум скелета)
CORNER_W_MM, CORNER_H_MM = 110.0, 35.0  # зона знака шероховатости, правый верхний угол поля
SIMPLIFY_MM = 0.12       # допуск упрощения Дугласа–Пекера


@dataclass
class Component:
    """Связный кусок графа: его рёбра, длина, габарит, петли, хвосты и решение."""
    edges: list[int]
    length_px: float
    bbox: tuple[float, float, float, float]   # x0, y0, x1, y1 (пиксели)
    cycles: int
    endpoints: int
    keep: bool = True
    reason: str = ""


@dataclass
class Result:
    """Результат листа: масштаб, отрезки контура в пикселях, метрики и лог."""
    scale: Scale
    segments_px: list[tuple[float, float, float, float]]
    metrics: dict
    log: list[str] = field(default_factory=list)
    frame_mask: np.ndarray | None = None
    field_bbox: tuple[int, int, int, int] | None = None
    dropped_segments_px: list[tuple[float, float, float, float]] = field(default_factory=list)


# ---------------------------------------------------------------------------
def binarize(gray: np.ndarray) -> tuple[np.ndarray, float]:
    """Чернила/бумага по порогу Оцу (порог ограничен 100–200, чтобы почти пустой лист не дал абсурд)."""
    t, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    t = float(np.clip(t, 100, 200))
    return (gray < t).astype(np.uint8), t


def line_weights(ink: np.ndarray) -> dict:
    """Разделить полуширины штрихов (измеренные на скелете) на тонкие и
    толстые критерием Оцу по их гистограмме."""
    dt = cv2.distanceTransform(ink, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    sk = cv2.ximgproc.thinning(ink * 255) > 0
    v = dt[sk]
    edges = np.arange(0.5, 16.01, 0.25)
    h, e = np.histogram(v, bins=edges)
    c = e[:-1]
    p = h / max(h.sum(), 1)
    best, k_best = -1.0, 1
    for k in range(1, len(p)):
        w0, w1 = p[:k].sum(), p[k:].sum()
        if w0 < 0.05 or w1 < 0.05:
            continue
        m0 = (p[:k] * c[:k]).sum() / w0
        m1 = (p[k:] * c[k:]).sum() / w1
        var = w0 * w1 * (m1 - m0) ** 2
        if var > best:
            best, k_best = var, k
    lo, hi = p[:k_best], p[k_best:]
    thin = float(c[:k_best][int(np.argmax(lo))])
    thick = float(c[k_best:][int(np.argmax(hi))])
    thr = float(e[k_best])  # левая граница первого «толстого» столбца гистограммы
    separable = thick - thin >= 0.9 and thick / max(thin, 1e-6) >= 1.3
    return {"dt": dt, "thin_hw": thin, "thick_hw": thick, "thr": thr,
            "thick_share": round(float(hi.sum()), 3), "separable": separable,
            "hist": {f"{c[i]:.2f}": round(float(p[i]), 3) for i in range(len(p)) if p[i] >= 0.02}}


def thick_mask(ink: np.ndarray, dt: np.ndarray, thr: float, thick_hw: float) -> np.ndarray:
    """Маска толстых линий: ядра (полуширина не меньше порога), расширенные до полной толщины и обрезанные по чернилам."""
    core = (dt >= thr).astype(np.uint8)
    r = int(np.ceil(thick_hw)) + 1
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.dilate(core, k) & ink


# ---------------------------------------------------------------------------
def frame_rect_from_lines(ink: np.ndarray):
    """Прямоугольники рамки по крайним длинным горизонтальным линиям.

    Линии рамки бывают разной толщины, поэтому учитывается только длина:
    верх и низ рамки — крайние горизонтальные линии длиннее половины
    ширины листа, их концы дают левую и правую стороны. Возвращает
    вложенные прямоугольники, начиная с внешнего.
    """
    h, w = ink.shape
    k = max(25, int(0.5 * w))
    soft = cv2.dilate(ink, np.ones((3, 1), np.uint8))           # терпим дрожание линии на 1 пиксель
    long_h = cv2.morphologyEx(soft, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1)))
    rows = np.nonzero(long_h.any(axis=1))[0]
    # линия обреза листа прижата к краю картинки; линии ближе 1% от края — не рамка
    edge = max(3, int(0.01 * min(h, w)))
    cl = [c for c in _clusters(rows) if c[0] > edge and c[1] < h - 1 - edge]
    rects = []
    # вложенные кандидаты, сначала внешний: на скане с белыми полями линия
    # обреза бумаги может дать лишний прямоугольник снаружи настоящей рамки
    for i in range(len(cl) // 2):
        (t0, t1), (b0, b1) = cl[i], cl[-1 - i]
        if (b0 - t1) < 0.5 * h:
            break
        top_cols = np.nonzero(long_h[t0:t1 + 1].any(axis=0))[0]
        bot_cols = np.nonzero(long_h[b0:b1 + 1].any(axis=0))[0]
        top_cols = top_cols[(top_cols > edge) & (top_cols < w - 1 - edge)]
        bot_cols = bot_cols[(bot_cols > edge) & (bot_cols < w - 1 - edge)]
        if len(top_cols) == 0 or len(bot_cols) == 0:
            continue
        x0 = int(max(top_cols.min(), bot_cols.min()))
        x1 = int(min(top_cols.max(), bot_cols.max()))
        rects.append((x0, int(t0), x1, int(b1)))
    return rects


def frame_from_ink(ink: np.ndarray, rect) -> np.ndarray:
    """Все чернила, связанные с прямоугольником рамки (рамка, штамп, графы)."""
    x0, y0, x1, y1 = rect
    band = np.zeros_like(ink)
    cv2.rectangle(band, (x0, y0), (x1, y1), 1, 5)
    n, lab = cv2.connectedComponents(ink, connectivity=8)
    ids = np.unique(lab[(band > 0) & (ink > 0)])
    ids = ids[ids > 0]
    return np.isin(lab, ids).astype(np.uint8)


def find_frame(thick: np.ndarray):
    """Запасной путь: рамка — толстый кусок, занимающий не меньше 60% листа по обеим осям."""
    h, w = thick.shape
    n, lab, st, _ = cv2.connectedComponentsWithStats(thick, connectivity=8)
    best = None
    for i in range(1, n):
        x, y, bw, bh, area = st[i]
        if bw >= 0.6 * w and bh >= 0.6 * h:
            if best is None or bw * bh > st[best][2] * st[best][3]:
                best = i
    if best is None:
        return None
    return (lab == best).astype(np.uint8)


def drawing_field(frame: np.ndarray):
    """Поле чертежа: самая большая область бумаги, замкнутая рамкой.

    Ячейки штампа и графы на полях — отдельные области, поэтому они
    исключаются автоматически.
    """
    h, w = frame.shape
    free = (cv2.dilate(frame, np.ones((3, 3), np.uint8)) == 0).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(free, connectivity=4)
    best, best_area = None, 0
    for i in range(1, n):
        x, y, bw, bh, area = st[i]
        touches = x == 0 or y == 0 or x + bw >= w or y + bh >= h
        if not touches and area > best_area:
            best, best_area = i, area
    if best is None or best_area < 0.2 * h * w:
        return None
    return (lab == best).astype(np.uint8)


def _clusters(idx: np.ndarray, gap: int = 2):
    """Сгруппировать отсортированные индексы в непрерывные отрезки; вернуть [(начало, конец)]."""
    if len(idx) == 0:
        return []
    out, s, p = [], idx[0], idx[0]
    for v in idx[1:]:
        if v - p > gap:
            out.append((s, p))
            s = v
        p = v
    out.append((s, p))
    return out


def measure_stamp(frame: np.ndarray):
    """Размер (в пикселях) и форма основной надписи по ГОСТ, или None.

    Находит нижнюю и правую стороны рамки, затем горизонтальную линию,
    которая начинается от правой стороны и у которой отношение «длина /
    высота над нижней стороной» как у штампа ГОСТ 2.104: 185/55 (форма 1),
    иначе 185/40 (форма 2), иначе 185/15 (форма 2а). Форма 1 проверяется
    первой, поэтому внутренние строки её штампа (например, строка на высоте
    15 мм через часть штампа) не принимаются за штамп поменьше. Ширина у
    всех форм 185 мм.
    """
    h, w = frame.shape
    cols = np.nonzero(frame.sum(axis=0) >= 0.5 * h)[0]
    rows = np.nonzero(frame.sum(axis=1) >= 0.4 * w)[0]
    if len(cols) == 0 or len(rows) == 0:
        return None
    rc = _clusters(cols)[-1]
    bc = _clusters(rows)[-1]
    xr = (rc[0] + rc[1]) / 2
    yb = (bc[0] + bc[1]) / 2
    y_lo = int(max(0, yb - 0.45 * h))
    runs = []                       # (y, длина отрезка чернил от правой стороны рамки, высота над нижней стороной)
    for y in range(int(bc[0]) - 1, y_lo, -1):
        row = frame[y, : int(rc[0]) + 1][::-1]
        # сплошной отрезок чернил от правой стороны рамки, разрывы до 2 пикселей
        gaps, L = 0, 0
        for v in row:
            if v:
                gaps = 0
            else:
                gaps += 1
                if gaps > 2:
                    break
            L += 1
        L -= gaps
        if L >= 0.15 * w:
            runs.append((y, L, yb - y))
    for h_mm, form in STAMP_FORMS:
        target = STAMP_W_MM / h_mm
        # штамп 15 мм — малая доля высоты большого листа
        min_hgt = max(5.0, (0.008 if h_mm < 20 else 0.03) * h)
        best = None
        for y, L, hgt in runs:
            if hgt < min_hgt:
                continue
            err = abs(np.log((L / hgt) / target))
            if err < 0.12 and (best is None or err < best[0]):
                best = (err, L, hgt, y)
        if best is not None:
            return {"w_px": float(best[1]), "h_px": float(best[2]), "top_y": int(best[3]),
                    "right_x": float(xr), "bottom_y": float(yb), "h_mm": h_mm, "form": form}
    return None


# ---------------------------------------------------------------------------
def _components(g: Graph, alive: list[bool]) -> list[Component]:
    """Разбить живые рёбра графа на связные куски и посчитать у каждого длину, габарит, петли и хвосты."""
    parent = list(range(len(g.nodes)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, e in enumerate(g.edges):
        if alive[i] and e.a >= 0 and e.b >= 0:
            ra, rb = find(e.a), find(e.b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    groups: dict = {}
    for i, e in enumerate(g.edges):
        if not alive[i]:
            continue
        key = ("n", find(e.a)) if e.a >= 0 else (("n", find(e.b)) if e.b >= 0 else ("e", i))
        groups.setdefault(key, []).append(i)

    deg = np.zeros(len(g.nodes), int)
    for i, e in enumerate(g.edges):
        if alive[i]:
            if e.a >= 0:
                deg[e.a] += 1
            if e.b >= 0:
                deg[e.b] += 1
    comps = []
    for key in sorted(groups, key=lambda k: min(groups[k])):
        ids = groups[key]
        pts = np.vstack([g.edges[i].pts for i in ids])
        nodes = {n for i in ids for n in (g.edges[i].a, g.edges[i].b) if n >= 0}
        n_closed = sum(1 for i in ids if g.edges[i].closed)
        n_dead = sum(1 for i in ids if not g.edges[i].closed and g.edges[i].b < 0)
        # число петель связного графа: рёбра − узлы + 1 (цепочка с тупиком добавляет условный узел)
        cycles = (len(ids) - len(nodes) - n_dead + 1) if nodes else n_closed
        ends = sum(1 for n in nodes if deg[n] == 1) + sum(1 for i in ids if g.edges[i].b < 0 and not g.edges[i].closed)
        comps.append(Component(ids, sum(g.edges[i].length for i in ids),
                               (pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()),
                               max(cycles, 0), ends))
    return comps


def _main_width(e: Edge, dt: np.ndarray, thr: float) -> bool:
    """Ребро полной толщины основной линии: медиана полуширины вдоль него не меньше thr."""
    h, w = dt.shape
    xs = np.clip(np.round(e.pts[:, 0]).astype(int), 0, w - 1)
    ys = np.clip(np.round(e.pts[:, 1]).astype(int), 0, h - 1)
    return float(np.median(dt[ys, xs])) >= thr


def prune_spurs(g: Graph, alive: list[bool], max_len: float, rounds: int = 3,
                dt: np.ndarray | None = None, thr: float = 0.0, keep_len: float = 0.0) -> int:
    """Удалить короткие висящие ветки (заусенцы, обрубки штриховки).

    Ветка меряется от свободного конца через узлы степени 2 до первой
    настоящей развилки. Если мерить отдельные цепочки, повторные проходы
    съедали бы кромку контура, разрезанную примыкающей штриховкой на короткие куски.
    """
    removed = 0
    for _ in range(rounds):
        deg = np.zeros(len(g.nodes), int)
        inc: dict = {}
        for i, e in enumerate(g.edges):
            if alive[i] and not e.closed:
                for n in (e.a, e.b):
                    if n >= 0:
                        deg[n] += 1
                        inc.setdefault(n, []).append(i)
        to_remove: set = set()
        for i, e in enumerate(g.edges):
            if not alive[i] or e.closed or i in to_remove:
                continue
            da = deg[e.a] if e.a >= 0 else 1
            db = deg[e.b] if e.b >= 0 else 1
            # только ветки, висящие на чём-то; отдельные палочки решаются при разборе кусков
            if (da == 1) == (db == 1):
                continue
            # идти от свободного конца через узлы степени 2
            chain, total = [i], e.length
            cur, node = i, (e.b if da == 1 else e.a)
            while node >= 0 and deg[node] == 2 and total < max_len:
                nxt = [j for j in inc[node] if j != cur]
                if not nxt or nxt[0] in chain:
                    break
                cur = nxt[0]
                ce = g.edges[cur]
                chain.append(cur)
                total += ce.length
                node = ce.b if ce.a == node else ce.a
            ends_at_junction = node >= 0 and deg[node] >= 3
            # обрубок полной толщины основной линии — кусок контура, отрезанный
            # остриём стрелки или штриховкой, а не заусенец: оставить
            if total < max_len and ends_at_junction:
                if dt is None:
                    to_remove.update(chain)
                    continue
                is_main = [_main_width(g.edges[j], dt, thr) for j in chain]
                main_len = sum(g.edges[j].length for j, m in zip(chain, is_main) if m)
                if main_len >= keep_len:
                    # убрать только тонкий кончик до первого куска полной толщины
                    for j, m in zip(chain, is_main):
                        if m:
                            break
                        to_remove.add(j)
                else:
                    to_remove.update(chain)
        if not to_remove:
            break
        for i in to_remove:
            alive[i] = False
        removed += len(to_remove)
    return removed


def prune_arrowheads(g: Graph, alive: list[bool], dt: np.ndarray, thick_hw: float, max_len: float) -> int:
    """Залитые наконечники стрелок, касающиеся контура, проходят фильтр толщины.
    Это висящие ветки, которые где-то намного шире основной линии, или
    перемычки между двумя линиями с утолщением посередине."""
    h, w = dt.shape
    deg = np.zeros(len(g.nodes), int)
    for i, e in enumerate(g.edges):
        if alive[i]:
            if e.a >= 0:
                deg[e.a] += 1
            if e.b >= 0:
                deg[e.b] += 1
    removed = 0
    wide = 1.3 * thick_hw + 0.5
    skip = int(np.ceil(2.5 * thick_hw))   # у развилок тоже утолщение: концы цепочки не учитываем
    for i, e in enumerate(g.edges):
        if not alive[i] or e.closed or e.length >= max_len:
            continue
        da = deg[e.a] if e.a >= 0 else 1
        db = deg[e.b] if e.b >= 0 else 1
        xs = np.clip(np.round(e.pts[:, 0]).astype(int), 0, w - 1)
        ys = np.clip(np.round(e.pts[:, 1]).astype(int), 0, h - 1)
        prof = dt[ys, xs]
        if (da == 1) != (db == 1):
            # висящая ветка: наконечник основанием на контуре
            hit = prof.max() >= wide
        else:
            # перемычка между двумя линиями контура: размерные стрелки, зажатые между ними.
            # Нужно МЕСТНОЕ утолщение: ровно широкий штрих (контур с прилипшей вдоль
            # тонкой линией, порезанный пересечениями) — не стрелка. Зажатые стрелки
            # касаются линий тонкими остриями, поэтому широкая часть (основания) —
            # посередине. Кусок контура, кончающийся развилкой со стрелкой, широк
            # только у конца -> не удаляется.
            n = len(prof)
            idx = np.nonzero(prof >= wide)[0]
            hit = (n > 2 * skip and len(idx) > 0 and np.median(prof) < wide
                   and idx.min() > max(skip, 0.2 * n) and idx.max() < min(n - skip, 0.8 * n))
        if hit:
            alive[i] = False
            removed += 1
    return removed


def prune_tapered_arrows(g: Graph, alive: list[bool], dt: np.ndarray, thick_hw: float, max_len: float) -> int:
    """Наконечники стрелок, упёртые остриём в контур, у которых тонкая размерная
    линия ушла по фильтру толщины.

    Они остаются ветками, висящими на контуре, и выглядят одинаково на любом
    листе: от развилки на контуре к свободному концу у ветки сначала узкая
    «шейка» (остриё тоньше основной линии), потом ширина растёт до основания
    стрелки, которое шире основной линии. У настоящего куска контура ширина
    ровная. prune_spurs такие ветки оставляет (в них есть 1 мм полной
    толщины), поэтому эта проверка идёт после неё.
    """
    h, w = dt.shape
    deg = np.zeros(len(g.nodes), int)
    inc: dict = {}
    for i, e in enumerate(g.edges):
        if alive[i] and not e.closed:
            for n in (e.a, e.b):
                if n >= 0:
                    deg[n] += 1
                    inc.setdefault(n, []).append(i)
    wide = 1.3 * thick_hw + 0.5
    neck = 0.7 * thick_hw
    removed = 0
    for i, e in enumerate(g.edges):
        if not alive[i] or e.closed:
            continue
        da = deg[e.a] if e.a >= 0 else 1
        db = deg[e.b] if e.b >= 0 else 1
        if (da == 1) == (db == 1):
            continue
        # идти от свободного конца через узлы степени 2 до развилки;
        # куски складываются в направлении «от развилки к свободному концу»
        chain = [i]
        parts = [e.pts if db == 1 else e.pts[::-1]]
        cur, node, total = i, (e.a if db == 1 else e.b), e.length
        while node >= 0 and deg[node] == 2 and total < max_len:
            nxt = [j for j in inc[node] if j != cur]
            if not nxt or nxt[0] in chain:
                break
            cur = nxt[0]
            ce = g.edges[cur]
            chain.append(cur)
            parts.insert(0, ce.pts if ce.b == node else ce.pts[::-1])
            total += ce.length
            node = ce.a if ce.b == node else ce.b
        if total >= max_len or node < 0 or deg[node] < 3:
            continue
        pts = np.vstack(parts)
        xs = np.clip(np.round(pts[:, 0]).astype(int), 0, w - 1)
        ys = np.clip(np.round(pts[:, 1]).astype(int), 0, h - 1)
        prof = dt[ys, xs]
        n = len(prof)
        if n < 6:
            continue
        head, tail = prof[: max(2, int(0.35 * n))], prof[int(0.5 * n):]
        third = max(1, n // 3)
        rising = prof[-third:].mean() >= prof[:third].mean() + 1.0
        if head.min() <= neck and tail.max() >= wide and rising:
            for j in chain:
                alive[j] = False
            removed += 1
    return removed


def glyph_boxes(ink: np.ndarray, ppm: float) -> np.ndarray:
    """Габариты (x0, y0, x1, y1) пятен чернил размером с буквы и цифры (высота 1,5–10 мм, не палочки)."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    if n <= 1:
        return np.zeros((0, 4))
    # вытянутость по моментам второго порядка: обрывки штриховки — прямые палочки,
    # а буквы и цифры — нет
    ys, xs = np.nonzero(lab)
    ids = lab[ys, xs]
    xs = xs.astype(np.float64)
    ys = ys.astype(np.float64)
    cnt = np.bincount(ids, minlength=n).astype(np.float64)
    cnt[cnt == 0] = 1
    mx, my = np.bincount(ids, xs, n) / cnt, np.bincount(ids, ys, n) / cnt
    sxx = np.bincount(ids, xs * xs, n) / cnt - mx ** 2
    syy = np.bincount(ids, ys * ys, n) / cnt - my ** 2
    sxy = np.bincount(ids, xs * ys, n) / cnt - mx * my
    tr, det = sxx + syy, sxx * syy - sxy ** 2
    disc = np.sqrt(np.maximum(tr ** 2 / 4 - det, 0))
    lmax, lmin = tr / 2 + disc, np.maximum(tr / 2 - disc, 0)
    elong = np.sqrt(lmin / np.maximum(lmax, 1e-9))[1:]
    x, y, bw, bh = st[1:, 0], st[1:, 1], st[1:, 2], st[1:, 3]
    ok = (bh >= 1.5 * ppm) & (bh <= 10 * ppm) & (bw <= 10 * ppm) & (elong >= 0.25)
    return np.stack([x[ok], y[ok], x[ok] + bw[ok], y[ok] + bh[ok]], axis=1).astype(float)


def looks_like_table(g: Graph, c: Component, glyphs: np.ndarray, eps: float) -> tuple[bool, str]:
    """Сетка из горизонтальных и вертикальных линий с текстом внутри — таблица, а не деталь."""
    tot = ax = 0.0
    for i in c.edges:
        e = g.edges[i]
        for x1, y1, x2, y2 in polyline_segments(simplify(e.pts, eps, e.closed)):
            L = float(np.hypot(x2 - x1, y2 - y1))
            tot += L
            if abs(x2 - x1) <= 0.03 * L or abs(y2 - y1) <= 0.03 * L:
                ax += L
    frac = ax / tot if tot else 0
    if frac < 0.95:
        return False, ""
    x0, y0, x1, y1 = c.bbox
    inside = ((glyphs[:, 0] >= x0) & (glyphs[:, 1] >= y0) & (glyphs[:, 2] <= x1) & (glyphs[:, 3] <= y1)).sum() \
        if len(glyphs) else 0
    if inside >= 15:
        return True, f"table: {frac * 100:.0f}% axis-aligned, {c.cycles} cells, {int(inside)} text glyphs inside"
    return False, ""


def _interior_mask(g: Graph, comps: list[Component], shape, ppm: float):
    """Залитые силуэты крупных кусков (отверстия залиты) на сетке с шагом q
    пикселей. Нужны, чтобы отличить элементы внутри вида (отверстия) от
    пометок рядом с ним (обозначения разрезов, буквы)."""
    q = max(1, int(round(ppm / 4)))          # ячейки примерно 0,25 мм
    h, w = shape[0] // q + 1, shape[1] // q + 1
    m = np.zeros((h, w), np.uint8)
    for c in comps:
        for i in c.edges:
            p = np.round(g.edges[i].pts / q).astype(np.int32)
            cv2.polylines(m, [p.reshape(-1, 1, 2)], g.edges[i].closed, 1, 1)
    k = max(3, int(round(INSIDE_PAD_MM * ppm / q)) | 1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    # заливка: всё, до чего нельзя дойти от края, — внутри
    pad = np.zeros((h + 2, w + 2), np.uint8)
    pad[1:-1, 1:-1] = m
    ff = pad.copy()
    mask = np.zeros((h + 4, w + 4), np.uint8)
    cv2.floodFill(ff, mask, (0, 0), 2)
    interior = (ff[1:-1, 1:-1] != 2).astype(np.uint8)
    return interior, q


# ---------------------------------------------------------------------------
UPSAMPLE_BELOW_HW = 2.5   # при основной полуширине (px) не больше этой работаем в 2× увеличении


def extract(sheet: Sheet) -> Result:
    """Контур листа. Тонко нарисованный лист обрабатывается в 2× увеличении, координаты потом делятся обратно."""
    log: list[str] = []
    H0, W0 = sheet.gray.shape
    log.append(f"image {W0}x{H0} px, source={sheet.source}, file_dpi={sheet.file_dpi}")
    for n in sheet.notes:
        log.append(f"note: {n}")

    gray = sheet.gray
    ink, t = binarize(gray)
    log.append(f"binarize: Otsu threshold {t:.0f}, ink {ink.mean() * 100:.1f}% of sheet")
    lw = line_weights(ink)
    f = 1
    if lw["thick_hw"] <= UPSAMPLE_BELOW_HW:
        # тонкие и основные линии здесь различаются на ~1 пиксель: в 2× увеличении
        # сглаживание даёт точность в полпикселя
        f = 2
        log.append(f"main lines only {lw['thick_hw']:.2f}px half-width: working at {f}x resolution")
        gray = cv2.resize(sheet.gray, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
        ink, t = binarize(gray)
        lw = line_weights(ink)
    work = Sheet(sheet.path, gray, sheet.source, sheet.pages,
                 sheet.file_dpi * f if sheet.file_dpi else None, upscale=f,
                 geometry_changed=sheet.geometry_changed)
    res = _extract_work(work, ink, lw, log)
    if f != 1:
        res.segments_px = [tuple(v / f for v in s) for s in res.segments_px]
        res.dropped_segments_px = [tuple(v / f for v in s) for s in res.dropped_segments_px]
        res.scale.px_per_mm /= f
        res.metrics["px_per_mm"] = round(res.scale.px_per_mm, 4)
        res.field_bbox = tuple(int(v / f) for v in res.field_bbox)
    res.metrics["work_scale"] = f
    res.frame_mask = None
    return res


def _extract_work(sheet: Sheet, ink: np.ndarray, lw: dict, log: list[str]) -> Result:
    """Всё построение контура на рабочей картинке: рамка и масштаб, скелет, чистка графа, решение по кускам, метрики."""
    gray = sheet.gray
    H, W = gray.shape
    log.append(f"line weights: thin half-width {lw['thin_hw']:.2f}px, thick {lw['thick_hw']:.2f}px, "
               f"threshold {lw['thr']:.2f}px, thick share {lw['thick_share']}, separable={lw['separable']}; "
               f"histogram {lw['hist']}")
    thick = thick_mask(ink, lw["dt"], lw["thr"], lw["thick_hw"])
    log.append(f"thick strokes: {int(thick.sum())} px ({thick.sum() / max(ink.sum(), 1) * 100:.0f}% of ink)")

    rects = frame_rect_from_lines(ink)
    frame = None
    if rects:
        # рамка — прямоугольник, к которому пристыкован штамп; если штампа нет
        # ни у одного — внешний
        for k, rect in enumerate(rects):
            cand = frame_from_ink(ink, rect)
            if k == 0:
                frame, chosen = cand, rect
            if len(rects) == 1 or measure_stamp(cand) is not None:
                frame, chosen = cand, rect
                break
        rect = chosen
        log.append(f"frame: rectangle from long lines x={rect[0]}..{rect[2]} y={rect[1]}..{rect[3]} px "
                   f"({len(rects)} nested candidate(s)); attached ink (title block, boxes) {int(frame.sum())} px")
    else:
        frame = find_frame(thick)
        if frame is not None:
            log.append("frame: long lines not found, used largest thick component")
    stamp = None
    fieldm = None
    if frame is None:
        log.append("frame: NOT found")
    else:
        x, y, bw, bh = cv2.boundingRect(frame)
        log.append(f"frame: bbox x={x} y={y} w={bw} h={bh} px (with attached title block/boxes)")
        stamp = measure_stamp(frame)
        log.append("title block: " + (f"{stamp['w_px']:.0f}x{stamp['h_px']:.0f} px" if stamp else "NOT measured")
                   + (f", GOST 2.104 {stamp['form']} (185 x {stamp['h_mm']:g} mm)"
                      if stamp and stamp["form"] != "form 1" else ""))
        fieldm = drawing_field(frame)
        log.append("drawing field: " + ("enclosed region found" if fieldm is not None else "NOT enclosed, using frame bbox"))

    scale = decide_scale(sheet, stamp["w_px"] if stamp else None)
    ppm = scale.px_per_mm
    log.append(f"format: {scale.fmt} {scale.orientation}, {ppm:.3f} px/mm ({ppm * 25.4:.0f} dpi) "
               f"via {scale.method}; aspect candidates {scale.candidates}")
    if stamp:
        log.append(f"title block in mm: {stamp['w_px'] / ppm:.1f} x {stamp['h_px'] / ppm:.1f} "
                   f"(GOST 185 x {stamp['h_mm']:g})")

    # остаются только толстые штрихи внутри поля чертежа
    if frame is not None:
        if fieldm is not None:
            keep = fieldm
        else:
            x, y, bw, bh = cv2.boundingRect(frame)
            keep = np.zeros_like(thick)
            m = int(2 * ppm)
            keep[y + m:y + bh - m, x + m:x + bw - m] = 1
        part = thick & keep & (1 - frame)
    else:
        m = int(10 * ppm)
        part = thick.copy()
        part[:m] = 0
        part[-m:] = 0
        part[:, :m] = 0
        part[:, -m:] = 0
    field_bbox = cv2.boundingRect(fieldm) if fieldm is not None else (
        cv2.boundingRect(frame) if frame is not None else (0, 0, W, H))

    skel = cv2.ximgproc.thinning(part * 255) > 0
    g = skeleton_graph(skel)
    alive = [True] * len(g.edges)
    log.append(f"skeleton graph: {len(g.nodes)} nodes, {len(g.edges)} chains")

    n_tiny = 0
    for i, e in enumerate(g.edges):
        if e.length < TINY_LEN_MM * ppm and (e.a < 0 or e.b < 0 or e.closed or e.a == e.b):
            alive[i] = False
            n_tiny += 1
    n_arrow = prune_arrowheads(g, alive, lw["dt"], lw["thick_hw"], ARROW_MAX_MM * ppm)
    n_spur = prune_spurs(g, alive, SPUR_MM * ppm, dt=lw["dt"], thr=0.85 * lw["thick_hw"],
                         keep_len=1.0 * ppm)
    # крупность куска решается, пока стрелки ещё на месте: маленький выносной
    # вид теряет на следующем шаге свои стрелки и не должен из-за этого
    # опуститься ниже порога «крупного куска»
    pre_big_edges: set = set()
    for c in _components(g, alive):
        if c.length_px >= BIG_LEN_MM * ppm and \
                max(c.bbox[2] - c.bbox[0], c.bbox[3] - c.bbox[1]) >= BIG_SIZE_MM * ppm:
            pre_big_edges.update(c.edges)
    n_taper = prune_tapered_arrows(g, alive, lw["dt"], lw["thick_hw"], ARROW_MAX_MM * ppm)
    if n_taper:
        # на развилке, где висела стрелка, может остаться короткий заусенец
        n_spur += prune_spurs(g, alive, SPUR_MM * ppm, dt=lw["dt"], thr=0.85 * lw["thick_hw"],
                              keep_len=1.0 * ppm)
    log.append(f"pruned: {n_tiny} tiny chains (<{TINY_LEN_MM} mm), {n_arrow} arrowheads, "
               f"{n_taper} arrowheads tapering off the outline, {n_spur} dangling spurs (<{SPUR_MM} mm)")

    comps = _components(g, alive)
    fx, fy, fw, fh = field_bbox
    big = [c for c in comps
           if (c.length_px >= BIG_LEN_MM * ppm and max(c.bbox[2] - c.bbox[0], c.bbox[3] - c.bbox[1]) >= BIG_SIZE_MM * ppm)
           or any(i in pre_big_edges for i in c.edges)]
    # знак шероховатости в правом верхнем углу поля (ГОСТ 2.309)
    for c in big:
        in_corner = (c.bbox[0] >= fx + fw - CORNER_W_MM * ppm and c.bbox[3] <= fy + CORNER_H_MM * ppm)
        if in_corner and c.cycles == 0:
            c.keep, c.reason = False, "roughness sign zone (top-right corner)"
    glyphs = glyph_boxes(ink, ppm)
    for c in big:
        if c.keep:
            is_tab, why = looks_like_table(g, c, glyphs, SIMPLIFY_MM * ppm)
            if is_tab:
                c.keep, c.reason = False, why
    big_kept = [c for c in big if c.keep]
    interior, q = _interior_mask(g, big_kept, (H, W), ppm)
    for c in comps:
        if c in big:
            continue
        pts = np.vstack([g.edges[i].pts for i in c.edges])
        ix = np.clip((pts[:, 0] / q).astype(int), 0, interior.shape[1] - 1)
        iy = np.clip((pts[:, 1] / q).astype(int), 0, interior.shape[0] - 1)
        inside = bool(interior[iy, ix].mean() >= 0.8) if len(pts) else False
        size = max(c.bbox[2] - c.bbox[0], c.bbox[3] - c.bbox[1])
        if inside and c.cycles > 0 and c.endpoints == 0:
            c.reason = "small closed element inside a part view (hole/feature)"
        elif inside and c.cycles > 0 and c.endpoints > 0 and size <= LETTER_MAX_MM * ppm:
            # маленькая петля с хвостами — буква (А, Б, Р, 6, 9); у отверстия хвостов нет
            c.keep = False
            c.reason = "letter inside a part view (loop with tails)"
        elif c.cycles >= 1 and c.endpoints == 0 and size >= RINGS_MIN_MM * ppm:
            # замкнутая фигура без хвостов крупнее любой буквы (маленький вид-
            # шестигранник, окружности ступицы): вид детали, не текст
            c.reason = "closed figure >= 10 mm without loose ends, kept"
        elif inside and c.length_px >= SMALL_OPEN_KEEP_MM * ppm:
            c.reason = "small element inside a part view"
        elif not big_kept and c.length_px >= 5 * ppm:
            c.reason = "kept: no large component on sheet"
        else:
            c.keep = False
            c.reason = "small isolated stroke (text/arrow/symbol)"

    kept = [c for c in comps if c.keep]
    dropped = [c for c in comps if not c.keep]
    reasons: dict = {}
    for c in dropped:
        reasons[c.reason] = reasons.get(c.reason, 0) + 1
    log.append(f"components: {len(comps)} total, {len(big)} large, kept {len(kept)}, dropped {len(dropped)} {reasons}")

    def _desc(c: Component) -> str:
        x0, y0, x1, y1 = (v / ppm for v in c.bbox)
        return (f"bbox mm ({x0:.0f},{y0:.0f})-({x1:.0f},{y1:.0f}) len {c.length_px / ppm:.0f} mm, "
                f"loops {c.cycles}, open ends {c.endpoints}")
    for c in sorted(kept, key=lambda c: -c.length_px):
        log.append(f"  kept: {_desc(c)}" + (f" [{c.reason}]" if c.reason else " [large]"))
    for c in sorted(dropped, key=lambda c: -c.length_px)[:8]:
        log.append(f"  dropped: {_desc(c)} [{c.reason}]")
    if len(dropped) > 8:
        log.append(f"  ... and {len(dropped) - 8} more small dropped pieces")

    eps = SIMPLIFY_MM * ppm
    segs, dsegs = [], []
    for c in comps:
        for i in c.edges:
            e = g.edges[i]
            s = polyline_segments(simplify(e.pts, eps, e.closed))
            (segs if c.keep else dsegs).extend(s)

    total_len = sum(c.length_px for c in kept) / ppm
    cycles = sum(c.cycles for c in kept)
    ends = sum(c.endpoints for c in kept)
    if kept:
        x0 = min(c.bbox[0] for c in kept); y0 = min(c.bbox[1] for c in kept)
        x1 = max(c.bbox[2] for c in kept); y1 = max(c.bbox[3] for c in kept)
        bbox_mm = [round(v / ppm, 1) for v in (x0, y0, x1, y1)]
        area_frac = (x1 - x0) * (y1 - y0) / max(fw * fh, 1)
    else:
        bbox_mm, area_frac = None, 0.0
    # какая доля толстых линий поля попала в контур
    cov = 0.0
    if part.sum() > 0 and segs:
        m = np.zeros_like(part)
        for x1_, y1_, x2_, y2_ in segs:
            cv2.line(m, (int(round(x1_)), int(round(y1_))), (int(round(x2_)), int(round(y2_))), 1,
                     max(3, int(2 * lw["thick_hw"] + 3)))
        cov = float((part & m).sum() / part.sum())

    metrics = {
        "image_px": [W, H],
        "format": scale.fmt, "orientation": scale.orientation,
        "px_per_mm": round(ppm, 4), "scale_method": scale.method,
        "line_weight_thin_hw_px": lw["thin_hw"], "line_weight_thick_hw_px": lw["thick_hw"],
        "line_weights_separable": lw["separable"],
        "thick_share": lw["thick_share"],
        "field_bbox_mm": [round(v / ppm, 1) for v in (fx, fy, fx + fw, fy + fh)],
        "frame_found": frame is not None, "title_block_found": stamp is not None,
        "field_enclosed": fieldm is not None,
        "segments": len(segs), "dropped_segments": len(dsegs),
        "components_kept": len(kept), "components_dropped": len(dropped),
        "drop_reasons": reasons,
        "contour_length_mm": round(total_len, 1),
        "closed_loops": cycles, "open_ends": ends,
        "contour_bbox_mm": bbox_mm, "contour_bbox_field_fraction": round(area_frac, 3),
        "thick_ink_covered": round(cov, 3),
        "largest_dropped_mm": round(max((c.length_px for c in dropped
                                         if not c.reason.startswith(("roughness", "table"))), default=0) / ppm, 1),
        "tables_removed": sum(1 for c in dropped if c.reason.startswith("table")),
        "largest_kept_mm": round(max((c.length_px for c in kept), default=0) / ppm, 1),
    }
    log.append(f"result: {len(segs)} segments, length {total_len:.0f} mm, closed loops {cycles}, "
               f"open ends {ends}, thick ink in field covered {cov * 100:.0f}%")
    return Result(scale, segs, metrics, log, frame, field_bbox, dsegs)
