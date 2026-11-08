"""Скелет -> граф цепочек -> отрезки.

Всё детерминировано: пиксели обходятся построчно, случайности нет, поэтому
одна и та же маска всегда даёт одни и те же отрезки.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

# 8 соседей; сначала 4 прямых, чтобы обход предпочитал прямые шаги диагональным
_NEIGH = [(-1, 0), (0, -1), (0, 1), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]


@dataclass
class Edge:
    """Ребро графа: цепочка пикселей скелета между двумя узлами."""
    a: int                      # узел в начале (-1 у замкнутого кольца)
    b: int                      # узел в конце  (-1 у замкнутого кольца)
    pts: np.ndarray             # массив (N, 2) координат (x, y) в пикселях
    closed: bool = False

    @property
    def length(self) -> float:
        d = np.diff(self.pts, axis=0)
        n = float(np.hypot(d[:, 0], d[:, 1]).sum())
        if self.closed and len(self.pts) > 1:
            n += float(np.hypot(*(self.pts[0] - self.pts[-1])))
        return n


@dataclass
class Graph:
    """Граф скелета: узлы (концы и развилки) и рёбра между ними."""
    nodes: list[tuple[float, float]] = field(default_factory=list)  # центры узлов (x, y)
    edges: list[Edge] = field(default_factory=list)


def skeleton_graph(skel: np.ndarray) -> Graph:
    """Перевести скелет толщиной 1 пиксель (массив bool) в граф.

    Узел — группа соседних пикселей, у которых число соседей не равно двум
    (концы линий и развилки). Рёбра — цепочки пикселей между узлами.
    Цепочка совсем без узлов (отдельное кольцо, например окружность
    отверстия) становится замкнутым ребром.
    """
    sk = skel.astype(np.uint8)
    h, w = sk.shape
    k = np.ones((3, 3), np.float32)
    k[1, 1] = 0
    nb = cv2.filter2D(sk, cv2.CV_16S, k, borderType=cv2.BORDER_CONSTANT)
    node_mask = (sk > 0) & (nb != 2)

    n_lab, lab = cv2.connectedComponents(node_mask.astype(np.uint8), connectivity=8)
    g = Graph()
    if n_lab > 1:
        ys, xs = np.nonzero(lab)
        ids = lab[ys, xs]
        cnt = np.bincount(ids, minlength=n_lab).astype(float)
        cx = np.bincount(ids, weights=xs, minlength=n_lab) / np.maximum(cnt, 1)
        cy = np.bincount(ids, weights=ys, minlength=n_lab) / np.maximum(cnt, 1)
        g.nodes = [(float(cx[i]), float(cy[i])) for i in range(1, n_lab)]
    node_id = lab.astype(np.int32) - 1  # -1 там, где не узел

    visited = np.zeros_like(sk, dtype=bool)
    skb = sk > 0

    def walk(y0, x0, y1, x1, start_node):
        """Идти от (y0, x0) через (y1, x1), пока не встретится пиксель узла."""
        path = [(x0, y0)]
        py, px = y0, x0
        cy_, cx_ = y1, x1
        while True:
            nid = node_id[cy_, cx_]
            if nid >= 0:
                path.append((cx_, cy_))
                return path, nid
            visited[cy_, cx_] = True
            path.append((cx_, cy_))
            nxt = None
            node_hit = None
            for dy, dx in _NEIGH:
                ny, nx = cy_ + dy, cx_ + dx
                if ny < 0 or nx < 0 or ny >= h or nx >= w or not skb[ny, nx]:
                    continue
                if ny == py and nx == px:
                    continue
                if node_id[ny, nx] >= 0:
                    # не возвращаться сразу в узел, из которого только что вышли
                    if node_id[ny, nx] == start_node and len(path) <= 2:
                        continue
                    if node_hit is None:
                        node_hit = (ny, nx)
                elif not visited[ny, nx] and nxt is None:
                    nxt = (ny, nx)
            if node_hit is not None:
                path.append((node_hit[1], node_hit[0]))
                return path, int(node_id[node_hit])
            if nxt is None:  # тупик без узла: цепочка обрывается здесь
                return path, -1
            py, px = cy_, cx_
            cy_, cx_ = nxt

    # 1) цепочки, начинающиеся в узлах
    nys, nxs = np.nonzero(node_mask)
    for y0, x0 in zip(nys.tolist(), nxs.tolist()):
        a = int(node_id[y0, x0])
        for dy, dx in _NEIGH:
            y1, x1 = y0 + dy, x0 + dx
            if y1 < 0 or x1 < 0 or y1 >= h or x1 >= w or not skb[y1, x1]:
                continue
            if node_id[y1, x1] >= 0 or visited[y1, x1]:
                continue
            path, b = walk(y0, x0, y1, x1, a)
            pts = np.array(path, dtype=np.float64)
            # концы притягиваются к центрам узлов, чтобы соседние рёбра сходились точно
            pts[0] = g.nodes[a]
            if b >= 0:
                pts[-1] = g.nodes[b]
            g.edges.append(Edge(a, b, pts))

    # 2) замкнутые кольца без единого узла
    rest = skb & ~visited & ~node_mask
    rys, rxs = np.nonzero(rest)
    for y0, x0 in zip(rys.tolist(), rxs.tolist()):
        if visited[y0, x0]:
            continue
        visited[y0, x0] = True
        path = [(x0, y0)]
        py, px, cy_, cx_ = -10, -10, y0, x0
        while True:
            nxt = None
            for dy, dx in _NEIGH:
                ny, nx = cy_ + dy, cx_ + dx
                if 0 <= ny < h and 0 <= nx < w and rest[ny, nx] and not visited[ny, nx]:
                    nxt = (ny, nx)
                    break
            if nxt is None:
                break
            visited[nxt] = True
            path.append((nxt[1], nxt[0]))
            cy_, cx_ = nxt
        if len(path) >= 3:
            g.edges.append(Edge(-1, -1, np.array(path, dtype=np.float64), closed=True))
    return g


def simplify(pts: np.ndarray, eps: float, closed: bool) -> np.ndarray:
    """Упростить ломаную методом Дугласа–Пекера: все точки результата в
    пределах eps пикселей от исходной цепочки."""
    if len(pts) <= 2:
        return pts
    c = pts.reshape(-1, 1, 2).astype(np.float32)
    out = cv2.approxPolyDP(c, eps, closed).reshape(-1, 2).astype(np.float64)
    if closed and len(out) >= 2:
        out = np.vstack([out, out[:1]])
    # у незамкнутой цепочки концы остаются точно в узлах
    if not closed:
        out[0], out[-1] = pts[0], pts[-1]
    return out


def polyline_segments(pts: np.ndarray) -> list[tuple[float, float, float, float]]:
    """Ломаная -> список отрезков (x1, y1, x2, y2), без отрезков нулевой длины."""
    segs = []
    for (x1, y1), (x2, y2) in zip(pts[:-1], pts[1:]):
        if x1 == x2 and y1 == y2:
            continue
        segs.append((x1, y1, x2, y2))
    return segs
