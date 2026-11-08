"""Масштаб и формат: по штампу, по DPI с подгонкой к формату ГОСТ, по PDF."""
import unittest
from pathlib import Path

import numpy as np

from contour.extract import binarize, frame_from_ink, frame_rect_from_lines, measure_stamp
from contour.sheet import Sheet, decide_scale

from synth import A4, PPM, draw_sheet, px


def blank(w_mm, h_mm, ppm):
    return np.full((int(round(h_mm * ppm)), int(round(w_mm * ppm))), 255, np.uint8)


class Scale(unittest.TestCase):
    def test_stamp_width_gives_scale(self):
        s = Sheet(Path("x.png"), blank(*A4, PPM), "image")
        sc = decide_scale(s, stamp_w_px=185 * PPM)
        self.assertEqual((sc.fmt, sc.orientation), ("A4", "portrait"))
        self.assertAlmostEqual(sc.px_per_mm, PPM, delta=PPM * 0.01)

    def test_wrong_dpi_within_19_percent_snaps_to_format(self):
        # A3 альбомный при 200 dpi, а в файле записано 220 dpi (ошибка 10%)
        ppm = 200 / 25.4
        s = Sheet(Path("x.png"), blank(420, 297, ppm), "image", file_dpi=220)
        sc = decide_scale(s, stamp_w_px=None)
        self.assertEqual((sc.fmt, sc.orientation), ("A3", "landscape"))
        self.assertAlmostEqual(sc.px_per_mm, ppm, delta=ppm * 0.01)

    def test_wrong_dpi_beyond_19_percent_picks_neighbour(self):
        # то же, но записано 300 dpi (ошибка 50%): лист кажется мельче, выбирается
        # соседний A4, мм неверны в √2 раза. Это известное ограничение (README,
        # «Ограничения»), тест его фиксирует.
        ppm = 200 / 25.4
        s = Sheet(Path("x.png"), blank(420, 297, ppm), "image", file_dpi=300)
        sc = decide_scale(s, stamp_w_px=None)
        self.assertEqual(sc.fmt, "A4")
        self.assertAlmostEqual(sc.px_per_mm / ppm, 2 ** 0.5, delta=0.01)

    def test_pdf_uses_render_dpi(self):
        ppm = 300 / 25.4
        s = Sheet(Path("x.pdf"), blank(250, 180, ppm), "pdf", file_dpi=300)
        sc = decide_scale(s, stamp_w_px=None)
        self.assertAlmostEqual(sc.px_per_mm, ppm, places=6)


class Stamp(unittest.TestCase):
    def _stamp(self, h_mm):
        ink, _ = binarize(draw_sheet(stamp_h_mm=h_mm, part=False))
        rects = frame_rect_from_lines(ink)
        self.assertTrue(rects)
        return measure_stamp(frame_from_ink(ink, rects[0]))

    def test_form1(self):
        st = self._stamp(55)
        self.assertEqual(st["form"], "form 1")
        self.assertAlmostEqual(st["w_px"] / PPM, 185, delta=2)

    def test_form2a_small(self):
        st = self._stamp(15)
        self.assertEqual(st["form"], "form 2a")
        self.assertAlmostEqual(st["w_px"] / PPM, 185, delta=2)

    def test_frame_found(self):
        ink, _ = binarize(draw_sheet())
        x0, y0, x1, y1 = frame_rect_from_lines(ink)[0]
        self.assertAlmostEqual(x0 / PPM, 20, delta=1)
        self.assertAlmostEqual(y0 / PPM, 5, delta=1)
        self.assertAlmostEqual(x1 / PPM, A4[0] - 5, delta=1)
        self.assertAlmostEqual(y1 / PPM, A4[1] - 5, delta=1)


if __name__ == "__main__":
    unittest.main()
