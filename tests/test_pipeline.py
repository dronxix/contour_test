"""Весь лист целиком: контур в мм, детерминизм, пропуск готового, red для битого файла."""
import json
import os
import tempfile
import unittest
from pathlib import Path

os.environ["VERDICT_MODE"] = "off"          # модель не нужна: проверяется только программа
os.environ["VERDICT_REVIEW_FILE"] = ""

from contour.pipeline import is_done, process_file

from synth import PART, draw_sheet, save


def bbox(segments):
    xs = [v for s in segments for v in (s[0], s[2])]
    ys = [v for s in segments for v in (s[1], s[3])]
    return min(xs), min(ys), max(xs), max(ys)


class Pipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.src = save(draw_sheet(), cls.root / "in" / "лист.png")
        cls.od = cls.root / "out" / "лист"
        cls.summary = process_file(cls.src, cls.od, "лист.png")
        cls.contour = json.loads((cls.od / "contour.json").read_text(encoding="utf-8"))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def segments(self):
        c = self.contour
        segs = c.get("segments_mm") or c.get("segments")
        self.assertTrue(segs, f"в contour.json нет отрезков: {list(c)}")
        return [s if isinstance(s, (list, tuple)) else (s["x1"], s["y1"], s["x2"], s["y2"]) for s in segs]

    def test_all_files_written(self):
        for f in ("overlay.png", "contour.json", "verdict.txt", "log.txt", "done.json"):
            self.assertTrue((self.od / f).exists(), f)

    def test_part_outline_in_mm(self):
        # контур — только деталь: рамка, штамп, размеры и осевая не попали
        x0, y0, x1, y1 = bbox(self.segments())
        for got, want in zip((x0, y0, x1, y1), PART):
            self.assertAlmostEqual(got, want, delta=1.0)

    def test_without_model_is_yellow(self):
        self.assertEqual(self.summary["verdict"], "yellow")

    def test_deterministic(self):
        od2 = self.root / "out2" / "лист"
        process_file(self.src, od2, "лист.png")
        self.assertEqual((self.od / "contour.json").read_bytes(), (od2 / "contour.json").read_bytes())

    def test_skip_and_redo(self):
        self.assertTrue(is_done(self.src, self.od))
        changed = save(draw_sheet(stamp_h_mm=40), self.root / "in2" / "лист.png")
        self.assertFalse(is_done(changed, self.od))
        os.environ["CONTOUR_DPI"] = "200"
        try:
            self.assertFalse(is_done(self.src, self.od))
        finally:
            del os.environ["CONTOUR_DPI"]

    def test_broken_file_is_red(self):
        bad = self.root / "in" / "битый.jpg"
        bad.write_bytes(b"not an image")
        od = self.root / "out" / "битый"
        s = process_file(bad, od, "битый.jpg")
        self.assertEqual(s["verdict"], "red")
        self.assertTrue((od / "overlay.png").exists())
        self.assertTrue((od / "contour.json").exists())
        self.assertFalse((od / "done.json").exists())      # повторится при следующем запуске


if __name__ == "__main__":
    unittest.main()
