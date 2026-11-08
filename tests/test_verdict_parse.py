"""Разбор ответа модели: green только когда модель действительно сказала green."""
import unittest

from contour.verdict import _parse


class ParseVerdict(unittest.TestCase):
    def test_json(self):
        self.assertEqual(_parse('{"verdict": "green", "reason": "деталь обведена целиком"}'),
                         ("green", "деталь обведена целиком"))

    def test_json_inside_text(self):
        v = _parse('Ответ:\n```json\n{"verdict": "yellow", "reason": "в контур попала буква А"}\n```')
        self.assertEqual(v, ("yellow", "в контур попала буква А"))

    def test_plain_text_one_color(self):
        self.assertEqual(_parse("red\nконтур не построен, лист пустой")[0], "red")

    def test_negation_is_not_green(self):
        for t in ("Не green: главный вид обведён не полностью",
                  "not green, the main view is missing",
                  "non-green: не обведён выносной элемент",
                  '{"verdict": "not green", "reason": "главный вид не обведён"}'):
            self.assertIsNone(_parse(t), t)

    def test_two_colors_rejected(self):
        self.assertIsNone(_parse("это не green, а yellow: главный вид не обведён"))
        self.assertIsNone(_parse("green или yellow, сложно сказать по картинке"))

    def test_no_reason_rejected(self):
        self.assertIsNone(_parse("green"))
        self.assertIsNone(_parse('{"verdict": "green", "reason": ""}'))
        self.assertIsNone(_parse('{"verdict": "green", "reason": "green"}'))

    def test_negation_inside_reason_is_fine(self):
        v = _parse('{"verdict": "yellow", "reason": "не обведён главный вид, поэтому не green"}')
        self.assertEqual(v[0], "yellow")

    def test_garbage(self):
        self.assertIsNone(_parse(""))
        self.assertIsNone(_parse("не могу оценить картинку"))


if __name__ == "__main__":
    unittest.main()
