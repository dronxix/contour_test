"""Обработка одного листа и общий отчёт report.md."""
from __future__ import annotations

import functools
import hashlib
import json
import re
import time
import traceback
from pathlib import Path

from .extract import extract
from .output import (view_boxes, write_contour_json, write_empty_contour, write_failure_image, write_overlay,
                     write_preview, write_zoom)
from .sheet import SUPPORTED_EXT, deskew_mode, load_sheet, pdf_page_count, render_dpi
from .verdict import Verdict, decide, image_px, verdict_fingerprint, zoom_max, zoom_plan, zoom_px

# файлы прошлого запуска, которые не должны остаться рядом с новым результатом листа
OUTPUT_FILES = ("overlay.png", "contour.json", "preview.png", "verdict.txt", "log.txt", "done.json")

DONE_FILE = "done.json"


def find_inputs(root: Path, exclude: Path | None = None) -> list[Path]:
    """Чертежи в папке root (рекурсивно). exclude — папка результата, если она
    лежит внутри root: её overlay и preview не должны попасть во вход как
    новые чертежи."""
    ex = exclude.resolve() if exclude is not None else None
    files = []
    for p in root.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in SUPPORTED_EXT:
            continue
        if ex is not None and p.resolve().is_relative_to(ex):
            continue
        files.append(p)
    return sorted(files, key=lambda p: p.relative_to(root).as_posix())


def out_dir_for(src: Path, in_root: Path, out_root: Path, all_inputs: list[Path]) -> Path:
    """Папка результата листа: повторяет путь чертежа относительно входа."""
    rel = src.relative_to(in_root)
    name = rel.stem
    # одно имя с разными расширениями в одной папке -> разные папки (x_pdf, x_jpg)
    if sum(1 for p in all_inputs if p.parent == src.parent and p.stem == src.stem) > 1:
        name = rel.name.replace(".", "_")
    return out_root / rel.parent / name


PAGE_DIR = "_лист{}"            # окончание папки результата для страницы N многостраничного PDF
PAGE_REL = " [лист {}]"         # то же в report.md, файлах ревью и логах
_PAGE_REL_RE = re.compile(r"^(.*) \[лист (\d+)\]$")


def plan_jobs(inputs: list[Path], in_root: Path, out_root: Path) -> list[tuple[Path, Path, str, int]]:
    """Для каждого листа: (файл, папка результата, имя в отчёте, страница с 0).

    Многостраничный PDF — по листу на страницу: папки <имя>_лист1,
    <имя>_лист2, …; имя в отчёте «файл.pdf [лист 1]». Одностраничный PDF и
    картинки остаются под своим именем.
    """
    jobs = []
    for src in inputs:
        od = out_dir_for(src, in_root, out_root, inputs)
        rel = src.relative_to(in_root).as_posix()
        n = pdf_page_count(src)
        if n == 1:
            jobs.append((src, od, rel, 0))
        else:
            for k in range(n):
                jobs.append((src, od.with_name(od.name + PAGE_DIR.format(k + 1)), rel + PAGE_REL.format(k + 1), k))
    return jobs


def split_rel(rel: str) -> tuple[str, int]:
    """Имя в отчёте -> (путь файла относительно папки чертежей, страница с 0)."""
    m = _PAGE_REL_RE.match(rel)
    return (m.group(1), int(m.group(2)) - 1) if m else (rel, 0)


def sha1(path: Path) -> str:
    """SHA-1 содержимого файла (читается кусками по 1 МБ)."""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


REQUIRED_FILES = ("overlay.png", "contour.json", "verdict.txt", "log.txt", DONE_FILE)


def is_done(src: Path, od: Path) -> bool:
    """Лист пропускается, только если его результат полный и сделан из того же
    файла, тем же кодом и с теми же настройками (контура и вердикта)."""
    if not all((od / f).exists() for f in REQUIRED_FILES):
        return False
    try:
        j = json.loads((od / DONE_FILE).read_text(encoding="utf-8"))
        return j.get("sha1") == sha1(src) and j.get("fingerprint") == run_fingerprint()
    except Exception:
        return False


def contour_settings() -> dict:
    """Настройки, от которых зависят координаты контура (а не только вердикт)."""
    return {"CONTOUR_DPI": render_dpi(), "CONTOUR_DESKEW": deskew_mode()}


@functools.lru_cache(maxsize=1)
def code_version() -> str:
    """Отпечаток кода обработки: любое изменение в contour/*.py делает старые
    результаты устаревшими."""
    h = hashlib.sha1()
    for p in sorted(Path(__file__).parent.glob("*.py")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:12]


def run_fingerprint() -> dict:
    """Отпечаток прогона: код + настройки контура + настройки вердикта."""
    return {"code": code_version(), "contour": contour_settings(), "verdict": verdict_fingerprint()}


# вердикт, который может получиться при следующем запуске (модель недоступна,
# таймаут): done.json не пишется, и лист считается заново, а не остаётся с
# yellow «нет ответа»
RETRY_SOURCES = ("program: model call failed", "program: model answer unparseable", "program: review file error")


def _model_images(od: Path, gray, res, log: list[str]) -> list[Path]:
    """Картинки для модели: весь лист и, если включён зум (VERDICT_ZOOM),
    крупные планы видов. Сохраняются рядом с результатом, чтобы было видно,
    что именно видела модель."""
    side = image_px()
    preview = od / "preview.png"
    write_preview(preview, gray, res.segments_px, max_side=side)
    images = [preview]
    ppm = res.scale.px_per_mm
    ppm_overview = ppm * min(1.0, side / max(gray.shape))
    zoom, why = zoom_plan(ppm_overview)
    log.append(f"model pictures: preview.png ({side} px); close-ups: {'yes' if zoom else 'no'} ({why})")
    if zoom:
        zpx = zoom_px()
        boxes = view_boxes(res.segments_px, gray.shape, ppm)
        nmax = zoom_max()
        # если видов больше лимита — берутся самые большие, порядок остаётся как при чтении
        if len(boxes) > nmax:
            keep = sorted(boxes, key=lambda b: -(b[2] - b[0]) * (b[3] - b[1]))[:nmax]
            boxes = [b for b in boxes if b in keep]
        for k, b in enumerate(boxes, 1):
            p = od / f"preview_zoom_{k}.png"
            write_zoom(p, gray, res.segments_px, b, max_side=zpx)
            images.append(p)
            mm = [round(v / ppm) for v in b]
            log.append(f"  close-up {k}: sheet mm x {mm[0]}..{mm[2]}, y {mm[1]}..{mm[3]}")
    return images


def process_file(src: Path, od: Path, rel: str, page: int = 0) -> dict:
    """Обработать один лист и записать все его файлы; page — страница PDF, с 0.

    Ошибка чтения или разбора листа превращается в red с причиной в log.txt.
    Наружу уходит только ошибка записи в саму папку результата (нет прав,
    файл занят): run.py повторяет такой лист отдельно и ставит ему red."""
    t0 = time.time()
    od.mkdir(parents=True, exist_ok=True)
    for name in OUTPUT_FILES:
        (od / name).unlink(missing_ok=True)
    for old in od.glob("preview_zoom_*.png"):
        old.unlink()
    log = [f"file: {rel}", "", "== program =="]
    metrics: dict = {}
    verdict: Verdict
    sheet = None
    failed = False
    src_sha = ""
    try:
        src_sha = sha1(src)
        sheet = load_sheet(src, page)
        res = extract(sheet)
        log += res.log
        metrics = res.metrics
        line_px = max(2, int(round(0.35 * res.scale.px_per_mm)))
        write_overlay(od / "overlay.png", sheet.gray, res.segments_px, res.dropped_segments_px, line_px)
        csha = write_contour_json(od / "contour.json", rel, res.segments_px, res.scale.px_per_mm, metrics)
        metrics["contour_sha1"] = csha
        log.append(f"contour fingerprint (sha1 of segments): {csha}")
        images = _model_images(od, sheet.gray, res, log)
        log.append("")
        log.append("== program notes (hints for the model, not a verdict) ==")
        verdict = decide(metrics, images, rel, csha, src_sha)
        log += [f"- {f}" for f in verdict.notes] or ["- nothing doubtful in metrics"]
    except Exception as e:
        failed = True
        tb = traceback.format_exc()
        log += ["", "== FAILED ==", f"{type(e).__name__}: {e}", tb]
        verdict = Verdict("red", f"лист не разобран: {type(e).__name__}: {e}",
                          "program: sheet not processed (model not asked)")
        try:
            write_empty_contour(od / "contour.json", rel, f"{type(e).__name__}: {e}")
        except Exception:
            pass
        try:
            # по ТЗ overlay нужен для каждого листа: чертёж без контура, если
            # его удалось прочитать, иначе картинка с надписью «не прочитан»
            if sheet is not None:
                write_overlay(od / "overlay.png", sheet.gray, [], [], 2)
            else:
                write_failure_image(od / "overlay.png", f"{type(e).__name__}: {e}")
        except Exception:
            pass
    log += ["", "== model =="]
    if verdict.model_request:
        log += ["request:", verdict.model_request, "", "response:", verdict.model_response or ""]
    else:
        log.append(f"not asked: {verdict.source}")
    log += ["", "== verdict ==", f"{verdict.verdict}: {verdict.reason}", f"decided by: {verdict.source}",
            f"time: {time.time() - t0:.1f}s"]
    (od / "log.txt").write_text("\n".join(log) + "\n", encoding="utf-8", newline="\n")
    (od / "verdict.txt").write_text(f"{verdict.verdict}\n{verdict.reason}\n", encoding="utf-8", newline="\n")
    summary = {"file": rel, "verdict": verdict.verdict, "reason": verdict.reason,
               "source": verdict.source, "metrics": metrics}
    # done.json отмечает результат, который не нужно переделывать; упавший лист
    # и вердикт, потерянный из-за временной ошибки модели, повторяются при следующем запуске
    if not failed and not verdict.source.startswith(RETRY_SOURCES):
        (od / DONE_FILE).write_text(json.dumps({"sha1": src_sha, "fingerprint": run_fingerprint(), **summary},
                                               ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    return summary


def write_crashed(od: Path, rel: str, err: BaseException) -> dict:
    """Файлы результата для листа, у которого упал сам процесс (кончилась
    память, сбой в библиотеке): red, пустой contour.json, без done.json
    (повторится при следующем запуске)."""
    od.mkdir(parents=True, exist_ok=True)
    reason = f"обработка упала: {type(err).__name__}: {err}"
    (od / "verdict.txt").write_text(f"red\n{reason}\n", encoding="utf-8", newline="\n")
    (od / "log.txt").write_text(f"file: {rel}\n\n== FAILED ==\nworker process died twice: {err!r}\n\n"
                                f"== verdict ==\nred: {reason}\ndecided by: program error\n", encoding="utf-8", newline="\n")
    write_empty_contour(od / "contour.json", rel, f"{type(err).__name__}: {err}")
    write_failure_image(od / "overlay.png", f"worker process died: {type(err).__name__}: {err}")
    return {"file": rel, "verdict": "red", "reason": reason, "source": "program error", "metrics": {}}


def weakness(s: dict) -> float:
    """Чем больше, тем слабее лист. Нужно только для выбора трёх самых слабых в отчёте:
    сначала цвет вердикта, затем доля непокрытых толстых линий, оборванные концы и число кусков."""
    rank = {"red": 2.0, "yellow": 1.0, "green": 0.0}.get(s.get("verdict"), 2.0)
    m = s.get("metrics") or {}
    cov = m.get("thick_ink_covered", 0.0)
    ends = m.get("open_ends", 0)
    comps = m.get("components_kept", 0)
    return rank * 10 + (1 - cov) * 3 + min(ends, 50) / 25 + min(comps, 60) / 60


def write_report(out_root: Path, summaries: list[dict]) -> Path:
    """report.md: число листов каждого цвета, кто ставил вердикт, три самых слабых, все листы."""
    counts = {v: sum(1 for s in summaries if s["verdict"] == v) for v in ("green", "yellow", "red")}
    weakest = sorted(summaries, key=lambda s: (-weakness(s), s["file"]))[:3]
    by_source: dict = {}
    for s in summaries:
        src = s["source"]
        k = src.split(" (")[0]
        if "offline review file" in src:
            k += " (офлайн-ревью из файла, не живой вызов API)"
        elif src.startswith("program:"):
            k = "программа (модель не спрашивали или она не ответила)"
        by_source[k] = by_source.get(k, 0) + 1
    lines = [
        "# Контур детали: отчёт по архиву", "",
        f"Листов: **{len(summaries)}**", "",
        "| вердикт | листов |", "|---|---|",
        f"| green | {counts['green']} |", f"| yellow | {counts['yellow']} |", f"| red | {counts['red']} |", "",
        "Кто ставил вердикт: " + ", ".join(f"{k}: {v}" for k, v in sorted(by_source.items())), "",
        "## Три самых слабых листа", "",
    ]
    for i, s in enumerate(weakest, 1):
        lines.append(f"{i}. `{s['file']}` — **{s['verdict']}**: {s['reason']}")
    lines += ["", "## Все листы", "", "| файл | вердикт | почему |", "|---|---|---|"]
    for s in sorted(summaries, key=lambda s: s["file"]):
        lines.append(f"| {s['file']} | {s['verdict']} | {s['reason'].replace('|', '/')} |")
    p = out_root / "report.md"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return p
