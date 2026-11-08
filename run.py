"""Контур детали на каждом чертеже в папке.

    python run.py <папка с чертежами> <папка результата> [--workers N] [--force] [--env FILE]

Обрабатывает каждый PDF/JPG/PNG/TIFF/BMP в папке (рекурсивно), по папке
результата на лист (каждая страница многостраничного PDF — отдельный лист),
затем пишет report.md рядом с результатами. Уже обработанные листы (тот же
файл, код и настройки, полный результат) пропускаются, если не задан --force.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

from contour.config import DEFAULT_ENV, load_env, setting
from contour.pipeline import DONE_FILE, find_inputs, is_done, plan_jobs, process_file, write_crashed, write_report
from contour.sheet import deskew_mode, render_dpi
from contour.verdict import api_base, api_type, validate_settings


def run_scope() -> str:
    """RUN_SCOPE: new (по умолчанию) — только листы без актуального результата;
    all — всё с нуля (как --force)."""
    v = setting("RUN_SCOPE", "new").lower()
    if v not in ("new", "all"):
        raise ValueError(f"RUN_SCOPE must be new|all, got {v!r}")
    return v


def _job(args):
    src, od, rel, page = args
    return process_file(Path(src), Path(od), rel, page)


def _run_pool(jobs, workers, report) -> list:
    """Выполнить листы в процессах-исполнителях; вернуть [(лист, исключение)]
    для тех, чей процесс погиб."""
    failed = []
    with ProcessPoolExecutor(max_workers=max(1, min(workers, len(jobs)))) as ex:
        futs = {ex.submit(_job, j): j for j in jobs}
        for f in as_completed(futs):
            try:
                report(f.result())
            except Exception as e:
                failed.append((futs[f], e))
    return failed


def main() -> int:
    """Разобрать аргументы, проверить настройки, обработать листы, записать report.md."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    ap.add_argument("--force", action="store_true",
                    help="re-process everything (same as RUN_SCOPE=all in .env)")
    ap.add_argument("--env", type=Path, default=None,
                    help="settings file (default: .env in the project folder, if present)")
    a = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    # настройки должны попасть в os.environ до запуска процессов-исполнителей
    try:
        taken = load_env(a.env)
        dpi = render_dpi()
        dsk = deskew_mode()
        md = validate_settings()
        scope = run_scope()
    except Exception as e:
        print(f"bad settings: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    if a.force:
        scope = "all"
    src = a.env or (DEFAULT_ENV if DEFAULT_ENV.exists() else None)
    print(f"settings: {src or 'environment only'} ({len(taken)} keys from file); dpi {dpi:g}; "
          f"deskew {dsk}; run scope: {scope}{' (--force)' if a.force else ''}; verdict mode: {md}"
          + (f", {api_type()} API, model {setting('VERDICT_MODEL')} at {api_base()}" if md == "model" else "")
          + (f", review file {setting('VERDICT_REVIEW_FILE')}" if md == "review" else "")
          + f"; zoom {setting('VERDICT_ZOOM', 'auto')}")
    if not a.input.is_dir():
        print(f"input folder not found: {a.input}", file=sys.stderr)
        return 2
    a.output.mkdir(parents=True, exist_ok=True)

    inputs = find_inputs(a.input, exclude=a.output)
    sheets = plan_jobs(inputs, a.input, a.output)
    extra = len(sheets) - len(inputs)
    print(f"found {len(inputs)} drawings in {a.input}"
          + (f" ({len(sheets)} sheets: multi-page PDFs give one sheet per page)" if extra else ""))
    jobs, summaries = [], []
    for src, od, rel, page in sheets:
        if scope == "new" and is_done(src, od):
            d = json.loads((od / DONE_FILE).read_text(encoding="utf-8"))
            d.pop("sha1", None)
            d.pop("fingerprint", None)
            summaries.append(d)
            continue
        jobs.append((str(src), str(od), rel, page))
    if len(jobs) < len(sheets):
        print(f"skipping {len(sheets) - len(jobs)} already processed and unchanged; "
              f"processing {len(jobs)} new or changed (RUN_SCOPE=all or --force redoes everything)")

    n = len(jobs)
    done = 0

    def report(s):
        nonlocal done
        done += 1
        summaries.append(s)
        print(f"[{done}/{n}] {s['verdict']:6} {s['file']}")

    # ошибки разбора листа process_file сам превращает в red. Сюда доходят сбой
    # процесса-исполнителя (кончилась память, сбой в библиотеке) и ошибка записи
    # в папку результата. Сбой процесса ломает весь пул, поэтому каждый
    # недоделанный лист повторяется отдельно в своём процессе: red получает
    # только тот лист, который падает и при повторе.
    failed = _run_pool(jobs, a.workers, report) if n else []
    if failed:
        print(f"{len(failed)} sheet(s) lost with a dead worker process, retrying each in its own process")
        lock = threading.Lock()

        def locked(s):
            with lock:
                report(s)

        def retry(j):
            again = _run_pool([j], 1, locked)
            if again:
                err = again[0][1]
                try:
                    s = write_crashed(Path(j[1]), j[2], err)
                except Exception as e:
                    # даже файлы red записать не удалось: лист остаётся только в отчёте
                    s = {"file": j[2], "verdict": "red", "reason": f"обработка упала: {type(err).__name__}: {err}; "
                         f"результат не записан: {type(e).__name__}: {e}", "source": "program error", "metrics": {}}
                locked(s)

        with ThreadPoolExecutor(max_workers=max(1, a.workers)) as tx:
            list(tx.map(retry, [j for j, _ in failed]))

    rep = write_report(a.output, summaries)
    c = {v: sum(1 for s in summaries if s["verdict"] == v) for v in ("green", "yellow", "red")}
    print(f"done: green {c['green']}, yellow {c['yellow']}, red {c['red']} -> {rep}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
