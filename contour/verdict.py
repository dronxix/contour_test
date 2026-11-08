"""Вердикт: green / yellow / red и одна фраза почему.

Вердикт ставит модель. Программа только:
  * сама ставит red, если лист вообще не удалось обработать (файл не
    читается, обработка упала, не найдено ни одной основной линии) — так
    требует ТЗ;
  * собирает по своим метрикам заметки о том, что выглядит сомнительно, и
    передаёт их модели как подсказки, а не как вердикт.

Все настройки берутся из окружения, обычно из файла .env (см. .env.example):
    VERDICT_MODE        auto | model | review | off
                        auto: файл ревью, если задан VERDICT_REVIEW_FILE,
                        иначе модель, если задан VERDICT_MODEL, иначе off
    VERDICT_API_TYPE    openai (по умолчанию: OpenAI и совместимые — LM Studio,
                        Ollama, vLLM, OpenRouter) | anthropic (Claude) | gemini
    VERDICT_API_BASE    по умолчанию для типа: http://localhost:1234/v1 (LM Studio),
                        https://api.anthropic.com, https://generativelanguage.googleapis.com
    VERDICT_MODEL       имя модели (нужна поддержка картинок)
    VERDICT_API_KEY     для локальных серверов openai-типа не нужен, для anthropic/gemini обязателен
    VERDICT_REASONING   по умолчанию "none": «думающая» модель должна сразу отвечать
    VERDICT_TIMEOUT     секунды, по умолчанию 300
    VERDICT_MAX_TOKENS  по умолчанию 1000
    VERDICT_PROMPT_FILE системный промпт, по умолчанию prompts/verdict_system.txt
    VERDICT_IMAGE_PX    длинная сторона картинки всего листа, по умолчанию 1024
    VERDICT_ZOOM        off | on | auto (по умолчанию auto): посылать ли крупные планы видов
    VERDICT_ZOOM_PX     длинная сторона крупного плана, по умолчанию 1024
    VERDICT_ZOOM_MAX    не больше стольких крупных планов, по умолчанию 6
    VERDICT_ZOOM_MIN_PX_PER_MM  при auto: зум, если на картинке всего листа на
                        1 мм листа меньше стольких пикселей, по умолчанию 5
    VERDICT_REVIEW_FILE файл офлайн-ревью
Если вердикт не получен, лист получает yellow: непроверенный контур —
сомнительный контур, green он не бывает.
"""
from __future__ import annotations

import base64
import hashlib
import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from .config import float_setting, int_setting, resolve, setting

VERDICTS = ("green", "yellow", "red")
DEFAULT_PROMPT_FILE = "prompts/verdict_system.txt"


@dataclass
class Verdict:
    verdict: str
    reason: str
    source: str                       # "model" / "program: ..." — кто поставил вердикт
    notes: list[str] = field(default_factory=list)
    model_request: str | None = None
    model_response: str | None = None


def mode() -> str:
    """VERDICT_MODE с раскрытым auto: review, model или off."""
    m = setting("VERDICT_MODE", "auto").lower()
    if m not in ("auto", "model", "review", "off"):
        raise ValueError(f"VERDICT_MODE must be auto|model|review|off, got {m!r}")
    if m == "auto":
        if setting("VERDICT_REVIEW_FILE"):
            return "review"
        return "model" if setting("VERDICT_MODEL") else "off"
    return m


def image_px() -> int:
    return int_setting("VERDICT_IMAGE_PX", 1024, 256, 4096)


def zoom_px() -> int:
    return int_setting("VERDICT_ZOOM_PX", 1024, 256, 4096)


def zoom_max() -> int:
    return int_setting("VERDICT_ZOOM_MAX", 6, 1, 20)


def max_tokens() -> int:
    # с запасом: короткий JSON-ответ плюс скрытые рассуждения «думающих» моделей
    return int_setting("VERDICT_MAX_TOKENS", 1000, 16, 32000)


def timeout_s() -> float:
    return float_setting("VERDICT_TIMEOUT", 300, 1, 3600)


def validate_settings() -> str:
    """Проверить все настройки вердикта один раз до обработки листов: опечатка
    останавливает запуск, а не делает все листы red. Возвращает режим."""
    md = mode()
    zoom_plan(1.0)
    image_px(), zoom_px(), zoom_max(), max_tokens(), timeout_s()
    kind = api_type()
    if md == "model":
        if not setting("VERDICT_MODEL"):
            raise ValueError("VERDICT_MODE=model but VERDICT_MODEL is empty")
        if kind in ("anthropic", "gemini") and not setting("VERDICT_API_KEY"):
            raise ValueError(f"VERDICT_API_TYPE={kind} needs VERDICT_API_KEY")
        system_prompt()
    if md == "review":
        p = setting("VERDICT_REVIEW_FILE")
        if not p:
            raise ValueError("VERDICT_MODE=review but VERDICT_REVIEW_FILE is empty")
        if not resolve(p).exists():
            raise ValueError(f"review file not found: {resolve(p)}")
    return md


def zoom_plan(ppm_overview: float) -> tuple[bool, str]:
    """(посылать ли крупные планы, почему) для листа, у которого на картинке
    всего листа ppm_overview пикселей на миллиметр листа."""
    z = setting("VERDICT_ZOOM", "auto").lower()
    if z in ("off", "0", "no", "false"):
        return False, "VERDICT_ZOOM=off"
    if z in ("on", "1", "yes", "true"):
        return True, "VERDICT_ZOOM=on"
    if z != "auto":
        raise ValueError(f"VERDICT_ZOOM must be off|on|auto, got {z!r}")
    lim = float_setting("VERDICT_ZOOM_MIN_PX_PER_MM", 5, 0, 100)
    if ppm_overview < lim:
        return True, f"auto: whole sheet is {ppm_overview:.1f} px/mm < {lim:g}, arrows and letters are too small"
    return False, f"auto: whole sheet is {ppm_overview:.1f} px/mm, readable without close-ups"


def program_notes(m: dict) -> tuple[str | None, list[str]]:
    """(причина red или None, список заметок).

    Причина red — лист не разобран: по ТЗ это red без вопроса модели.
    Заметки — подсказки модели, на что посмотреть внимательнее.
    """
    if m.get("segments", 0) == 0:
        return "программа не нашла на листе ни одной основной линии детали", []
    notes = []
    if m.get("contour_length_mm", 0) < 15:
        notes.append(f"контур очень короткий ({m['contour_length_mm']} мм)")
    if not m.get("line_weights_separable", True):
        notes.append("толщины основных и тонких линий на листе плохо разделяются, в контур мог попасть мусор")
    elif m.get("thick_share", 0) > 0.6:
        notes.append(f"толстыми признаны {m['thick_share'] * 100:.0f}% всех линий листа (обычно 20-50%): "
                     "размеры и текст могли попасть в контур")
    if not m.get("frame_found", True):
        notes.append("рамка не найдена, штамп и надписи могли попасть в контур")
    elif not m.get("title_block_found", True):
        notes.append("штамп не найден, масштаб в мм взят из DPI файла или по умолчанию и может быть неточным")
    if m.get("closed_loops", 0) == 0:
        notes.append("в контуре нет ни одного замкнутого очертания")
    if m.get("open_ends", 0) > max(6, 0.6 * max(m.get("closed_loops", 0), 1) + 4):
        notes.append(f"много оборванных концов линий ({m['open_ends']}): контур разорван или есть лишние штрихи")
    if m.get("largest_dropped_mm", 0) >= 30:
        notes.append(f"программа выбросила крупный кусок толстых линий ({m['largest_dropped_mm']:.0f} мм): "
                     "возможно, это выносной элемент или вид детали, который не обведён")
    if m.get("components_kept", 0) > 25:
        notes.append(f"контур раздроблен на {m['components_kept']} кусков, могли попасть текст или значки")
    if m.get("tables_removed", 0):
        notes.append(f"на листе есть таблица ({m['tables_removed']} шт.), программа её выбросила; проверь, что её линии не малиновые")
    return None, notes


def system_prompt() -> str:
    p = resolve(setting("VERDICT_PROMPT_FILE", DEFAULT_PROMPT_FILE))
    text = p.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"prompt file is empty: {p}")
    return text


PROMPT_METRICS = ("format", "segments", "contour_length_mm", "components_kept", "closed_loops", "open_ends",
                  "largest_dropped_mm", "tables_removed", "thick_share")


class ModelError(Exception):
    """Модель не ответила; request — что было отправлено (для log.txt)."""
    def __init__(self, msg: str, request: str):
        super().__init__(msg)
        self.request = request


def build_request_text(metrics: dict, notes: list[str], n_zoom: int = 0) -> str:
    """Текст запроса к модели: что на картинках, заметки и главные метрики программы."""
    short = {k: metrics[k] for k in PROMPT_METRICS if k in metrics}
    pics = "Картинка 1 — весь лист."
    if n_zoom:
        pics += (f" Картинки 2–{n_zoom + 1} — крупные планы видов детали с того же листа "
                 "(контур тоже малиновый): по ним проверяй стрелки, буквы и мелкие элементы.")
    return (pics +
            "\nЗаметки программы (подсказки, не вердикт): " + ("; ".join(notes) or "нет") +
            "\nМетрики программы: " + json.dumps(short, ensure_ascii=False) +
            "\nПосмотри на картинки и дай вердикт.")


def _sha(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def verdict_fingerprint() -> dict:
    """Всё, от чего зависит вердикт (но не контур): при изменении старые
    результаты устаревают, и листы оцениваются заново. Ключ API сюда не
    входит: на оценку он не влияет, а записывать его нельзя."""
    md = mode()
    fp: dict = {"mode": md}
    if md == "model":
        fp.update(api_type=api_type(), model=setting("VERDICT_MODEL"), api_base=api_base(),
                  reasoning=setting("VERDICT_REASONING", "none"), max_tokens=max_tokens(),
                  prompt_sha1=_sha(system_prompt().encode()), image_px=image_px(),
                  zoom=setting("VERDICT_ZOOM", "auto"), zoom_px=zoom_px(), zoom_max=zoom_max(),
                  zoom_min_ppm=setting("VERDICT_ZOOM_MIN_PX_PER_MM", "5"))
    elif md == "review":
        p = resolve(setting("VERDICT_REVIEW_FILE"))
        fp["review_sha1"] = _sha(p.read_bytes()) if p.exists() else None
    return fp


API_TYPES = ("openai", "anthropic", "gemini")
DEFAULT_BASE = {
    "openai": "http://localhost:1234/v1",            # LM Studio; для OpenAI: https://api.openai.com/v1
    "anthropic": "https://api.anthropic.com",
    "gemini": "https://generativelanguage.googleapis.com",
}
ANTHROPIC_VERSION = "2023-06-01"
# поля тела запроса с промптом и картинками (в лог как параметры не пишутся)
_HEAVY = ("messages", "system", "contents", "systemInstruction")


def api_type() -> str:
    """VERDICT_API_TYPE: в каком формате говорить с API.
    openai    — OpenAI Chat Completions и всё совместимое
                (OpenAI, LM Studio, Ollama, vLLM, OpenRouter, ...)
    anthropic — Anthropic Messages API (Claude)
    gemini    — Google Gemini generateContent"""
    t = setting("VERDICT_API_TYPE", "openai").lower()
    if t not in API_TYPES:
        raise ValueError(f"VERDICT_API_TYPE must be {'|'.join(API_TYPES)}, got {t!r}")
    return t


def api_base() -> str:
    """Адрес API: VERDICT_API_BASE или адрес по умолчанию для типа."""
    return setting("VERDICT_API_BASE", DEFAULT_BASE[api_type()]).rstrip("/")


def _request_log(url: str, sysp: str, user_text: str, images: list[Path], body: dict) -> str:
    """Что отправлено, для log.txt: всё, кроме ключа и байтов картинок
    (картинки названы по имени и SHA-1, сами файлы лежат рядом с логом)."""
    params = {k: v for k, v in body.items() if k not in _HEAVY}
    imgs = [f"  {p.name}  sha1 {_sha(p.read_bytes())[:12]}" for p in images]
    return "\n".join([f"api: {api_type()}", f"endpoint: {url}",
                      f"parameters: {json.dumps(params, ensure_ascii=False)}",
                      f"system prompt ({setting('VERDICT_PROMPT_FILE', DEFAULT_PROMPT_FILE)}, "
                      f"sha1 {_sha(sysp.encode())[:12]}):", sysp, "", "user text:", user_text, "",
                      f"images ({len(images)}):", *imgs])


def _build_request(kind: str, base: str, model: str, key: str, sysp: str, user_text: str,
                   images_b64: list[str]) -> tuple[str, dict, dict]:
    """(адрес, заголовки, тело) в формате выбранного API. Содержание у всех
    одно: системный промпт, текст, картинки (PNG), temperature 0."""
    reasoning = setting("VERDICT_REASONING", "none")
    if kind == "anthropic":
        content = [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b}}
                   for b in images_b64] + [{"type": "text", "text": user_text}]
        body = {"model": model, "max_tokens": max_tokens(), "temperature": 0, "system": sysp,
                "messages": [{"role": "user", "content": content}]}
        headers = {"x-api-key": key, "anthropic-version": ANTHROPIC_VERSION}
        return f"{base}/v1/messages", headers, body
    if kind == "gemini":
        parts = [{"text": user_text}] + [{"inline_data": {"mime_type": "image/png", "data": b}} for b in images_b64]
        gen: dict = {"temperature": 0, "maxOutputTokens": max_tokens()}
        if reasoning == "none":
            gen["thinkingConfig"] = {"thinkingBudget": 0}     # сразу отвечать, без «размышлений»
        body = {"systemInstruction": {"parts": [{"text": sysp}]},
                "contents": [{"role": "user", "parts": parts}], "generationConfig": gen}
        headers = {"x-goog-api-key": key}
        return f"{base}/v1beta/models/{model}:generateContent", headers, body
    content = [{"type": "text", "text": user_text}] + [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b}"}} for b in images_b64]
    body = {"model": model, "temperature": 0, "seed": 0, "max_tokens": max_tokens(),
            "messages": [{"role": "system", "content": sysp}, {"role": "user", "content": content}]}
    # «думающая» модель может потратить весь лимит на скрытые рассуждения, а нужен короткий вердикт
    if reasoning:
        body["reasoning_effort"] = reasoning
    headers = {"Authorization": f"Bearer {key or 'none'}"}
    return f"{base}/chat/completions", headers, body


def _answer_text(kind: str, resp: dict) -> str:
    """Текст ответа модели из JSON ответа API (без скрытых рассуждений)."""
    if kind == "anthropic":
        return "".join(b.get("text", "") for b in resp.get("content", []) if b.get("type") == "text")
    if kind == "gemini":
        cands = resp.get("candidates") or [{}]
        return "".join(p.get("text", "") for p in (cands[0].get("content") or {}).get("parts", [])
                       if not p.get("thought"))
    return resp["choices"][0]["message"].get("content") or ""


def _model_call(images: list[Path], metrics: dict, notes: list[str]) -> tuple[str, str]:
    """Отправить запрос модели; вернуть (запрос для лога, текст ответа).

    Ошибка 400 с названием неподдержанного параметра — параметр убирается и
    запрос повторяется; 429/5xx/529 — до трёх повторов с паузой. Любая другая
    ошибка — ModelError.
    """
    kind, base = api_type(), api_base()
    model = setting("VERDICT_MODEL")
    if not model:
        raise ModelError("VERDICT_MODEL is not set", "")
    key = setting("VERDICT_API_KEY")
    user_text = build_request_text(metrics, notes, len(images) - 1)
    try:
        sysp = system_prompt()
    except Exception as e:
        raise ModelError(f"prompt: {type(e).__name__}: {e}", user_text) from e
    images_b64 = [base64.b64encode(p.read_bytes()).decode() for p in images]
    url, headers, body = _build_request(kind, base, model, key, sysp, user_text, images_b64)
    headers["Content-Type"] = "application/json"
    notes_api: list[str] = []
    timeout = timeout_s()
    busy_retries = 0
    for _ in range(8):
        logged = _request_log(url, sysp, user_text, images, body)
        if notes_api:
            logged += "\nadjusted for this API: " + "; ".join(notes_api)
        req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                resp = json.loads(r.read().decode())
            return logged, _answer_text(kind, resp)
        except urllib.error.HTTPError as e:
            detail = _http_error_text(e)
            fix = _adapt_body(kind, body, detail) if e.code == 400 else None
            if fix:
                notes_api.append(fix)
                continue
            if e.code in (429, 500, 502, 503, 504, 529) and busy_retries < 3:
                busy_retries += 1
                time.sleep(5 * busy_retries)          # лимит запросов или перегрузка: подождать и повторить
                continue
            raise ModelError(f"HTTP {e.code}: {detail[:300]}", logged) from e
        except Exception as e:
            raise ModelError(f"{type(e).__name__}: {e}", logged) from e
    raise ModelError("API kept rejecting the request", logged)


def _http_error_text(e) -> str:
    """Текст ошибки из ответа API (поле error.message, если есть)."""
    try:
        raw = e.read().decode("utf-8", "replace")
        d = json.loads(raw)
        return str((d.get("error") or {}).get("message") or raw) if isinstance(d, dict) else raw
    except Exception:
        return str(e)


def _adapt_body(kind: str, body: dict, detail: str) -> str | None:
    """Модели одного API принимают разные необязательные параметры. Если API
    назвал параметр, который не принимает, он убирается (или переименовывается),
    чтобы один и тот же запрос работал с разными моделями. Возвращает, что
    изменено, или None."""
    msg = detail.lower()
    if kind == "gemini":
        gen = body.get("generationConfig", {})
        for p, word in (("thinkingConfig", "thinking"), ("temperature", "temperature")):
            if p in gen and word in msg:
                gen.pop(p)
                return f"{p} not supported, dropped"
        return None
    if kind == "anthropic":
        if "temperature" in body and "temperature" in msg:      # max_tokens там обязателен, его не убираем
            body.pop("temperature")
            return "temperature not supported, dropped"
        return None
    if "max_tokens" in msg and "max_tokens" in body and "max_completion_tokens" in msg:
        body["max_completion_tokens"] = body.pop("max_tokens")
        return "max_tokens -> max_completion_tokens"
    for p in ("reasoning_effort", "seed", "temperature", "max_tokens"):
        if p in body and p in msg:
            body.pop(p)
            return f"{p} not supported, dropped"
    return None


def _has_reason(reason: str) -> bool:
    """По ТЗ нужна фраза почему; одно слово-вердикт причиной не считается."""
    r = re.sub(r"[\W_]+", " ", reason.lower()).strip()
    return bool(r) and r not in VERDICTS and len(r) >= 8


def _parse(text: str) -> tuple[str, str] | None:
    """(вердикт, причина) или None. Ответ без причины не принимается: лист
    тогда остаётся yellow, это безопасная сторона."""
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            d = json.loads(m.group(0))
            v = str(d.get("verdict", "")).strip().lower()
            reason = str(d.get("reason", "")).strip()
            if v in VERDICTS:
                return (v, reason) if _has_reason(reason) else None
        except json.JSONDecodeError:
            pass
    # JSON нет: слово-вердикт принимается, только если в ответе назван ровно один
    # цвет («это не green, а yellow» называет два и не должно стать green)
    # и перед ним нет отрицания («не green», «not green» — это не вердикт green)
    low = text.lower()
    if re.search(r"\b(не|нет|not|no|non|isn't|isnt)\W{0,3}(green|yellow|red)\b", low):
        return None
    found = set(re.findall(r"\b(green|yellow|red)\b", low))
    if len(found) == 1:
        reason = text.strip().splitlines()[-1][:300]
        return (found.pop(), reason) if _has_reason(reason) else None
    return None


def _no_model(reason: str, notes: list[str], source: str, req=None, resp=None) -> Verdict:
    """yellow, когда вердикта модели нет; к причине добавляется первая заметка программы."""
    tail = f"; заметки программы: {notes[0]}" if notes else ""
    return Verdict("yellow", reason + tail, source, notes, req, resp)


_REVIEW_CACHE: dict = {}


def _load_review(path: str) -> dict:
    """Файл ревью: вердикты, поставленные моделью-ревьюером офлайн для конкретных контуров.

    {"reviewer": "...", "items": {"<файл>": {"contour_sha1", "source_sha1", "verdict", "reason"}}}
    """
    if path not in _REVIEW_CACHE:
        _REVIEW_CACHE[path] = json.loads(resolve(path).read_text(encoding="utf-8"))
    return _REVIEW_CACHE[path]


def _review_verdict(rel: str | None, csha: str | None, metrics: dict, notes: list[str],
                    src_sha: str | None = None) -> Verdict:
    """Вердикт из файла ревью; применяется, только если совпали отпечатки контура и файла чертежа."""
    path = setting("VERDICT_REVIEW_FILE")
    req = build_request_text(metrics, notes)
    if not path:
        return _no_model("вердикт модели не получен (файл ревью не задан), контур не проверен",
                         notes, "program: VERDICT_MODE=review but VERDICT_REVIEW_FILE is empty")
    try:
        data = _load_review(path)
    except Exception as e:
        return _no_model("вердикт модели не получен (файл ревью не прочитан), контур не проверен",
                         notes, f"program: review file error: {type(e).__name__}: {e}")
    reviewer = data.get("reviewer", "reviewer")
    item = data.get("items", {}).get(rel or "")
    if not item:
        return _no_model("вердикт модели не получен (лист не проходил ревью), контур не проверен",
                         notes, f"program: no review for this file in {path}")
    if item.get("contour_sha1") != csha:
        return _no_model("вердикт модели не получен (ревью было для другого контура), контур не проверен",
                         notes, f"program: review is for contour {item.get('contour_sha1')}, current is {csha}")
    # ревьюер смотрел именно этот чертёж; тот же контур на изменённом чертеже
    # (например, дорисован вид тонкими линиями) — уже не то, что проверялось
    if "source_sha1" not in item:
        return _no_model("вердикт модели не получен (ревью не привязано к файлу чертежа), контур не проверен",
                         notes, "program: review item has no source_sha1")
    if item["source_sha1"] != src_sha:
        return _no_model("вердикт модели не получен (ревью было для другой версии чертежа), контур не проверен",
                         notes, f"program: review is for file {item['source_sha1']}, current is {src_sha}")
    v = str(item.get("verdict", "")).lower()
    if v not in VERDICTS:
        return _no_model("ответ ревьюера не разобран, контур не проверен", notes, "program: bad review item")
    resp = json.dumps({"verdict": v, "reason": item.get("reason", "")}, ensure_ascii=False)
    return Verdict(v, item.get("reason", ""), f"model: {reviewer} (offline review file, not a live API call)",
                   notes, f"[same request as for a live model; image = overlay of this contour]\n{req}", resp)


def decide(metrics: dict, images: list[Path], rel: str | None = None, csha: str | None = None,
           src_sha: str | None = None) -> Verdict:
    """Вердикт листа. images: сначала картинка всего листа, потом крупные планы
    видов (может не быть). csha / src_sha: отпечатки контура и файла чертежа."""
    fatal, notes = program_notes(metrics)
    if fatal:
        return Verdict("red", fatal, "program: sheet not processed (model not asked)", notes)
    md = mode()
    if md == "review":
        return _review_verdict(rel, csha, metrics, notes, src_sha)
    if md == "off" or not images:
        return _no_model("вердикт модели не получен (модель не настроена), контур не проверен",
                         notes, "program: no model configured (VERDICT_MODE=off or VERDICT_MODEL empty)")
    try:
        req, resp = _model_call(images, metrics, notes)
    except ModelError as e:
        return _no_model("вердикт модели не получен (модель не ответила), контур не проверен",
                         notes, f"program: model call failed: {e}", e.request, "(no answer)")
    parsed = _parse(resp)
    if not parsed:
        return _no_model("ответ модели не разобран или без причины, контур не проверен",
                         notes, "program: model answer unparseable", req, resp)
    v, reason = parsed
    return Verdict(v, reason, "model", notes, req, resp)
