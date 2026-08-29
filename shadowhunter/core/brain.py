"""LLM copilot: a decision brain and a bilingual human reporter.

Honesty rules live in code (see ``models.honesty``), not only in the prompt.
A silent Ollama must still produce a structured verdict - the operator never
waits on a model that is not running.
"""
from __future__ import annotations

import base64
import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any

from .cache import hash_key, instance as cache
from .config import SETTINGS
from .logging import get_logger
from ..models.honesty import imagery_honesty

log = get_logger(__name__)

_JSON_FENCE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)
_THINK_FENCE = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)

_TAGS_TTL_S = 20.0
_tags_memo: tuple[float, list[str]] = (0.0, [])

_SYSTEM = (
    "You are Shadow Hunter's operator copilot. You help a human read satellite "
    "shadows. Never claim centimetre accuracy on undated basemap mosaics "
    "(Esri/OSM/Carto). Heights there are INDICATIVE. Never invent acquisition "
    "dates. Persian reports must be professional, precise, and free of hype. "
    "Reply with a single JSON object: verdict, warnings, actions "
    "(list of {id,label,why}), report_fa, report_en."
)

_VERDICTS = frozenset({
    "indicative", "measurable", "poor_geometry", "night", "undated_basemap",
})


def _base() -> str:
    return (os.getenv("SH_LLM_URL") or SETTINGS.llm_url).rstrip("/")


def _model() -> str:
    return (os.getenv("SH_LLM_MODEL") or SETTINGS.llm_model or "qwen3.5:4b").strip()


def _embed_model() -> str:
    return (os.getenv("SH_EMBED_MODEL") or SETTINGS.embed_model
            or "mxbai-embed-large:latest").strip()


def _vlm_model() -> str:
    return (os.getenv("SH_VLM_MODEL") or SETTINGS.vlm_model or "glm-ocr:latest").strip()


def _key() -> str:
    return (os.getenv("SH_LLM_KEY") or SETTINGS.llm_key or "").strip()


def _ollama_root(base: str) -> str:
    return base[:-3] if base.endswith("/v1") else base


def _headers() -> dict[str, str]:
    hdrs = {"Content-Type": "application/json", "Accept": "application/json"}
    key = _key()
    if key:
        hdrs["Authorization"] = f"Bearer {key}"
    return hdrs


def _http_json(url: str, *, payload: dict[str, Any] | None = None,
               timeout: float = 2.0) -> Any | None:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    method = "POST" if payload is not None else "GET"
    request = urllib.request.Request(url, data=data, method=method, headers=_headers())
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        log.debug("llm request failed %s: %s", url, exc)
        return None
    if not body:
        return {}
    try:
        return json.loads(body.decode("utf-8", "replace"))
    except json.JSONDecodeError:
        return None


def _strip_think(text: str) -> str:
    return _THINK_FENCE.sub("", text or "").strip()


def _model_present(want: str, names: list[str] | None = None) -> bool:
    needle = (want or "").strip().lower()
    if not needle:
        return False
    stem = needle.split(":")[0]
    hay = names if names is not None else list_local_models()
    for name in hay:
        low = str(name).lower()
        if needle == low or needle in low or stem in low:
            return True
    return False


def configured_models() -> dict[str, Any]:
    names = list_local_models()
    llm, embed, vlm = _model(), _embed_model(), _vlm_model()
    return {
        "llm": llm,
        "embed": embed,
        "vlm": vlm,
        "llm_ready": _model_present(llm, names),
        "embed_ready": _model_present(embed, names),
        "vlm_ready": _model_present(vlm, names),
        "host": _base(),
    }


def list_local_models() -> list[str]:
    """Names Ollama (or an OpenAI-compat host) currently has. Empty if down."""
    global _tags_memo
    now = time.monotonic()
    stamp, cached = _tags_memo
    if cached and (now - stamp) < _TAGS_TTL_S:
        return list(cached)
    base = _base()
    names: list[str] = []
    tags = _http_json(f"{_ollama_root(base)}/api/tags", timeout=2.0)
    if isinstance(tags, dict):
        for item in tags.get("models") or []:
            if isinstance(item, dict):
                name = item.get("name") or item.get("model") or item.get("id")
            else:
                name = str(item)
            if name:
                names.append(str(name))
    if names:
        _tags_memo = (time.monotonic(), names)
        return names
    listing = _http_json(f"{base}/models", timeout=2.0)
    if isinstance(listing, dict):
        for item in listing.get("data") or listing.get("models") or []:
            if isinstance(item, dict):
                ident = item.get("id") or item.get("name")
            else:
                ident = str(item)
            if ident:
                names.append(str(ident))
    _tags_memo = (time.monotonic(), names)
    return names


def llm_up() -> bool:
    """True if the local OpenAI-compat host answered, even with no models."""
    base = _base()
    tags = _http_json(f"{_ollama_root(base)}/api/tags", timeout=2.0)
    if isinstance(tags, dict):
        return True
    listing = _http_json(f"{base}/models", timeout=2.0)
    return isinstance(listing, dict)


def _completion_text(raw: Any) -> str:
    if not isinstance(raw, dict):
        return ""
    msg = raw.get("message")
    if isinstance(msg, dict):
        content = msg.get("content")
        if isinstance(content, str) and content.strip():
            return _strip_think(content)
    try:
        content = raw["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return ""
    return _strip_think(content) if isinstance(content, str) else ""


def complete(messages: list[dict[str, str]], *, tools_context: dict[str, Any] | None = None,
             timeout: float = 90.0) -> str:
    """Chat completion via local Ollama (qwen3.5:4b). Empty string on failure."""
    msgs = list(messages)
    if tools_context:
        blob = json.dumps(tools_context, ensure_ascii=False, default=str)
        if len(blob) > 12_000:
            blob = blob[:12_000] + "…"
        msgs = [{"role": "system", "content": f"tools_context JSON:\n{blob}"}] + msgs
    model = _model()
    root = _ollama_root(_base())
    raw = _http_json(
        f"{root}/api/chat",
        payload={
            "model": model,
            "messages": msgs,
            "stream": False,
            "think": False,
            "options": {"temperature": 0.3},
        },
        timeout=timeout,
    )
    text = _completion_text(raw)
    if text:
        return text
    raw = _http_json(
        f"{_base()}/chat/completions",
        payload={"model": model, "messages": msgs, "temperature": 0.3},
        timeout=timeout,
    )
    return _completion_text(raw)


def embed(texts: list[str], *, timeout: float = 30.0) -> list[list[float]] | None:
    """mxbai-embed-large vectors. None if the embed model is down."""
    cleaned = [str(t).strip() for t in texts if str(t).strip()]
    if not cleaned:
        return []
    root = _ollama_root(_base())
    model = _embed_model()
    raw = _http_json(
        f"{root}/api/embed",
        payload={"model": model, "input": cleaned},
        timeout=timeout,
    )
    vectors = None
    if isinstance(raw, dict):
        vectors = raw.get("embeddings") or raw.get("embedding")
    if vectors is None:
        raw = _http_json(
            f"{_base()}/embeddings",
            payload={"model": model, "input": cleaned},
            timeout=timeout,
        )
        if isinstance(raw, dict):
            data = raw.get("data") or []
            vectors = [row.get("embedding") for row in data if isinstance(row, dict)]
    if isinstance(vectors, list) and vectors and isinstance(vectors[0], (int, float)):
        vectors = [vectors]
    if not isinstance(vectors, list) or not vectors:
        return None
    out: list[list[float]] = []
    for row in vectors:
        if not isinstance(row, list):
            return None
        try:
            out.append([float(x) for x in row])
        except (TypeError, ValueError):
            return None
    return out


def _cosine(a: list[float], b: list[float]) -> float:
    n = min(len(a), len(b))
    if n < 1:
        return 0.0
    dot = s1 = s2 = 0.0
    for i in range(n):
        x, y = a[i], b[i]
        dot += x * y
        s1 += x * x
        s2 += y * y
    denom = math.sqrt(s1) * math.sqrt(s2)
    return float(dot / denom) if denom > 1e-12 else 0.0


def _jpeg_b64(bgr: Any, max_side: int = 512, quality: int = 80) -> str | None:
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None
    img = np.ascontiguousarray(bgr)
    if img.size == 0:
        return None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = img[:, :, :3]
    h, w = img.shape[:2]
    scale = min(1.0, float(max_side) / float(max(h, w, 1)))
    if scale < 0.999:
        img = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                         interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return None
    return base64.b64encode(buf.tobytes()).decode("ascii")


def see_image(bgr: Any, *, cache_key: str | None = None,
              prompt: str | None = None, timeout: float = 120.0) -> dict[str, Any]:
    """glm-ocr read of a mosaic. Fail-soft; never invents an acquisition date."""
    out: dict[str, Any] = {
        "text": "", "model": _vlm_model(), "used_vlm": False,
    }
    if bgr is None:
        return out
    key = hash_key("vlm", _vlm_model(), cache_key or "anon")
    hit = cache().get("vlm", key)
    if isinstance(hit, dict) and hit.get("text"):
        return {**hit, "cached": True}
    jpeg = _jpeg_b64(bgr)
    if not jpeg:
        return out
    question = prompt or (
        "Describe the aerial/satellite scene. Read any visible text or signs. "
        "Name the likely landmark if it is obvious. List compact structures and "
        "shadow direction. Do not invent an acquisition date. Do not claim "
        "cadastral metres. Reply in Persian, then a short English line."
    )
    root = _ollama_root(_base())
    raw = _http_json(
        f"{root}/api/chat",
        payload={
            "model": _vlm_model(),
            "messages": [{"role": "user", "content": question, "images": [jpeg]}],
            "stream": False,
            "think": False,
        },
        timeout=timeout,
    )
    text = _completion_text(raw)
    if not text:
        data_url = f"data:image/jpeg;base64,{jpeg}"
        raw = _http_json(
            f"{_base()}/chat/completions",
            payload={
                "model": _vlm_model(),
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }],
                "temperature": 0.2,
            },
            timeout=timeout,
        )
        text = _completion_text(raw)
    if not text:
        return out
    out["text"] = text[:4000]
    out["used_vlm"] = True
    cache().set("vlm", key, {"text": out["text"], "model": out["model"],
                             "used_vlm": True}, ttl_s=3600)
    return out


def rerank_intel(intel: dict[str, Any] | None, query: str,
                 extra: str | None = None) -> dict[str, Any] | None:
    """Re-order Wikipedia hits with mxbai-embed-large. Unchanged if embed fails."""
    if not isinstance(intel, dict):
        return intel
    wiki = [dict(w) for w in (intel.get("wikipedia") or []) if isinstance(w, dict)]
    q = " ".join(part for part in (query, extra) if part).strip()
    if len(wiki) < 2 or not q:
        return intel
    docs = [f"{w.get('title') or ''} {w.get('summary') or ''}".strip() or "?" for w in wiki]
    vectors = embed([q] + docs)
    if not vectors or len(vectors) != len(docs) + 1:
        return intel
    qv = vectors[0]
    scored = []
    for item, vec in zip(wiki, vectors[1:]):
        item["embed_score"] = round(_cosine(qv, vec), 4)
        scored.append(item)
    scored.sort(key=lambda row: -float(row.get("embed_score") or 0))
    out = dict(intel)
    out["wikipedia"] = scored
    out["embed_query"] = q[:240]
    out["embed_model"] = _embed_model()
    return out


# --------------------------------------------------------------------------- #
# Heuristic brief
# --------------------------------------------------------------------------- #
def _scene(payload: dict[str, Any]) -> dict[str, Any]:
    scene = payload.get("scene")
    return scene if isinstance(scene, dict) else {}


def _sun(payload: dict[str, Any]) -> dict[str, Any]:
    sun = payload.get("sun")
    return sun if isinstance(sun, dict) else {}


def _is_daylight(payload: dict[str, Any]) -> bool:
    sun = _sun(payload)
    flag = sun.get("is_daylight")
    if flag is None:
        try:
            return float(sun.get("elevation_deg", 90.0)) > 0.0
        except (TypeError, ValueError):
            return True
    return bool(flag)


def _quality(payload: dict[str, Any]) -> float:
    try:
        return float(_sun(payload).get("quality", 1.0))
    except (TypeError, ValueError):
        return 1.0


def _honesty_block(survey: dict[str, Any]) -> dict[str, Any]:
    scene = _scene(survey)
    return imagery_honesty(
        provider=str(scene.get("provider") or "esri"),
        source=str(scene.get("source") or "basemap"),
        is_daylight=_is_daylight(survey),
        quality=_quality(survey),
        sun_estimate=survey.get("sun_estimate") if isinstance(survey.get("sun_estimate"), dict) else None,
    )


def _fmt_pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:.0f}%"
    except (TypeError, ValueError):
        return "—"


def _fmt_num(value: Any, digits: int = 1) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def _best_when(timeline: Any) -> str | None:
    if not isinstance(timeline, dict):
        return None
    for key in ("best_when", "best_hour", "best"):
        val = timeline.get(key)
        if val:
            return str(val)
    return None


def _place_name(survey: dict[str, Any], intel: Any) -> str:
    if isinstance(intel, dict):
        reverse = intel.get("reverse") or intel.get("place") or {}
        if isinstance(reverse, dict) and reverse.get("name"):
            return str(reverse["name"])
    return str(_scene(survey).get("name") or "AOI")


def _heuristic_reports(survey: dict[str, Any], *, inspect: Any, timeline: Any,
                       intel: Any, measures: Any, honesty: dict[str, Any],
                       verdict: str) -> tuple[str, str]:
    count = survey.get("count")
    if count is None:
        count = len(survey.get("structures") or [])
    with_shadow = survey.get("with_shadow", "—")
    shadow = _fmt_pct(survey.get("shadow_coverage"))
    water = _fmt_pct(survey.get("water_coverage"))
    elev = _fmt_num(_sun(survey).get("elevation_deg"), 1)
    quality = _fmt_num(_sun(survey).get("quality"), 2)
    best = _best_when(timeline)
    place = _place_name(survey, intel)
    n_meas = len(measures) if isinstance(measures, list) else (
        len((measures or {}).get("items") or []) if isinstance(measures, dict) else 0
    )
    osm_n = 0
    if isinstance(intel, dict):
        osm_n = int((intel.get("overpass") or intel.get("buildings") or {}).get("count") or 0)

    fa = [
        f"{place}: در این پهنه {count} سازه شناسایی شد که {with_shadow} مورد سایهٔ قابل استفاده دارند.",
        f"پوشش سایه حدود {shadow} و پوشش آب {water} است؛ ارتفاع خورشید {elev} درجه (کیفیت هندسه {quality}) است.",
    ]
    en = [
        f"{place}: {count} structures were counted; {with_shadow} carry a usable shadow.",
        f"Shadow coverage is about {shadow} and water about {water}; solar elevation is {elev}° (geometry quality {quality}).",
    ]
    if honesty.get("height_indicative"):
        fa.append("مترهای گزارش‌شده روی موزاییک بدون تاریخ، تقریبی (indicative) هستند نه نقشه‌برداری قطعی.")
        en.append("Reported metres on this undated mosaic are INDICATIVE, not a cadastral measurement.")
    if verdict == "night":
        fa.append("چون خورشید زیر افق است، هر ارتفاعی از سایه در این لحظه نامعتبر است.")
        en.append("Because the sun is below the horizon, any shadow-derived height at this instant is invalid.")
    if osm_n:
        fa.append(f"{osm_n} ساختمان در OSM برچسب خورده است.")
        en.append(f"{osm_n} buildings are tagged in OSM.")
    if n_meas:
        fa.append(f"{n_meas} سازه تا این لحظه اندازه‌گیری شده است.")
        en.append(f"{n_meas} structures have been measured so far.")
    fa.append("توصیه می‌شود سازه‌های با بالاترین امتیاز که سایه دارند اندازه‌گیری شوند.")
    en.append("Measure the highest-score structures that still show a clean shadow.")
    if best:
        fa.append(f"بهترین ساعت تصویربرداری در این روز: {best}.")
        en.append(f"Best acquisition hour on this day: {best}.")
    else:
        fa.append("برای هندسه بهتر، ساعت اوج کیفیت سایه را از خط‌زمان خورشید انتخاب کنید.")
        en.append("For better geometry, pick the peak-quality hour from the solar timeline.")
    if isinstance(inspect, dict) and inspect.get("width"):
        fa.append(
            f"رستر {inspect.get('width')}×{inspect.get('height')} "
            f"{inspect.get('layout', 'BGR')} {inspect.get('bit_depth', 8)}-bit."
        )
        en.append(
            f"Raster is {inspect.get('width')}×{inspect.get('height')} "
            f"{inspect.get('layout', 'BGR')} {inspect.get('bit_depth', 8)}-bit."
        )
    season = (_sun(survey).get("season") or survey.get("season") or "")
    if season:
        fa.append(f"ساعت خورشید روی فصل «{season}» قفل شد (بهترین هندسهٔ سایه در سال).")
        en.append(f"Solar clock locked to the «{season}» marker (best shadow geometry of the year).")
    centre = _scene(survey).get("center")
    if isinstance(centre, (list, tuple)) and len(centre) >= 2:
        fa.append(f"مختصات مرکز: {float(centre[1]):.5f}، {float(centre[0]):.5f}.")
        en.append(f"AOI centre: {float(centre[1]):.5f}, {float(centre[0]):.5f}.")
    wiki = []
    if isinstance(intel, dict):
        wiki = [w.get("title") for w in (intel.get("wikipedia") or [])[:3] if isinstance(w, dict)]
    if wiki:
        fa.append("هویت پیشنهادی: " + "، ".join(str(t) for t in wiki if t) + ".")
        en.append("Candidate identity: " + ", ".join(str(t) for t in wiki if t) + ".")
    return " ".join(fa[:10]), " ".join(en[:10])


def _heuristic_actions(survey: dict[str, Any], *, timeline: Any, locale: str,
                       verdict: str) -> list[dict[str, str]]:
    fa = locale != "en"
    best = _best_when(timeline)
    actions = [{
        "id": "measure_top",
        "label": "اندازه‌گیری سازه‌های پرسایه با بالاترین امتیاز" if fa
        else "Measure highest-score shadowed structures",
        "why": "تمیزترین طول سایه، کم‌خطاترین h = L·tan(θ) را می‌دهد." if fa
        else "The cleanest shadow length gives the least-wrong h = L·tan(θ).",
    }]
    if best:
        actions.append({
            "id": "best_hour",
            "label": f"تصویربرداری نزدیک {best}" if fa else f"Re-acquire near {best}",
            "why": "کیفیت هندسه سایه در آن ساعت بیشینه است." if fa
            else "Shadow-geometry quality peaks at that hour.",
        })
    actions.append({
        "id": "scene3d",
        "label": "ساخت نمای سه‌بعدی" if fa else "Build a 3D field view",
        "why": "بیرون‌زدن ردپاها خوانش فضایی بلوک را آسان می‌کند." if fa
        else "Extrude footprints for a spatial read of the block.",
    })
    if verdict == "night":
        actions.append({
            "id": "wait_daylight",
            "label": "صبر تا روز" if fa else "Wait for daylight",
            "why": "سایهٔ شب وجود ندارد؛ ارتفاع از سایه تعریف نمی‌شود." if fa
            else "There is no cast shadow at night; height from shadow is undefined.",
        })
    if verdict in {"undated_basemap", "indicative"}:
        actions.append({
            "id": "dated_scene",
            "label": "در صورت امکان صحنهٔ تاریخ‌دار" if fa else "Prefer a dated scene if available",
            "why": "فقط با زمان برداشت می‌توان elevation را به این پیکسل‌ها نسبت داد." if fa
            else "Only a known acquisition time lets elevation be trusted on these pixels.",
        })
    return actions


def _extract_json(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    blob = _strip_think(text)
    fenced = _JSON_FENCE.search(blob)
    blob = fenced.group(1) if fenced else blob
    start, end = blob.find("{"), blob.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        parsed = json.loads(blob[start:end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _compact_context(survey: dict[str, Any], *, inspect: Any, timeline: Any,
                     intel: Any, measures: Any) -> dict[str, Any]:
    skip = {"image_png", "overlay_png", "crop_png", "strip_png", "depth_png", "preview_png"}
    structures = survey.get("structures") or []
    top = sorted(structures, key=lambda s: -float(s.get("score") or 0))[:12] if structures else []
    slim = []
    for item in top:
        slim.append({k: item[k] for k in (
            "score", "quick_height_m", "shadow_len_px", "shadow_support",
            "width_m", "height_m", "box", "center",
        ) if k in item})
    ctx: dict[str, Any] = {
        "aoi_id": survey.get("aoi_id"),
        "scene": {k: v for k, v in _scene(survey).items() if k not in skip},
        "sun": _sun(survey),
        "sun_estimate": survey.get("sun_estimate") or {},
        "count": survey.get("count"),
        "with_shadow": survey.get("with_shadow"),
        "mean_score": survey.get("mean_score"),
        "shadow_coverage": survey.get("shadow_coverage"),
        "water_coverage": survey.get("water_coverage"),
        "structures_top": slim,
    }
    if isinstance(inspect, dict):
        ctx["inspect"] = {k: v for k, v in inspect.items()
                          if not str(k).startswith("_") and k not in skip}
    if isinstance(timeline, dict):
        ctx["timeline_best"] = timeline.get("best_when")
        ctx["best_elevation_deg"] = (
            timeline.get("best_elevation_deg") or timeline.get("best_elevation")
        )
    if isinstance(intel, dict):
        overpass = intel.get("overpass") or intel.get("buildings") or {}
        ctx["intel"] = {
            "place": (intel.get("reverse") or intel.get("place") or {}).get("name")
            if isinstance(intel.get("reverse") or intel.get("place"), dict) else None,
            "osm_buildings": overpass.get("count") if isinstance(overpass, dict) else None,
            "wiki": [w.get("title") for w in (intel.get("wikipedia") or [])[:4]
                     if isinstance(w, dict)],
        }
    if isinstance(measures, dict):
        ctx["measures"] = {k: v for k, v in measures.items() if k not in skip}
        items = measures.get("items")
        if isinstance(items, list):
            ctx["measures"] = {**ctx["measures"], "items": items[:12]}
    elif isinstance(measures, list):
        ctx["measures"] = measures[:12]
    return ctx


def decide(survey_payload: dict[str, Any] | None, *, inspect: Any = None,
           timeline: Any = None, intel: Any = None, measures: Any = None,
           locale: str = "fa") -> dict[str, Any]:
    """Structured operator brief. Always returns a verdict, LLM or not."""
    survey = survey_payload if isinstance(survey_payload, dict) else {}
    honesty_full = _honesty_block(survey)
    honesty = {
        "imagery_dated": honesty_full["imagery_dated"],
        "sun_source": honesty_full["sun_source"],
        "elevation_trusted": honesty_full["elevation_trusted"],
        "height_indicative": honesty_full["height_indicative"],
    }
    verdict = honesty_full["verdict"]
    warnings = list(honesty_full["warnings"])
    report_fa, report_en = _heuristic_reports(
        survey, inspect=inspect, timeline=timeline, intel=intel,
        measures=measures, honesty=honesty, verdict=verdict,
    )
    actions = _heuristic_actions(survey, timeline=timeline, locale=locale, verdict=verdict)

    dossier = _dossier(survey, inspect=inspect, timeline=timeline,
                       intel=intel, honesty=honesty)

    key = hash_key("brief", survey.get("aoi_id"), locale,
                   len(measures) if isinstance(measures, list) else 0, verdict,
                   _sun(survey).get("when"), _sun(survey).get("season"),
                   survey.get("count"))
    cached = cache().get("brain", key)
    if isinstance(cached, dict) and cached.get("verdict"):
        cached.setdefault("dossier", dossier)
        return cached

    up = llm_up()
    out: dict[str, Any] = {
        "provider": "ollama" if up else "offline",
        "model": _model() if up else None,
        "verdict": verdict,
        "honesty": honesty,
        "warnings": warnings,
        "actions": actions,
        "report_fa": report_fa,
        "report_en": report_en,
        "used_llm": False,
        "dossier": dossier,
    }

    if not up:
        cache().set("brain", key, out, ttl_s=3600)
        return out

    ctx = _compact_context(survey, inspect=inspect, timeline=timeline,
                           intel=intel, measures=measures)
    user = (
        f"locale={locale}. Return JSON only. Keep verdict as one of: "
        + ", ".join(sorted(_VERDICTS))
        + ". Do not contradict the honesty block. Context (no images):\n"
        + json.dumps(ctx, ensure_ascii=False, default=str)[:8000]
    )
    raw = complete(
        [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": user}],
        tools_context={"honesty": honesty, "verdict_heuristic": verdict},
        timeout=90.0,
    )
    if not raw:
        cache().set("brain", key, out, ttl_s=3600)
        return out

    parsed = _extract_json(raw)
    if parsed is None:
        out["llm_raw"] = raw[:4000]
        cache().set("brain", key, out, ttl_s=3600)
        return out

    out["used_llm"] = True
    if isinstance(parsed.get("report_fa"), str) and parsed["report_fa"].strip():
        out["report_fa"] = parsed["report_fa"].strip()
    if isinstance(parsed.get("report_en"), str) and parsed["report_en"].strip():
        out["report_en"] = parsed["report_en"].strip()
    extra_warn = parsed.get("warnings")
    if isinstance(extra_warn, list):
        for item in extra_warn:
            text = str(item).strip()
            if text and text not in out["warnings"]:
                out["warnings"].append(text)
    extra_actions = parsed.get("actions")
    if isinstance(extra_actions, list) and extra_actions:
        cleaned: list[dict[str, str]] = []
        for item in extra_actions:
            if not isinstance(item, dict):
                continue
            ident = str(item.get("id") or "action")
            cleaned.append({
                "id": ident,
                "label": str(item.get("label") or ident),
                "why": str(item.get("why") or ""),
            })
        if cleaned:
            out["actions"] = cleaned
    # Honesty and night / poor_geometry / undated_basemap stay with the heuristic.
    llm_verdict = parsed.get("verdict")
    if verdict in {"indicative", "measurable"} and llm_verdict in _VERDICTS:
        out["verdict"] = llm_verdict
    cache().set("brain", key, out, ttl_s=3600)
    return out


# --------------------------------------------------------------------------- #
# Operator tools + one-click plan
# --------------------------------------------------------------------------- #
_TOOL_CATALOG: tuple[dict[str, str], ...] = (
    {"id": "seasonal_sun", "kind": "physics",
     "label": "خورشید چهارفصل", "label_en": "Four-season solar pick"},
    {"id": "solar_clock", "kind": "physics",
     "label": "ساعت خورشیدی ۱۰–۱۶", "label_en": "Local solar clock 10–16"},
    {"id": "inspect_spectra", "kind": "cv",
     "label": "طیف RGB/HSV/LAB/false", "label_en": "Multi-spectrum inspect"},
    {"id": "detect_structures", "kind": "cv",
     "label": "شمارش سازه", "label_en": "Structure detect"},
    {"id": "greedy_hunt", "kind": "ml",
     "label": "شکار حریص سایه", "label_en": "Greedy shadow hunt"},
    {"id": "ppo_policy", "kind": "ml",
     "label": "سیاست PPO", "label_en": "PPO policy"},
    {"id": "height_cnn", "kind": "ml",
     "label": "رگرسور CNN ارتفاع", "label_en": "Height CNN"},
    {"id": "web_intel", "kind": "intel",
     "label": "OSM / ویکی‌پدیا", "label_en": "OSM / Wikipedia"},
    {"id": "embed_intel", "kind": "intel",
     "label": "رتبه‌بندی ویکی با امبدینگ", "label_en": "Embed Wikipedia rank"},
    {"id": "vlm_ocr", "kind": "vis",
     "label": "خوانش تصویر GLM-OCR", "label_en": "VLM / OCR read"},
    {"id": "reconstruct_3d", "kind": "vis",
     "label": "بازسازی سه‌بعدی", "label_en": "3D reconstruct"},
)

_PLAN_SYSTEM = (
    "You configure Shadow Hunter for one undated satellite mosaic. "
    "Reply with a single JSON object only: min_size_m, auto_sun, policy "
    "(auto|greedy|learned), measure_limit, max_tiles, prefer_year. "
    "Never claim cadastral metres. Prefer auto_sun true and prefer_year true."
)


def list_tools() -> list[dict[str, Any]]:
    """Named analysis tools the copilot can mention. Availability is live."""
    cnn = policy = False
    try:
        from ..models.pipeline import REGISTRY
        cnn = REGISTRY.cnn is not None
        policy = REGISTRY.policy is not None
    except Exception:
        pass
    names = list_local_models()
    ready = {
        "seasonal_sun": True,
        "solar_clock": True,
        "inspect_spectra": True,
        "detect_structures": True,
        "greedy_hunt": True,
        "ppo_policy": policy,
        "height_cnn": cnn,
        "web_intel": True,
        "embed_intel": _model_present(_embed_model(), names),
        "vlm_ocr": _model_present(_vlm_model(), names),
        "reconstruct_3d": True,
    }
    out = []
    for row in _TOOL_CATALOG:
        item = dict(row)
        item["ready"] = bool(ready.get(row["id"], False))
        out.append(item)
    return out


def _clamp_plan(raw: dict[str, Any], base: dict[str, Any]) -> dict[str, Any]:
    plan = dict(base)
    try:
        plan["min_size_m"] = max(6.0, min(40.0, float(raw.get("min_size_m", plan["min_size_m"]))))
    except (TypeError, ValueError):
        pass
    if "auto_sun" in raw:
        plan["auto_sun"] = bool(raw["auto_sun"])
    if "prefer_year" in raw:
        plan["prefer_year"] = bool(raw["prefer_year"])
    pol = str(raw.get("policy") or plan["policy"]).lower()
    if pol in {"auto", "greedy", "learned"}:
        plan["policy"] = pol
    try:
        plan["measure_limit"] = max(4, min(40, int(raw.get("measure_limit", plan["measure_limit"]))))
    except (TypeError, ValueError):
        pass
    try:
        plan["max_tiles"] = max(8, min(80, int(raw.get("max_tiles", plan["max_tiles"]))))
    except (TypeError, ValueError):
        pass
    return plan


def _heuristic_plan(context: dict[str, Any]) -> dict[str, Any]:
    try:
        span = float(context.get("span_m") or 400.0)
    except (TypeError, ValueError):
        span = 400.0
    if span < 180:
        min_size, tiles, limit = 8.0, 24, 12
    elif span < 600:
        min_size, tiles, limit = 12.0, 36, 16
    else:
        min_size, tiles, limit = 16.0, 48, 24
    return {
        "min_size_m": min_size,
        "auto_sun": True,
        "policy": "auto",
        "measure_limit": limit,
        "max_tiles": tiles,
        "prefer_year": True,
        "tools": [t["id"] for t in list_tools() if t.get("ready")],
        "source": "heuristic",
        "used_llm": False,
    }


def plan_run(context: dict[str, Any] | None = None) -> dict[str, Any]:
    """Few knobs for RUN FIELD. LLM may refine; heuristics always return."""
    ctx = context if isinstance(context, dict) else {}
    plan = _heuristic_plan(ctx)
    if not llm_up():
        return plan
    user = (
        "Pick settings for this AOI. JSON only.\n"
        + json.dumps(ctx, ensure_ascii=False, default=str)[:3000]
    )
    raw = complete(
        [{"role": "system", "content": _PLAN_SYSTEM},
         {"role": "user", "content": user}],
        timeout=20.0,
    )
    parsed = _extract_json(raw) if raw else None
    if not isinstance(parsed, dict):
        return plan
    out = _clamp_plan(parsed, plan)
    out["source"] = "llm"
    out["used_llm"] = True
    out["model"] = _model()
    return out


_CONSTRUCT_SYSTEM = (
    "You write a bilingual construction/research brief for Shadow Hunter. "
    "The imagery is typically an undated Esri/OSM/Carto mosaic: heights are "
    "INDICATIVE, never cadastral metres. Never invent an acquisition date; "
    "solar capture hours are ephemeris, not the mosaic's shutter time. "
    "Reply with a single JSON object: build_fa, build_en, and optional massing "
    "object {kind: azadi_arch|tower|prism, reason}. Use azadi_arch for Azadi "
    "Tower / inverted-Y monuments; tower for shafts; prism otherwise."
)


def _fmt_coords(lat: float | None, lon: float | None) -> str:
    if lat is None or lon is None:
        return "—"
    try:
        return f"{float(lat):.5f}, {float(lon):.5f}"
    except (TypeError, ValueError):
        return "—"


def _tallest_height(measures: Any) -> float | None:
    items: list[Any] = []
    if isinstance(measures, list):
        items = measures
    elif isinstance(measures, dict):
        tallest = measures.get("tallest")
        if isinstance(tallest, dict) and tallest.get("height_m") is not None:
            try:
                return float(tallest["height_m"])
            except (TypeError, ValueError):
                pass
        items = measures.get("items") or []
    heights: list[float] = []
    for item in items:
        if not isinstance(item, dict) or item.get("height_m") is None:
            continue
        try:
            heights.append(float(item["height_m"]))
        except (TypeError, ValueError):
            continue
    return max(heights) if heights else None


def _capture_local_hours(timeline: Any) -> list[Any]:
    """Local civil hours from timeline captures/slots; empty if none."""
    if not isinstance(timeline, dict):
        return []
    rows: list[Any] = []
    for key in ("captures", "slots", "capture_slots"):
        val = timeline.get(key)
        if isinstance(val, list) and val:
            rows = list(val)
            break
    if not rows:
        seasons = timeline.get("seasons")
        if isinstance(seasons, list):
            for season in seasons:
                if not isinstance(season, dict):
                    continue
                for key in ("slots", "captures"):
                    extra = season.get(key)
                    if isinstance(extra, list):
                        rows.extend(extra)
    hours: list[Any] = []
    seen: set[Any] = set()
    for row in rows:
        hour: Any = None
        if isinstance(row, dict):
            hour = row.get("local_hour")
            if hour is None:
                hour = row.get("hour")
        elif isinstance(row, bool):
            continue
        elif isinstance(row, (int, float)):
            hour = int(row) if float(row).is_integer() else row
        elif isinstance(row, str) and row.strip():
            hour = row.strip()
        if isinstance(hour, float) and hour.is_integer():
            hour = int(hour)
        if hour is None or hour in seen:
            continue
        seen.add(hour)
        hours.append(hour)
    return hours


def _heuristic_construct(place: str, coords: str, identity: dict[str, Any],
                         *, height_m: float | None, indicative: bool,
                         shadow_hours: list[Any]) -> tuple[str, str]:
    wiki = [str(t) for t in (identity.get("wikipedia") or []) if t]
    try:
        osm_n = int(identity.get("osm_buildings") or 0)
    except (TypeError, ValueError):
        osm_n = 0
    hours = "، ".join(str(h) for h in shadow_hours)
    hours_en = ", ".join(str(h) for h in shadow_hours)

    fa: list[str] = [f"{place}: مختصات {coords}."]
    en: list[str] = [f"{place}: coordinates {coords}."]
    if wiki:
        fa.append("هویت پیشنهادی: " + "، ".join(wiki) + ".")
        en.append("Candidate identity: " + ", ".join(wiki) + ".")
    if osm_n:
        fa.append(f"{osm_n} ساختمان در OSM برچسب خورده است.")
        en.append(f"{osm_n} buildings are tagged in OSM.")
    if height_m is not None:
        htxt = _fmt_num(height_m, 1)
        if indicative:
            fa.append(
                f"بلندترین ارتفاع اندازه‌گیری‌شده حدود {htxt} متر است؛ "
                "روی موزاییک بدون تاریخ (Esri/OSM/Carto) این عدد تقریبی "
                "(indicative) است نه متر کاداستری."
            )
            en.append(
                f"Tallest measured height is about {htxt} m; on this undated "
                "Esri/OSM/Carto mosaic that figure is INDICATIVE, not a cadastral metre."
            )
        else:
            fa.append(f"بلندترین ارتفاع اندازه‌گیری‌شده {htxt} متر است.")
            en.append(f"Tallest measured height is {htxt} m.")
    elif indicative:
        fa.append(
            "ارتفاع‌ها روی این موزاییک بدون تاریخ تقریبی (indicative) هستند؛ "
            "متر کاداستری ادعا نمی‌شود."
        )
        en.append(
            "Heights on this undated mosaic are INDICATIVE; cadastral metres are not claimed."
        )
    if hours:
        fa.append(
            f"ساعات برداشت خورشیدی (افمریس محلی، نه تاریخ شاتر تصویر): {hours}."
        )
        en.append(
            f"Solar capture hours (local ephemeris, not the mosaic shutter time): {hours_en}."
        )
    else:
        fa.append(
            "ساعات برداشت خورشیدی از ساعت محلی قابل جدول‌بندی است؛ "
            "تاریخ برداشت موزاییک مشخص نیست و نباید ساخته شود."
        )
        en.append(
            "Solar capture hours can be tabulated from the local civil clock; "
            "the mosaic has no acquisition date and none is invented."
        )
    return " ".join(fa), " ".join(en)


def construct(survey: dict[str, Any] | None, *, inspect: Any = None,
              timeline: Any = None, intel: Any = None, measures: Any = None,
              locale: str = "fa") -> dict[str, Any]:
    """Construction/research brief. Always returns a dict, LLM or not."""
    payload = survey if isinstance(survey, dict) else {}
    honesty_full = _honesty_block(payload)
    honesty = {
        "imagery_dated": honesty_full["imagery_dated"],
        "sun_source": honesty_full["sun_source"],
        "elevation_trusted": honesty_full["elevation_trusted"],
        "height_indicative": honesty_full["height_indicative"],
    }
    dossier = _dossier(payload, inspect=inspect, timeline=timeline,
                       intel=intel, honesty=honesty)
    place = str(dossier.get("place") or "AOI")
    lat = dossier.get("lat")
    lon = dossier.get("lon")
    try:
        lat = float(lat) if lat is not None else None
    except (TypeError, ValueError):
        lat = None
    try:
        lon = float(lon) if lon is not None else None
    except (TypeError, ValueError):
        lon = None
    raw_ident = dossier.get("identity") if isinstance(dossier.get("identity"), dict) else {}
    wiki = [str(t) for t in (raw_ident.get("wikipedia") or []) if t]
    try:
        osm_n = int(raw_ident.get("osm_buildings") or 0)
    except (TypeError, ValueError):
        osm_n = 0
    identity = {"wikipedia": wiki, "osm_buildings": osm_n}
    coords = _fmt_coords(lat, lon)
    height_indicative = bool(honesty["height_indicative"])
    height_m = _tallest_height(measures)
    shadow_hours = _capture_local_hours(timeline)
    build_fa, build_en = _heuristic_construct(
        place, coords, identity, height_m=height_m,
        indicative=height_indicative, shadow_hours=shadow_hours,
    )

    massing: dict[str, Any] = {"kind": "prism", "reason": "default silhouette"}
    try:
        from ..models.landmarks import lookup as lookup_landmark
        from ..models.massing import infer_kind
        hit = lookup_landmark(float(lat), float(lon)) if lat is not None and lon is not None else None
        vlm = (inspect or {}).get("vlm") if isinstance(inspect, dict) else None
        kind, source = infer_kind(hit, vlm, None)
        massing = {"kind": kind, "reason": source, "source": source}
        if hit and hit.get("height_m") is not None:
            massing["stated_height_m"] = float(hit["height_m"])
            massing["landmark"] = hit.get("name")
    except Exception:
        pass

    out: dict[str, Any] = {
        "place": place,
        "lat": lat,
        "lon": lon,
        "identity": identity,
        "coords": coords,
        "height_indicative": height_indicative,
        "height_m": height_m,
        "shadow_hours": shadow_hours,
        "tools": list_tools(),
        "build_fa": build_fa,
        "build_en": build_en,
        "massing": massing,
        "used_llm": False,
    }

    if not llm_up():
        return out

    ctx = _compact_context(payload, inspect=inspect, timeline=timeline,
                           intel=intel, measures=measures)
    ctx["construct"] = {
        "place": place, "coords": coords, "identity": identity,
        "height_indicative": height_indicative, "height_m": height_m,
        "shadow_hours": shadow_hours,
    }
    user = (
        f"locale={locale}. Return JSON only with build_fa, build_en, optional massing. "
        "Do not contradict honesty. Do not invent an acquisition date. "
        "Facts:\n"
        + json.dumps(ctx, ensure_ascii=False, default=str)[:8000]
    )
    raw = complete(
        [{"role": "system", "content": _CONSTRUCT_SYSTEM},
         {"role": "user", "content": user}],
        tools_context={"honesty": honesty},
        timeout=90.0,
    )
    parsed = _extract_json(raw) if raw else None
    if not isinstance(parsed, dict):
        return out

    touched = False
    if isinstance(parsed.get("build_fa"), str) and parsed["build_fa"].strip():
        out["build_fa"] = parsed["build_fa"].strip()
        touched = True
    if isinstance(parsed.get("build_en"), str) and parsed["build_en"].strip():
        out["build_en"] = parsed["build_en"].strip()
        touched = True
    if isinstance(parsed.get("massing"), dict) and parsed["massing"].get("kind"):
        kind = str(parsed["massing"]["kind"]).strip().lower()
        if kind == "arch":
            kind = "azadi_arch"
        if kind in {"azadi_arch", "tower", "prism"}:
            out["massing"] = {
                **out.get("massing", {}),
                "kind": kind,
                "reason": parsed["massing"].get("reason") or "llm",
                "source": "llm",
            }
            touched = True
    if not touched:
        return out
    if height_indicative:
        if "indicative" not in out["build_en"].lower():
            out["build_en"] += (
                " Reported metres on this undated mosaic are INDICATIVE, "
                "not a cadastral measurement."
            )
        if "indicative" not in out["build_fa"].lower() and "تقریبی" not in out["build_fa"]:
            out["build_fa"] += (
                " مترهای گزارش‌شده روی موزاییک بدون تاریخ، تقریبی (indicative) "
                "هستند نه نقشه‌برداری قطعی."
            )
    out["used_llm"] = True
    return out


def _dossier(survey: dict[str, Any], *, inspect: Any, timeline: Any,
             intel: Any, honesty: dict[str, Any]) -> dict[str, Any]:
    scene = _scene(survey)
    centre = scene.get("center") or []
    lat = lon = None
    if isinstance(centre, (list, tuple)) and len(centre) >= 2:
        lon, lat = float(centre[0]), float(centre[1])
    wiki = []
    osm_n = 0
    place = _place_name(survey, intel)
    if isinstance(intel, dict):
        wiki = [w.get("title") for w in (intel.get("wikipedia") or [])[:4]
                if isinstance(w, dict) and w.get("title")]
        overpass = intel.get("overpass") or intel.get("buildings") or {}
        if isinstance(overpass, dict):
            osm_n = int(overpass.get("count") or 0)
    spectra = (inspect or {}).get("spectra") if isinstance(inspect, dict) else None
    return {
        "place": place,
        "lat": lat,
        "lon": lon,
        "when": _sun(survey).get("when"),
        "season": _sun(survey).get("season") or survey.get("season"),
        "year_curve": _sun(survey).get("year_curve") or survey.get("year_curve") or [],
        "verdict": honesty.get("verdict"),
        "height_indicative": honesty.get("height_indicative"),
        "identity": {"wikipedia": wiki, "osm_buildings": osm_n},
        "tools": list_tools(),
        "spectra": list((spectra or {}).keys()) if isinstance(spectra, dict) else [],
        "timeline_best": _best_when(timeline),
        "vlm": ((inspect or {}).get("vlm") if isinstance(inspect, dict) else None),
    }
