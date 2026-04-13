"""Real reverse image search via Serper Google Lens.

Workflow:
  1. Upload crop to litterbox.catbox.moe (1-hour temp host, no API key)
  2. Call Serper Google Lens (POST /lens) with the public URL
  3. Extract visual matches → candidate titles + source pages

Falls back gracefully when SERPER_KEY is not configured (returns empty dict).

注：和 SerpApi 不同，Serper 是 serper.dev 提供的 Google SERP 代理，
更便宜、更快，和现有 web_search/image_search 共用同一个 KEY。
"""
from __future__ import annotations

import os
from urllib.parse import urlparse

import httpx

from core.config import SERPER_KEY
from core.logging_setup import get_logger

SERPER_LENS_URL = "https://google.serper.dev/lens"
LITTERBOX_URL = "https://litterbox.catbox.moe/resources/internals/api.php"

logger = get_logger("lens", "lens.log")

# 不走系统代理
_http = httpx.Client(trust_env=False, timeout=60)


def _extract_domain(url: str) -> str:
    try:
        return urlparse(url).netloc
    except Exception:
        return ""


def upload_to_litterbox(file_path: str, expire: str = "1h") -> str:
    """Upload local file to catbox litterbox temporary host.

    Returns the public URL (e.g., https://litter.catbox.moe/abc123.jpg).
    Raises on failure.

    Note: litterbox is anonymous, no API key, files expire automatically.
    Available expire values: 1h / 12h / 24h / 72h.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)
    with open(file_path, "rb") as f:
        files = {"fileToUpload": (os.path.basename(file_path), f, "image/jpeg")}
        data = {"reqtype": "fileupload", "time": expire}
        r = _http.post(LITTERBOX_URL, files=files, data=data, timeout=30)
    r.raise_for_status()
    text = (r.text or "").strip()
    if not text.startswith("http"):
        raise RuntimeError(f"litterbox unexpected response: {text[:200]}")
    return text


def serper_google_lens(image_url: str) -> dict:
    """Call Serper Google Lens. 带 disk cache（同一 image_url 不重复调）。

    POST https://google.serper.dev/lens
    Body: {"url": image_url}
    Header: X-API-KEY: ...

    Returns the parsed JSON response (dict), or empty dict on failure.
    """
    if not SERPER_KEY:
        return {}
    # disk cache: Lens 调用特别贵且 litterbox URL 是临时的（1h 后过期），
    # 但同一 session 内 URL 不变，缓存有效。
    import hashlib, json as _json
    _cache_dir = os.path.join("output", ".cache", "serper_lens")
    _key = hashlib.sha256(image_url.encode()).hexdigest()[:16]
    _path = os.path.join(_cache_dir, f"{_key}.json")
    if not os.environ.get("DISABLE_API_CACHE") and os.path.exists(_path):
        try:
            with open(_path, encoding="utf-8") as f:
                return _json.load(f)
        except Exception:
            pass
    try:
        r = _http.post(
            SERPER_LENS_URL,
            headers={"X-API-KEY": SERPER_KEY, "Content-Type": "application/json"},
            json={"url": image_url},
            timeout=60,
        )
        r.raise_for_status()
        result = r.json()
        # 写缓存
        if not os.environ.get("DISABLE_API_CACHE"):
            os.makedirs(_cache_dir, exist_ok=True)
            try:
                with open(_path, "w", encoding="utf-8") as f:
                    _json.dump(result, f, ensure_ascii=False)
            except Exception:
                pass
        return result
    except Exception as e:
        logger.warning(f"serper lens call failed: {e}")
        return {}


def reverse_search_entity(crop_path: str, use_reverse_image: bool = True) -> dict:
    """High-level reverse search: upload crop → call Serper Lens → return structured result.

    `use_reverse_image` is kept for backward compat but unused (Serper has 1 lens endpoint).

    Returns:
      {
        "available": bool,                       # True if SERPER_KEY set + upload + lens succeeded
        "lens_url": str,                         # the public URL we uploaded to
        "candidate_titles": [str, ...],          # raw titles from organic results
        "source_pages": [                        # to feed into visit queue
            {"url": str, "domain": str, "title": str, "provenance": "lens"},
            ...
        ],
        "knowledge_graph": dict | None,
        "n_visual_matches": int,                 # count of organic results
        "n_exact_matches": int,                  # always 0 (Serper doesn't separate)
      }

    Returns {"available": False} if SERPER_KEY missing or upload/lens failed.
    """
    result = {
        "available": False,
        "lens_url": "",
        "candidate_titles": [],
        "source_pages": [],
        "knowledge_graph": None,
        "products": [],
        "n_visual_matches": 0,
        "n_exact_matches": 0,
    }

    if not SERPER_KEY:
        return result
    if not os.path.exists(crop_path):
        return result

    # 1. Upload to litterbox
    try:
        public_url = upload_to_litterbox(crop_path)
        result["lens_url"] = public_url
    except Exception as e:
        logger.warning(f"litterbox upload failed for {crop_path}: {e}")
        return result

    # 2. Call Serper Lens
    lens = serper_google_lens(public_url)
    if not lens:
        return result

    # 3. Extract organic results (Serper Lens 主结构是 organic)
    candidates: list[str] = []
    source_pages: list[dict] = []
    seen_urls: set[str] = set()

    organic = lens.get("organic") or []
    for item in organic[:20]:
        title = (item.get("title") or "").strip()
        if title and len(title) >= 3:
            candidates.append(title)
        link = item.get("link") or item.get("source") or ""
        if link and link not in seen_urls:
            seen_urls.add(link)
            source_pages.append({
                "url": link,
                "domain": _extract_domain(link),
                "title": title,
                "provenance": "lens",
            })
    result["n_visual_matches"] = len(organic)

    # 4. Knowledge graph (Serper sometimes returns it)
    kg = lens.get("knowledgeGraph") or lens.get("knowledge_graph")
    if kg:
        result["knowledge_graph"] = kg

    # 5. Dedupe candidate titles
    seen_lower: set[str] = set()
    deduped = []
    for c in candidates:
        cl = c.lower()
        if cl not in seen_lower:
            seen_lower.add(cl)
            deduped.append(c)
    result["candidate_titles"] = deduped[:20]
    result["source_pages"] = source_pages[:15]
    result["available"] = True
    return result
