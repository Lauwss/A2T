#!/usr/bin/env python3
"""从 Wikimedia Commons 下载雨声/水流环境声素材(CC0/CC BY/公有领域)。

许可过滤:仅接受 CC0、Public domain、CC BY(不含 SA/NC)。
Wikimedia 政策要求可识别的 UA;抓取间隔 1 秒。
输出:
  Datasets/env_sounds/raw/wikimedia_commons/<category>/<filename>
  Datasets/env_sounds/meta/wikimedia_commons.json  逐文件许可与来源元数据
"""

import json
import re
import sys
import time
import urllib.parse
from pathlib import Path

import requests

API = "https://commons.wikimedia.org/w/api.php"
UA = "A2T-DatasetCollector/0.1 (https://github.com/Lauwss/A2T; research on licensed environmental sound)"
DELAY = 1.0

CATEGORIES = {
    "rain": ["Category:Sounds of rain", "Category:Audio files of rain"],
    "water_flow": [
        "Category:Sounds of water",
        "Category:Audio files of streams",
        "Category:Streams (audio files)",
        "Category:Audio files of rivers",
        "Category:Waterfalls (audio files)",
    ],
}
OUT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("Datasets/env_sounds")

ALLOWED_LICENSE = re.compile(
    r"^(cc0|public domain|publicdomain|cc by(?![- ](?:sa|nc)))(?![a-z-]*\b(?:sa|nc)\b)",
    re.IGNORECASE,
)
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": UA})


def api_get(params: dict, attempts: int = 4) -> dict:
    params = dict(params, format="json")
    for attempt in range(attempts):
        try:
            response = SESSION.get(API, params=params, timeout=90)
            response.raise_for_status()
            return response.json()
        except (requests.Timeout, requests.ConnectionError) as error:
            if attempt == attempts - 1:
                raise
            print(f"  retry {attempt + 1}/{attempts - 1} after {error.__class__.__name__}")
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("unreachable")


def list_category_files(category: str) -> list[str]:
    titles: list[str] = []
    continuation: dict = {}
    while True:
        params = dict(
            action="query",
            generator="categorymembers",
            gcmtitle=category,
            gcmtype="file",
            gcmlimit=50,
            prop="imageinfo",
            iiprop="url|size|mime|extmetadata",
            **continuation,
        )
        data = api_get(params)
        pages = (data.get("query") or {}).get("pages") or {}
        for page in pages.values():
            info = (page.get("imageinfo") or [{}])[0]
            mime = info.get("mime", "")
            if mime.startswith("audio/"):
                extmeta = info.get("extmetadata") or {}
                license_name = (extmeta.get("LicenseShortName") or {}).get("value", "")
                artist = re.sub(
                    r"<[^>]+>", "", (extmeta.get("Artist") or {}).get("value", "")
                ).strip()
                titles.append(
                    json.dumps(
                        {
                            "title": page.get("title", ""),
                            "url": info.get("url", ""),
                            "mime": mime,
                            "size_bytes": info.get("size", 0),
                            "license": license_name,
                            "license_url": (extmeta.get("LicenseUrl") or {}).get("value", ""),
                            "artist": artist,
                            "category": category,
                        },
                        ensure_ascii=False,
                    )
                )
        continuation = data.get("continue") or {}
        if not continuation:
            break
        time.sleep(DELAY)
    return [json.loads(t) for t in titles]


def license_allowed(license_name: str) -> bool:
    return bool(ALLOWED_LICENSE.match(license_name.strip()))


def sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._ -]", "_", name.replace("File:", ""))[:120]


def main() -> None:
    meta_dir = OUT_DIR / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    candidates: list[dict] = []
    seen_titles: set[str] = set()
    empty_categories: list[str] = []
    for group, categories in CATEGORIES.items():
        for category in categories:
            files = list_category_files(category)
            if not files:
                empty_categories.append(category)
            for item in files:
                if item["title"] in seen_titles:
                    continue
                seen_titles.add(item["title"])
                item["group"] = group
                candidates.append(item)
            time.sleep(DELAY)

    records = []
    skipped_license = []
    for item in candidates:
        if not license_allowed(item["license"]):
            skipped_license.append(
                {"title": item["title"], "license": item["license"], "reason": "license not CC0/PD/CC-BY"}
            )
            continue
        if item["size_bytes"] > 100 * 1024 * 1024:
            skipped_license.append({"title": item["title"], "reason": "file larger than 100MB"})
            continue
        group_dir = OUT_DIR / "raw" / "wikimedia_commons" / item["group"]
        group_dir.mkdir(parents=True, exist_ok=True)
        target = group_dir / sanitize(item["title"])
        suffix = {
            "audio/ogg": ".ogg",
            "audio/midi": ".mid",
            "audio/x-flac": ".flac",
            "audio/flac": ".flac",
            "audio/wav": ".wav",
            "audio/mpeg": ".mp3",
        }.get(item["mime"], Path(urllib.parse.urlparse(item["url"]).path).suffix or ".audio")
        target = target.with_suffix(suffix)
        try:
            downloaded = False
            for attempt in range(3):
                try:
                    response = SESSION.get(item["url"], timeout=180, stream=True)
                    response.raise_for_status()
                    with open(target, "wb") as handle:
                        for chunk in response.iter_content(chunk_size=1 << 20):
                            handle.write(chunk)
                    downloaded = True
                    break
                except (requests.Timeout, requests.ConnectionError) as error:
                    if attempt == 2:
                        raise
                    print(f"  download retry {attempt + 1} after {error.__class__.__name__}")
                    time.sleep(10)
            if not downloaded:
                continue
            item["local_path"] = str(target)
            records.append(item)
            print(f"DOWNLOADED {target.name} [{item['license']}] {item['size_bytes'] // 1024} KB")
        except Exception as error:  # noqa: BLE001
            skipped_license.append({"title": item["title"], "reason": f"download failed: {error}"})
            print(f"FAILED {item['title']}: {error}")
        time.sleep(DELAY)

    summary = {
        "date": time.strftime("%Y-%m-%d"),
        "user_agent": UA,
        "license_policy": "CC0 / Public domain / CC BY only; SA and NC rejected",
        "empty_categories": empty_categories,
        "downloaded": len(records),
        "skipped": skipped_license,
        "records": records,
    }
    (meta_dir / "wikimedia_commons.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n完成:候选 {len(candidates)},下载 {len(records)},跳过 {len(skipped_license)}")
    print(f"空分类:{empty_categories}")
    print(f"元数据:{meta_dir / 'wikimedia_commons.json'}")


if __name__ == "__main__":
    main()
