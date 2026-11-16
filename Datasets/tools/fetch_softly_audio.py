#!/usr/bin/env python3
"""Softly 音频抓取(权利人授权,内部使用,数据集不公开)。

前提:Datasets/env_sounds/meta/softly_permission_record.json 记录的项目所有者授权声明。
范围:仅 TAXONOMY v0.1 核心类别(rain-sounds → rain_stable, stream-sounds → water_flow);
     扩展类别需要 --categories 显式开启。
行为:yt-dlp 只取每条视频的前 1 小时音频(--download-sections),原格式 m4a/opus 落盘,
     不转码;逐条记录元数据;条目间限速。

代理:读取环境变量 HTTPS_PROXY/HTTP_PROXY。YouTube 本机直连不通,必须配置代理:
     export HTTPS_PROXY=http://IP:PORT && python fetch_softly_audio.py

输出:
  Datasets/env_sounds/raw/softly/<category>/<videoid>.m4a
  Datasets/env_sounds/meta/softly.json
"""

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent  # Datasets/tools -> repo root
TAXONOMY_DIR = ROOT / "Datasets/reference/softly_taxonomy"
RAW_DIR = ROOT / "Datasets/env_sounds/raw/softly"
META_DIR = ROOT / "Datasets/env_sounds/meta"

YT_DLP = str(Path.home() / ".conda/envs/bird/bin/yt-dlp")
CATEGORY_MAP = {  # softly 分类 slug -> A2T category(全套 12 类;guqin-ambient 无嵌入视频,抓不到)
    "rain-sounds": "rain_stable",
    "stream-sounds": "water_flow",
    "ocean-waves": "ocean_waves",
    "wind-sounds": "wind",
    "thunderstorm-sounds": "thunderstorm",
    "forest-sounds": "forest_ambience",
    "fireplace-sounds": "fireplace",
    "coffee-shop-sounds": "coffee_shop",
    "white-noise": "white_noise",
    "pink-noise": "pink_noise",
    "brown-noise": "brown_noise",
    "lofi-music": "lofi_music",
}
SECTION_MINUTES = 60
SLEEP_SECONDS = 8


def section_end(minutes: int) -> str:
    """yt-dlp 时间戳:60 分钟 → '01:00:00',90 分钟 → '01:30:00'。"""
    return f"{minutes // 60:02d}:{minutes % 60:02d}:00"


def check_proxy() -> None:
    import os
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    if not proxy:
        print("错误:YouTube 本机直连不通,必须先配置代理再运行:")
        print("  export HTTPS_PROXY=http://IP:PORT")
        print("  然后重新执行本脚本。")
        sys.exit(1)
    print(f"使用代理:{proxy}")


def select_videos() -> list[dict]:
    """每条视频一个条目;归属多个分类时全登记,文件存到第一个分类目录。"""
    index = json.loads((TAXONOMY_DIR / "softly_video_index.json").read_text(encoding="utf-8"))
    selected = []
    for video_id, info in index["videos"].items():
        categories = [CATEGORY_MAP[slug] for slug in info["categories"] if slug in CATEGORY_MAP]
        if not categories:
            continue
        selected.append({
            "video_id": video_id,
            "softly_categories": info["categories"],
            "categories": categories,
            "durations": info["durations"],
        })
    return selected


def download_one(entry: dict) -> dict:
    category_dir = RAW_DIR / entry["categories"][0]
    category_dir.mkdir(parents=True, exist_ok=True)
    target = category_dir / f"{entry['video_id']}.m4a"
    if target.exists() and target.stat().st_size > 100_000:
        return {**entry, "status": "exists", "path": str(target), "bytes": target.stat().st_size}
    cmd = [
        YT_DLP,
        "-f", "bestaudio[ext=m4a]/bestaudio",
        "--download-sections", f"*00:00-{section_end(SECTION_MINUTES)}",
        "--force-keyframes-at-cuts",
        "--no-playlist",
        "--no-warnings",
        "-o", str(category_dir / "%(id)s.%(ext)s"),
        f"https://www.youtube.com/watch?v={entry['video_id']}",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if target.exists() and target.stat().st_size > 100_000:
        return {**entry, "status": "ok", "path": str(target), "bytes": target.stat().st_size}
    return {**entry, "status": "failed",
            "error": (result.stderr or result.stdout)[-300:]}


def main() -> None:
    check_proxy()
    entries = select_videos()
    print(f"计划下载 {len(entries)} 条视频,每条取前 {SECTION_MINUTES} 分钟音频")

    results = []
    for i, entry in enumerate(entries):
        print(f"[{i + 1}/{len(entries)}] {entry['category']} {entry['video_id']} ...")
        results.append(download_one(entry))
        time.sleep(SLEEP_SECONDS)

    ok = [r for r in results if r["status"] in ("ok", "exists")]
    failed = [r for r in results if r["status"] == "failed"]
    print(f"\n完成:{len(ok)} 成功,{len(failed)} 失败,共 {sum(r.get('bytes', 0) for r in ok) / 1e6:.0f} MB")

    meta = {
        "date": time.strftime("%Y-%m-%d"),
        "authorization": "see softly_permission_record.json (project owner asserted; dataset private, no redistribution)",
        "license": "Softly-authorized-restricted",
        "commercial_use": False,
        "section": f"first {SECTION_MINUTES} minutes of each video",
        "counts": {},
        "failed": failed,
        "records": [
            {
                "source_id": f"softly_{r['video_id']}",
                "platform": "softly_youtube",
                "platform_ref": f"https://www.youtube.com/watch?v={r['video_id']}",
                "category": r["categories"][0],
                "additional_categories": r["categories"][1:],
                "softly_categories": r["softly_categories"],
                "subcategory": None,
                "license": "Softly-authorized-restricted",
                "attribution": "Softly (softly.cc)",
                "commercial_use": False,
                "durations_on_site": r["durations"],
                "local_path": r.get("path"),
                "processing_version": "raw-v1",
                "notes": "内部训练使用;数据集不公开;不得再分发",
            } for r in ok
        ],
    }
    counts: dict = {}
    for r in ok:
        for category in r["categories"]:
            counts[category] = counts.get(category, 0) + 1
    meta["counts"] = counts
    (META_DIR / "softly.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"元数据:{META_DIR / 'softly.json'},计数 {counts}")


if __name__ == "__main__":
    main()
