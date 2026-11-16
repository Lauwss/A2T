#!/usr/bin/env python3
"""从 hf-mirror 的 Fhrozen/FSD50k 镜像按清单下载雨声/水流片段。

相比从 Zenodo 拉 24.7 GB 整包,镜像按片段存成独立 wav,只下载所需部分。
Zenodo 对本机按请求数限流(x-ratelimit-limit: 133),整包下载不可行。

用法:
  python -I scripts/download_fsd50k_mirror.py [workers]

输入: Datasets/env_sounds/meta/fsd50k_subset_plan.json
输出: Datasets/env_sounds/raw/fsd50k/<category>/<fname>.wav
      Datasets/env_sounds/meta/fsd50k.json(逐条许可与来源)
已存在的文件跳过,可反复运行续传。
"""

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
META_DIR = ROOT / "Datasets/env_sounds/meta"
RAW_DIR = ROOT / "Datasets/env_sounds/raw/fsd50k"
CLIP_INFO_DIRS = [Path("/tmp/fsd50k_meta/FSD50K.metadata"), META_DIR]
URL = "https://hf-mirror.com/datasets/Fhrozen/FSD50k/resolve/main/clips/{split}/{fname}.wav"
CATEGORY = {"rain": "rain_stable", "water": "water_flow"}


def load_clip_info() -> dict:
    info: dict = {}
    for directory in CLIP_INFO_DIRS:
        for split in ("dev", "eval"):
            path = directory / f"{split}_clips_info_FSD50K.json"
            if path.exists():
                info.update(json.loads(path.read_text(encoding="utf-8")))
    return info


def download_one(item: dict, category: str) -> dict:
    fname, split = item["fname"], item["split"]
    target_dir = RAW_DIR / category
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{fname}.wav"
    if target.exists() and target.stat().st_size > 1000:
        return {"fname": fname, "status": "exists", "bytes": target.stat().st_size}
    tmp = target.with_suffix(".wav.part")
    for attempt in range(4):
        result = subprocess.run(
            ["curl", "-sL", "--max-time", "180", "--speed-limit", "2048", "--speed-time", "30",
             "-o", str(tmp), URL.format(split=split, fname=fname)],
            capture_output=True,
        )
        if result.returncode == 0 and tmp.exists() and tmp.stat().st_size > 1000:
            os.replace(tmp, target)
            return {"fname": fname, "status": "ok", "bytes": target.stat().st_size}
        time.sleep(3 * (attempt + 1))
    return {"fname": fname, "status": "failed", "bytes": 0}


def main() -> None:
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    plan = json.loads((META_DIR / "fsd50k_subset_plan.json").read_text(encoding="utf-8"))
    clip_info = load_clip_info()

    jobs = []
    for key, category in CATEGORY.items():
        for item in plan[key]:
            jobs.append((item, category))
    print(f"待处理 {len(jobs)} 条,{workers} 并行")

    t0 = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(download_one, item, category) for item, category in jobs]
        for index, future in enumerate(futures):
            results.append(future.result())
            if (index + 1) % 200 == 0:
                done = sum(r["bytes"] for r in results) / 1e6
                print(f"  {index + 1}/{len(jobs)}  {done:.0f} MB  {time.time() - t0:.0f}s")

    ok = [r for r in results if r["status"] in ("ok", "exists")]
    failed = [r for r in results if r["status"] == "failed"]
    elapsed = time.time() - t0
    total_bytes = sum(r["bytes"] for r in ok)
    print(f"下载完成:{len(ok)}/{len(jobs)} 成功,{total_bytes / 1e6:.0f} MB,"
          f"用时 {elapsed / 60:.1f} 分钟,失败 {len(failed)}")

    records = []
    for item, category in jobs:
        fname = item["fname"]
        path = RAW_DIR / category / f"{fname}.wav"
        if not path.exists():
            continue
        info = clip_info.get(fname, {})
        records.append({
            "source_id": f"fsd50k_{fname}",
            "platform": "fsd50k",
            "platform_ref": f"https://freesound.org/s/{fname}/",
            "session_id": f"fsd50k_{fname}",
            "category": category,
            "subcategory": None,
            "license": item["license"],
            "license_url": info.get("license", ""),
            "attribution": f"{info.get('uploader', 'unknown')} (FSD50K clip {fname})",
            "commercial_use": True,
            "title": info.get("title", ""),
            "description": (info.get("description") or "")[:300],
            "local_path": str(path),
            "fsd50k_split": item["split"],
            "split": "train" if item["split"] == "dev" else "val",
            "processing_version": "raw-v1",
            "quality_flags": {},
            "notes": "clip 时长 0.3-30s;训练切片规格见 docs/dataset/TAXONOMY.md",
        })
    counts: dict = {}
    for record in records:
        counts[record["category"]] = counts.get(record["category"], 0) + 1
    meta = {
        "date": time.strftime("%Y-%m-%d"),
        "source": "FSD50K via hf-mirror Fhrozen/FSD50k (Zenodo record 4060432)",
        "license_policy": "CC0 / CC BY only; CC BY-NC and unknown excluded at plan stage",
        "counts": counts,
        "failed": failed,
        "split_note": "FSD50K dev 用作 train,eval 用作 val;与原数据集划分一致",
        "records": records,
    }
    (META_DIR / "fsd50k.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"元数据:{META_DIR / 'fsd50k.json'},计数 {counts}")


if __name__ == "__main__":
    main()
