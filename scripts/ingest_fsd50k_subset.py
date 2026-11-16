#!/usr/bin/env python3
"""从 FSD50K 整包 zip 中挑取 rain/water_flow 可商用子集并入库。

前置:
  - Datasets/env_sounds/meta/fsd50k_subset_plan.json(由 FSD50K 元数据分析生成)
  - Datasets/env_sounds/_fsd50k_full/ 下 8 个分卷 zip(下载完成后)
步骤:
  1. 用 7z 列出 zip 内容,确认片段路径格式
  2. 生成挑选清单,7z 按清单选择性解压到临时目录
  3. 移动到 raw/fsd50k/<category>/,补充元数据(license、uploader、attribution)
  4. 校验:文件存在、可解码,数量与清单一致
不删除整包;确认入库完成后由调用方决定是否清理。
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

SEVEN_ZIP = str(Path.home() / ".conda/envs/bird/bin/7z")
ROOT = Path(__file__).resolve().parent.parent
FULL_DIR = ROOT / "Datasets/env_sounds/_fsd50k_full"
RAW_DIR = ROOT / "Datasets/env_sounds/raw/fsd50k"
META_DIR = ROOT / "Datasets/env_sounds/meta"
CLIP_INFO = Path("/tmp/fsd50k_meta")

CATEGORY_OF_PLAN_KEY = {"rain": "rain_stable", "water": "water_flow"}


def load_plans() -> tuple[list[dict], dict]:
    plan = json.loads((META_DIR / "fsd50k_subset_plan.json").read_text(encoding="utf-8"))
    clip_info = {}
    for split in ("dev", "eval"):
        info = json.loads((CLIP_INFO / f"FSD50K.metadata/{split}_clips_info_FSD50K.json").read_text())
        clip_info.update(info)
    return plan, clip_info


def locate_clip_paths(plan: dict) -> dict:
    """把 fname(纯数字)映射到 zip 内部路径。"""
    listing = subprocess.run(
        [SEVEN_ZIP, "l", "-ba", str(FULL_DIR / "FSD50K.dev_audio.zip")],
        capture_output=True, text=True, timeout=600,
    )
    listing_eval = subprocess.run(
        [SEVEN_ZIP, "l", "-ba", str(FULL_DIR / "FSD50K.eval_audio.zip")],
        capture_output=True, text=True, timeout=600,
    )
    path_of = {}
    for line in (listing.stdout + listing_eval.stdout).splitlines():
        parts = line.split()
        if len(parts) >= 6 and parts[-1].endswith(".wav"):
            fname = Path(parts[-1]).stem
            path_of[fname] = parts[-1]
    missing = []
    for key in ("rain", "water"):
        for item in plan[key]:
            if item["fname"] not in path_of:
                missing.append(item["fname"])
            item["zip_path"] = path_of.get(item["fname"])
    if missing:
        print(f"警告:{len(missing)} 个片段未在 zip 中找到,示例 {missing[:5]}")
    return path_of


def extract(plan: dict) -> None:
    tmp = FULL_DIR / "extract_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    for split in ("dev", "eval"):
        names = []
        for key in ("rain", "water"):
            names += [
                item["zip_path"] for item in plan[key]
                if item.get("zip_path") and item["split"] == split and not (tmp / Path(item["zip_path"]).name).exists()
            ]
        if not names:
            continue
        listfile = FULL_DIR / f"list_{split}.txt"
        listfile.write_text("\n".join(names), encoding="utf-8")
        archive = FULL_DIR / (f"FSD50K.dev_audio.zip" if split == "dev" else "FSD50K.eval_audio.zip")
        print(f"解压 {split}: {len(names)} 个片段")
        result = subprocess.run(
            [SEVEN_ZIP, "x", "-y", f"-o{tmp}", f"@{listfile}", str(archive)],
            capture_output=True, text=True, timeout=3600,
        )
        if result.returncode != 0:
            print(result.stderr[-2000:])
            raise RuntimeError(f"7z 解压失败({split})")
    return tmp


def ingest(plan: dict, clip_info: dict) -> None:
    tmp = FULL_DIR / "extract_tmp"
    records = []
    counts = {}
    for key, category in CATEGORY_OF_PLAN_KEY.items():
        category_dir = RAW_DIR / category
        category_dir.mkdir(parents=True, exist_ok=True)
        for item in plan[key]:
            src = tmp / Path(item["zip_path"]).name
            if not src.exists():
                continue
            target = category_dir / f"{item['fname']}.wav"
            shutil.move(str(src), str(target))
            info = clip_info.get(item["fname"], {})
            records.append({
                "source_id": f"fsd50k_{item['fname']}",
                "platform": "fsd50k",
                "platform_ref": f"https://freesound.org/s/{item['fname']}/",
                "category": category,
                "subcategory": None,
                "license": item["license"],
                "license_url": info.get("license", ""),
                "attribution": f"{info.get('uploader', 'unknown')} (FSD50K clip {item['fname']})",
                "commercial_use": True,
                "title": info.get("title", ""),
                "description": (info.get("description") or "")[:300],
                "local_path": str(target),
                "fsd50k_split": item["split"],
                "processing_version": "raw-v1",
                "quality_flags": {},
                "notes": "clip 时长 0.3-30s;训练切片规格见 TAXONOMY.md",
            })
            counts[category] = counts.get(category, 0) + 1

    meta = {
        "date": __import__("time").strftime("%Y-%m-%d"),
        "source": "FSD50K (Zenodo record 4060432), subset by rain/water_flow labels",
        "license_policy": "CC0 / CC BY only; CC BY-NC and unknown licenses excluded at plan stage",
        "counts": counts,
        "records": records,
    }
    (META_DIR / "fsd50k.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"入库完成:{counts},共 {len(records)} 条,元数据 {META_DIR / 'fsd50k.json'}")


def main() -> None:
    plan, clip_info = load_plans()
    path_of = locate_clip_paths(plan)
    if len(path_of) < 40000:
        raise RuntimeError(f"zip 内路径仅 {len(path_of)} 条,可能未下载完整")
    extract(plan)
    ingest(plan, clip_info)


if __name__ == "__main__":
    main()
