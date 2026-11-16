#!/usr/bin/env python3
"""环境声入库质检:采样率、声道、时长、削波、静音、直流偏移、精确重复。

用法:
  python -I scripts/inspect_env_sounds.py Datasets/env_sounds/meta/wikimedia_commons.json

对元数据 JSON 中每条已下载记录(raw 文件存在)执行:
  - ffprobe:codec、采样率、声道、时长
  - soundfile 解码(失败时用 ffmpeg 转 wav 再读):削波比例、静音比例、DC 偏移、峰值、RMS
  - sha256 精确重复检测
结果写回原 JSON 的 records[i].probing / records[i].quality_flags,并给出质检摘要。
"""

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

CLIP_THRESHOLD = 0.999
SILENCE_DB = -60.0


def ffprobe_info(path: Path) -> dict:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_format", "-show_streams", str(path)],
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0:
        return {"error": result.stderr.strip()[:200]}
    data = json.loads(result.stdout)
    audio = next((s for s in data.get("streams", []) if s.get("codec_type") == "audio"), {})
    return {
        "codec": audio.get("codec_name"),
        "sample_rate": int(audio["sample_rate"]) if audio.get("sample_rate") else None,
        "channels": int(audio["channels"]) if audio.get("channels") else None,
        "duration_s": float(audio["duration"]) if audio.get("duration")
        else (float(data.get("format", {}).get("duration", 0)) or None),
    }


def load_samples(path: Path, max_seconds: float = 600.0) -> tuple[np.ndarray, int]:
    try:
        audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception:  # noqa: BLE001 — soundfile 不支持时用 ffmpeg 转 wav
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            subprocess.run(
                ["ffmpeg", "-v", "quiet", "-i", str(path), "-t", str(max_seconds),
                 "-ac", "1", "-ar", "24000", "-f", "wav", tmp.name],
                check=True, timeout=300,
            )
            audio, rate = sf.read(tmp.name, dtype="float32", always_2d=True)
    limit = int(max_seconds * rate)
    if audio.shape[0] > limit:
        audio = audio[:limit]
    return audio, rate


def quality_metrics(audio: np.ndarray, rate: int) -> dict:
    mono = audio.mean(axis=1)
    n = max(1, len(mono))
    peak = float(np.max(np.abs(mono))) if n else 0.0
    clipped = float(np.mean(np.abs(mono) >= CLIP_THRESHOLD)) if n else 0.0
    rms = float(np.sqrt(np.mean(np.square(mono)))) if n else 0.0
    rms_db = 20 * np.log10(max(rms, 1e-10))
    silence = float(np.mean(rms_db < SILENCE_DB)) if n else 0.0  # 粗略:整段低于阈值的样本占比
    frame = rate  # 每帧 1 秒
    n_frames = max(1, len(mono) // frame)
    frame_rms_db = np.array([
        20 * np.log10(max(float(np.sqrt(np.mean(np.square(mono[i * frame:(i + 1) * frame])))), 1e-10))
        for i in range(n_frames)
    ])
    silence = float(np.mean(frame_rms_db < SILENCE_DB))
    return {
        "peak": round(peak, 5),
        "clipping_ratio": round(clipped, 6),
        "rms_db": round(rms_db, 2),
        "silence_ratio": round(silence, 4),
        "dc_offset": round(float(np.mean(mono)), 6) if n else 0.0,
        "has_clipping": bool(clipped > 0.0005),
    }


def main() -> None:
    meta_path = Path(sys.argv[1])
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    hashes: dict[str, str] = {}
    problems: list[dict] = []
    checked = 0

    for record in meta.get("records", []):
        local = record.get("local_path")
        if not local or not Path(local).exists():
            continue
        checked += 1
        path = Path(local)
        info = ffprobe_info(path)
        record["probing"] = info
        if "error" in info:
            problems.append({"source": record["title"], "issue": "ffprobe failed", "detail": info["error"]})
            continue
        try:
            audio, rate = load_samples(path)
            record["quality_flags"] = quality_metrics(audio, rate)
        except Exception as error:  # noqa: BLE001
            problems.append({"source": record["title"], "issue": "decode failed", "detail": str(error)[:200]})
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        record["quality_flags"]["duplicate_of"] = hashes.get(digest)
        hashes.setdefault(digest, record["title"])
        if record["quality_flags"]["has_clipping"]:
            problems.append({"source": record["title"], "issue": "clipping detected",
                             "detail": f"ratio={record['quality_flags']['clipping_ratio']}"})
        if record["quality_flags"]["silence_ratio"] > 0.8:
            problems.append({"source": record["title"], "issue": "mostly silent",
                             "detail": f"silence_ratio={record['quality_flags']['silence_ratio']}"})
        if info.get("duration_s") is not None and info["duration_s"] < 10:
            problems.append({"source": record["title"], "issue": "shorter than 10s",
                             "detail": f"duration={info['duration_s']}"})

    summary = {
        "checked": checked,
        "problems": len(problems),
        "problem_list": problems,
    }
    meta["qc"] = summary
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"质检完成:{checked} 个文件,发现 {len(problems)} 个问题")
    for item in problems[:20]:
        print(f"  [{item['issue']}] {item['source']} :: {item['detail'][:80]}")
    print(f"结果已写回 {meta_path}")


if __name__ == "__main__":
    main()
