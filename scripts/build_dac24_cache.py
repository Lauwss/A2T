#!/usr/bin/env python3
"""S1-4:用冻结的 DAC 24kHz 对环境声库批量编码,建立潜变量缓存。

范围:fsd50k.json 中时长 ≥ 8 秒的片段(不足 8 秒的无法产出 8 秒训练窗,跳过并记录)。
输出:
  Datasets/env_sounds/features/dac24/<source_id>.npz   z(quantized latent),float16,[1024, T]
  Datasets/env_sounds/meta/dac24_cache_manifest.json    逐片段统计 + 全局规范

规范要点(AGENTS.md:记录潜变量缩放):
  - hop=320 样本 @24kHz,即 75 帧/秒;z 为量化后连续向量,不是离散 code
  - 归一化统计(mean/std)只从 train split 片段拟合,验证/测试固定使用
  - 断点续传:已存在的 npz 跳过
日志:swanlab offline(A2T-stage1 项目)。
"""

import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

import dac

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "Datasets/env_sounds/features/dac24"
META_PATH = ROOT / "Datasets/env_sounds/meta/fsd50k.json"
MANIFEST_PATH = ROOT / "Datasets/env_sounds/meta/dac24_cache_manifest.json"
MIN_DURATION_S = 8.0


def encode_clip(model, waveform: torch.Tensor) -> np.ndarray:
    """输入 [1, 1, T] @24kHz,返回 z [1024, T_frames] float32。"""
    with torch.no_grad():
        z, _, _, _, _ = model.encode(waveform)
    return z[0].float().cpu().numpy()


def main() -> None:
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    records = [
        r for r in meta["records"]
        if r.get("probing", {}).get("duration_s", 0) >= MIN_DURATION_S
    ]
    skipped = len(meta["records"]) - len(records)
    print(f"待缓存 {len(records)} 条(≥{MIN_DURATION_S}s),跳过 {skipped} 条短片段")

    model = dac.DAC.load(dac.utils.download(model_type="24khz"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    print(f"模型加载完成,device={device}")

    swan_run = None
    try:
        import swanlab
        swan_run = swanlab.init(
            project="A2T-stage1",
            experiment_name=f"dac24_cache_{time.strftime('%Y%m%d_%H%M%S')}",
            mode="offline",
            config={"clips": len(records), "device": device, "min_duration_s": MIN_DURATION_S},
        )
    except Exception as error:  # noqa: BLE001
        print(f"swanlab 初始化失败(继续):{error}")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    per_clip_stats = []
    train_z, val_z = [], []
    t0 = time.time()
    done = 0
    for index, record in enumerate(records):
        target = CACHE_DIR / f"{record['source_id']}.npz"
        if not target.exists():
            audio, rate = sf.read(record["local_path"], dtype="float32", always_2d=True)
            mono = audio.mean(axis=1)
            waveform = torchaudio.functional.resample(
                torch.from_numpy(mono)[None, None, :], orig_freq=rate, new_freq=model.sample_rate
            ).to(device)
            z = encode_clip(model, waveform)
            np.savez_compressed(target, z=z.astype(np.float16))
        else:
            z = np.load(target)["z"].astype(np.float32)

        stats = {
            "source_id": record["source_id"],
            "category": record["category"],
            "split": record["split"],
            "frames": int(z.shape[1]),
            "z_std": round(float(z.std()), 4),
            "z_absmax": round(float(np.abs(z).max()), 4),
        }
        per_clip_stats.append(stats)
        if record["split"] == "train":
            train_z.append(z)
        else:
            val_z.append(z)

        done += 1
        if done % 100 == 0:
            rate_ = done / (time.time() - t0)
            print(f"  {done}/{len(records)}  {rate_:.1f} 条/秒")
            if swan_run is not None:
                swanlab.log({"cache/progress": done / len(records), "cache/rate_cps": rate_})

    def global_stats(lists):
        all_z = np.concatenate([z.ravel() for z in lists]) if lists else np.array([0.0])
        return {
            "mean": round(float(all_z.mean()), 4),
            "std": round(float(all_z.std()), 4),
            "absmax": round(float(np.abs(all_z).max()), 4),
        }

    manifest = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "codec": "dac_24khz_8kbps_0.0.4 (MIT)",
        "spec": {
            "sample_rate": 24000,
            "channels": 1,
            "hop": 320,
            "frames_per_second": 75,
            "latent_dim": 1024,
            "dtype": "float16(存储)",
            "representation": "量化后连续向量 z,非离散 code;不得对 code 索引加噪",
        },
        "clips_cached": len(per_clip_stats),
        "clips_skipped_short": skipped,
        "normalization": {
            "note": "归一化统计只从 train split 拟合;val/test 固定使用 train 统计",
            "train": global_stats(train_z),
            "val_reference": global_stats(val_z),
        },
        "per_clip": per_clip_stats,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if swan_run is not None:
        swanlab.log({
            "cache/train_z_mean": manifest["normalization"]["train"]["mean"],
            "cache/train_z_std": manifest["normalization"]["train"]["std"],
        })
        swan_run.finish()
    print(f"完成:缓存 {len(per_clip_stats)} 条,耗时 {(time.time() - t0) / 60:.1f} 分钟")
    print(f"train 统计:{manifest['normalization']['train']}")
    print(f"清单:{MANIFEST_PATH}")


if __name__ == "__main__":
    main()
