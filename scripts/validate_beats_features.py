#!/usr/bin/env python3
"""S1-3:BEATs 预训练特征前端验证(AGENTS.md:先验证 BEATs 前处理)。

验证项:
  1. 预训练 Iter3 检查点加载(torch 2.11 需 weights_only=False)
  2. 输入规格:44.1kHz → 16kHz 重采样、[-1,1] 归一化(内部会乘 2^15 做 kaldi fbank)
  3. 特征形状:[B, patches, 768];patches 数与输入时长的映射实测
  4. padding mask 正确性:同一条片段「单独无填充」与「批内带填充」的特征必须一致(容差内)
  5. 数值健康:无 NaN、确定性(重复前向结果一致)

结果写 artifacts/beats_features/report.json;日志走 swanlab offline。
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "third_party/beats"))

from BEATs import BEATs, BEATsConfig  # noqa: E402

CKPT = ROOT / "Datasets/features/.checkpoints/BEATs_iter3.pt"
OUT_DIR = ROOT / "artifacts/beats_features"


def load_clip16k(path: Path, target_sr: int = 16000) -> torch.Tensor:
    audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    mono = mono / max(1.0, float(np.abs(mono).max()))
    return torchaudio.functional.resample(torch.from_numpy(mono), orig_freq=rate, new_freq=target_sr)


def main() -> None:
    meta = json.loads((ROOT / "Datasets/env_sounds/meta/fsd50k.json").read_text(encoding="utf-8"))
    picks = [
        r for r in meta["records"]
        if r["category"] == "rain_stable" and r.get("probing", {}).get("duration_s", 0) >= 10
    ][:1] + [
        r for r in meta["records"]
        if r["category"] == "water_flow" and r.get("probing", {}).get("duration_s", 0) >= 10
    ][:1]
    print("样本:", [(p["source_id"], p["category"]) for p in picks])

    checkpoint = torch.load(str(CKPT), map_location="cpu", weights_only=False)
    cfg = BEATsConfig(checkpoint["cfg"])
    model = BEATs(cfg)
    model.load_state_dict(checkpoint["model"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    print(f"BEATs_iter3 加载成功,device={device},embed_dim={cfg.encoder_embed_dim}")

    clips = [load_clip16k(ROOT / r["local_path"]) for r in picks]
    lengths = [c.shape[0] for c in clips]

    # 1) 单条无 padding
    single_feats = []
    with torch.no_grad():
        for clip in clips:
            feat, _ = model.extract_features(clip[None].to(device))
            single_feats.append(feat[0].cpu())
    patch_counts = [f.shape[0] for f in single_feats]
    print(f"单条特征形状: {[tuple(f.shape) for f in single_feats]}")

    # 2) 批内 padding 对齐(短的对齐到长的)——先按长的跑,再按短的跑,验证 mask 生效
    max_len = max(lengths)
    batch = torch.zeros(len(clips), max_len)
    padding_mask = torch.zeros(len(clips), max_len, dtype=torch.bool)
    for i, clip in enumerate(clips):
        batch[i, : clip.shape[0]] = clip
        padding_mask[i, clip.shape[0]:] = True
    with torch.no_grad():
        batch_feats, batch_mask_out = model.extract_features(batch.to(device), padding_mask.to(device))
    batch_feats = batch_feats.cpu()
    batch_mask_out = batch_mask_out.cpu()

    # 3) mask 正确性:批内短条的有效 patch 特征 vs 单条特征(前 P_short 个 patch)
    diffs = []
    for i, single in enumerate(single_feats):
        p = single.shape[0]
        valid_mask = ~batch_mask_out[i][:p] if batch_mask_out.shape[1] >= p else torch.ones(p, dtype=torch.bool)
        valid_idx = torch.where(valid_mask)[0]
        common = min(len(valid_idx), p)
        diff = float((batch_feats[i][valid_idx[:common]] - single[:common]).abs().max())
        diffs.append(diff)
        print(f"clip{i}: 单条 {p} patches,批内有效 {int((~batch_mask_out[i]).sum())} patches,最大特征差 {diff:.2e}")

    # 4) 确定性:同一输入重复前向
    with torch.no_grad():
        feat_again, _ = model.extract_features(clips[0][None].to(device))
    determinism_diff = float((feat_again[0].cpu() - single_feats[0]).abs().max())

    nan_count = int(sum(int(torch.isnan(f).sum()) for f in batch_feats))
    report = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint": "BEATs_iter3.pt (pretrained, AS2M)",
        "license": "unilm code MIT; checkpoint license statement see beats_README.md",
        "device": device,
        "input_spec": {"sample_rate": 16000, "channels": 1, "range": "[-1,1]"},
        "samples": [
            {"source_id": r["source_id"], "category": r["category"], "length_16k": l}
            for r, l in zip(picks, lengths)
        ],
        "patch_counts": patch_counts,
        "frames_per_patch_16k": [round(l / p, 1) for l, p in zip(lengths, patch_counts)],
        "embed_dim": int(batch_feats.shape[-1]),
        "padding_mask_max_feature_diff": diffs,
        "determinism_max_diff": determinism_diff,
        "nan_count": nan_count,
        "verdict": {
            "padding_mask_ok": all(d < 1e-3 for d in diffs),
            "deterministic": determinism_diff < 1e-5,
            "nan_free": nan_count == 0,
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        import swanlab
        run = swanlab.init(project="A2T-stage1", experiment_name=f"beats_check_{time.strftime('%Y%m%d_%H%M%S')}", mode="offline")
        swanlab.log({"beats/patch_counts": patch_counts, "beats/padding_max_diff": max(diffs)})
        run.finish()
    except Exception as error:  # noqa: BLE001
        print(f"swanlab 跳过:{error}")

    print(json.dumps(report["verdict"], ensure_ascii=False))
    print(f"报告:{OUT_DIR / 'report.json'}")


if __name__ == "__main__":
    main()
