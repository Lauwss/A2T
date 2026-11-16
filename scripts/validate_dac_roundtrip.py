#!/usr/bin/env python3
"""S1-2:DAC 24kHz encode/decode 往返验证(AGENTS.md 实施原则第一条)。

验证项(对应 AGENTS.md 验证要求):
  - 潜变量类型、形状、尺度:z(量化后连续向量)与 codes(离散索引)分别记录
  - 下采样率:帧数 = 样本数 / hop 实测
  - 采样率/声道:44.1k 输入重采样到 24k,输出 24k 单声道
  - 峰值/响度:输出是否削波,LUFS 前后对比
  - 重建质量:SI-SNR + 对数谱距离

结果写入 artifacts/dac_roundtrip/report.json,样本对写入 artifacts/dac_roundtrip/samples/。
日志按 AGENTS.md 要求写入 swanlab(offline 模式,无需账号;可后补同步)。

用法:
  CUDA_VISIBLE_DEVICES=1 python scripts/validate_dac_roundtrip.py [--num 5]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

import dac

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "artifacts/dac_roundtrip"
SAMPLES_DIR = OUT_DIR / "samples"

SI_SNR_EPS = 1e-8


def si_snr(reference: np.ndarray, estimate: np.ndarray) -> float:
    """尺度不变信噪比(dB);两输入均需零均值。"""
    ref = reference - reference.mean()
    est = estimate - estimate.mean()
    alpha = np.dot(est, ref) / (np.dot(ref, ref) + SI_SNR_EPS)
    target = alpha * ref
    noise = est - target
    return float(10 * np.log10((np.sum(target**2) + SI_SNR_EPS) / (np.sum(noise**2) + SI_SNR_EPS)))


def log_spectral_distance(reference: np.ndarray, estimate: np.ndarray, rate: int) -> float:
    """对数谱距离(dB):幅度谱 log10 差的均方根。"""
    n_fft = 2048
    hop = 512
    import librosa

    ref_spec = np.abs(librosa.stft(reference, n_fft=n_fft, hop_length=hop))**2
    est_spec = np.abs(librosa.stft(estimate, n_fft=n_fft, hop_length=hop))**2
    ref_db = 10 * np.log10(ref_spec + 1e-10)
    est_db = 10 * np.log10(est_spec + 1e-10)
    return float(np.sqrt(np.mean((ref_db - est_db) ** 2)))


def loudness_lufs(x: np.ndarray, rate: int) -> float | None:
    """响度;音频短于 pyloudnorm 块长(0.4s)时返回 None。"""
    import pyloudnorm as pyln

    if x.shape[0] < int(rate * 0.4):
        return None
    meter = pyln.Meter(rate)
    return float(meter.integrated_loudness(x))


def round_or_none(value, digits: int):
    return None if value is None else round(value, digits)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=5)
    parser.add_argument("--mode", default="offline", choices=["offline", "disabled"],
                        help="swanlab 模式;offline 本地落盘,后续可同步")
    args = parser.parse_args()

    meta = json.loads((ROOT / "Datasets/env_sounds/meta/fsd50k.json").read_text(encoding="utf-8"))
    # 只验 ≥8s 片段:超短片段(<1s)潜变量帧数过少,是边界个例,不代表缓存/训练分布
    records = [
        r for r in meta["records"]
        if r.get("quality_flags")
        and not r["quality_flags"].get("has_clipping")
        and r.get("probing", {}).get("duration_s", 0) >= 8
    ]
    # 覆盖两类 + 不同时长:每类取最长、中位、最短各一条附近样本
    picks = []
    for category in ("rain_stable", "water_flow"):
        group = sorted(
            (r for r in records if r["category"] == category and (ROOT / r["local_path"]).exists()),
            key=lambda r: r["probing"]["duration_s"],
        )
        n = max(1, args.num // 2)
        step = max(1, len(group) // n)
        picks += group[::step][:n]
    print(f"选取 {len(picks)} 条样本:", [(p["category"], Path(p['local_path']).name) for p in picks])

    model = dac.DAC.load(dac.utils.download(model_type="24khz"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    sample_rate = model.sample_rate
    print(f"DAC 24kHz 模型加载完成,device={device},模型采样率={sample_rate}")

    run_id = time.strftime("dac_roundtrip_%Y%m%d_%H%M%S")
    swan_run = None
    if args.mode != "disabled":
        try:
            import swanlab
            swan_run = swanlab.init(
                project="A2T-stage1",
                experiment_name=run_id,
                mode="offline" if args.mode == "offline" else "disabled",
                config={"model": "dac_24khz_8kbps_0.0.4", "num_clips": len(picks), "device": device},
            )
        except Exception as error:  # noqa: BLE001 — swanlab 失败不阻塞验证,记录即可
            print(f"swanlab 初始化失败(继续验证):{error}")

    results = []
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    for record in picks:
        path = ROOT / record["local_path"]
        original, rate_in = sf.read(str(path), dtype="float32", always_2d=True)
        mono = original.mean(axis=1)
        waveform = torchaudio.functional.resample(
            torch.from_numpy(mono)[None, None, :], orig_freq=rate_in, new_freq=sample_rate
        ).to(device)

        with torch.no_grad():
            # 在最后一维(时间)截断到 hop 整数倍;DAC.encode 对非整倍长度内部补零
            n_multiple = (waveform.shape[-1] // model.hop_length) * model.hop_length
            dac_input = waveform[..., :n_multiple]
            z, codes, latents, _, _ = model.encode(dac_input)
            reconstruction = model.decode(z)

        active_codebooks = int((codes.abs().sum(dim=(0, 2)) > 0).sum().item())

        rec_np = reconstruction[0, 0].float().cpu().numpy()
        ref_np = dac_input[0, 0].float().cpu().numpy()
        n = min(ref_np.shape[0], rec_np.shape[0])
        ref_aligned, rec_aligned = ref_np[:n], rec_np[:n]

        result = {
            "source_id": record["source_id"],
            "category": record["category"],
            "input_sample_rate": rate_in,
            "model_sample_rate": sample_rate,
            "channels": 1,
            "samples_in": int(ref_np.shape[0]),
            "duration_s": round(ref_np.shape[0] / sample_rate, 3),
            "z_shape": list(z.shape),
            "z_dtype": str(z.dtype),
            "z_scale": {
                "min": round(float(z.min()), 4), "max": round(float(z.max()), 4),
                "mean": round(float(z.mean()), 4), "std": round(float(z.std()), 4),
            },
            "codes_shape": list(codes.shape),
            "codes_dtype": str(codes.dtype),
            "codebook_size": int(model.quantizer.codebook_size),
            "n_codebooks": int(model.quantizer.n_codebooks),
            "active_codebooks": active_codebooks,
            "downsample_ratio": int(round(ref_np.shape[0] / z.shape[-1])),
            "si_snr_db": round(si_snr(ref_aligned, rec_aligned), 2),
            "log_spectral_distance_db": round(log_spectral_distance(ref_aligned, rec_aligned, sample_rate), 2),
            "loudness_in_lufs": round_or_none(loudness_lufs(ref_np, sample_rate), 2),
            "loudness_out_lufs": round_or_none(loudness_lufs(rec_np, sample_rate), 2),
            "output_peak": round(float(np.max(np.abs(rec_np))), 4),
            "output_clipping": bool(np.max(np.abs(rec_np)) >= 0.999),
        }
        results.append(result)
        print(
            f"{record['source_id']}: z{list(z.shape)} codes{list(codes.shape)} "
            f"hop={result['downsample_ratio']} ref={ref_np.shape[0]} rec={rec_np.shape[0]} "
            f"SI-SNR={result['si_snr_db']}dB LSD={result['log_spectral_distance_db']}dB"
        )
        sf.write(SAMPLES_DIR / f"{record['source_id']}_ref24k.wav", ref_np, sample_rate)
        sf.write(SAMPLES_DIR / f"{record['source_id']}_rec24k.wav", rec_np, sample_rate)
        if swan_run is not None:
            payload = {
                f"{record['source_id']}/si_snr": result["si_snr_db"],
                f"{record['source_id']}/lsd": result["log_spectral_distance_db"],
            }
            if result["loudness_in_lufs"] is not None and result["loudness_out_lufs"] is not None:
                payload[f"{record['source_id']}/loudness_delta"] = (
                    result["loudness_out_lufs"] - result["loudness_in_lufs"]
                )
            swanlab.log(payload)

    report = {
        "run_id": run_id,
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": "dac_24khz_8kbps_0.0.4 (MIT, Descript)",
        "device": device,
        "results": results,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if swan_run is not None:
        swanlab.log({"summary/mean_si_snr": float(np.mean([r["si_snr_db"] for r in results]))})
        swan_run.finish()
    print(f"报告:{OUT_DIR / 'report.json'}")


if __name__ == "__main__":
    main()
