"""Recompute listening-test boundaries from a saved results JSON.

The blind-listening page stores raw per-trial responses; this script
re-scores them with the documented rule: the boundary is the midpoint between
the highest level answered "no" and the lowest level answered "yes".
"Unsure" responses are shown but excluded, so one lapse at a quiet level
cannot drag the estimate. Offsets are reported against the model thresholds
recorded in ``answers.json``.

The script exists so saved sessions can be re-scored after scoring logic
changes: the page's v1 boundary display matched responses by label across
frequencies (labels collide), so the round-2 session of 2026-11-13 showed
wrong boundaries on the results screen. The raw trials were always correct.

Usage
-----
    .venv\\Scripts\\python.exe scripts/rescore_listening.py ^
        --results artifacts/listening_test/listening_results_<ts>.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def boundary_for(
    levels_by_label: dict[str, float],
    responses_by_label: dict[str, str],
) -> dict:
    """Midpoint of the highest "no" and the lowest "yes", in dBFS levels."""
    heard = [
        level
        for label, level in levels_by_label.items()
        if responses_by_label.get(label) == "yes"
    ]
    unheard = [
        level
        for label, level in levels_by_label.items()
        if responses_by_label.get(label) == "no"
    ]
    all_levels = list(levels_by_label.values())
    if not heard:
        return {"kind": "capped-above", "boundary_dbfs": max(all_levels)}
    if not unheard:
        return {"kind": "capped-below", "boundary_dbfs": min(all_levels)}
    highest_unheard = max(unheard)
    lowest_heard = min(heard)
    boundary = {
        "kind": "boundary",
        "boundary_dbfs": round((highest_unheard + lowest_heard) / 2.0, 2),
        "highest_unheard_dbfs": highest_unheard,
        "lowest_heard_dbfs": lowest_heard,
    }
    # A "yes" below an unheard level means a false alarm or a lapse; the
    # midpoint then lands between unrelated trials and must not be trusted.
    if lowest_heard < highest_unheard:
        boundary["inverted"] = True
    return boundary


def score_session(results: dict, answers: dict) -> dict[str, dict]:
    """Recompute per-frequency boundaries for one saved session."""
    levels_all = answers.get("levels", {})
    thresholds = answers.get("model_threshold_dbfs", {})
    collected: dict[str, dict] = {}
    for trial in results.get("trials", []):
        response = trial.get("response")
        if not response:
            continue
        hz_key = f"{trial['hz']:g}"
        level = levels_all.get(hz_key, {}).get(trial.get("label"))
        if level is None:
            continue
        entry = collected.setdefault(hz_key, {"levels": {}, "responses": {}})
        entry["levels"][trial["label"]] = float(level)
        entry["responses"][trial["label"]] = response

    out: dict[str, dict] = {}
    for hz_key, data in collected.items():
        boundary = boundary_for(data["levels"], data["responses"])
        threshold = thresholds.get(hz_key)
        if threshold is not None:
            boundary["model_threshold_dbfs"] = threshold
            boundary["offset_db"] = round(boundary["boundary_dbfs"] - threshold, 2)
        boundary["responses"] = data["responses"]
        boundary["levels"] = data["levels"]
        out[hz_key] = boundary
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, required=True, help="saved results JSON")
    parser.add_argument(
        "--answers", type=Path, default=Path(__file__).resolve().parents[1] / "artifacts" / "probe_test" / "answers.json"
    )
    parser.add_argument("--json", type=Path, help="write the re-scored boundaries here")
    args = parser.parse_args(argv)

    results = json.loads(args.results.read_text(encoding="utf-8"))
    answers = json.loads(args.answers.read_text(encoding="utf-8"))
    scored = score_session(results, answers)

    print(f"session: {args.results.name}  ({results.get('timestamp', '?')})")
    print(f"  {'Hz':>6}  {'boundary':>10}  {'offset':>8}  rule")
    offsets = []
    inverted_count = 0
    for hz_key, boundary in scored.items():
        offset = boundary.get("offset_db")
        inverted = boundary.get("inverted", False)
        if inverted:
            inverted_count += 1
        if boundary["kind"] == "boundary" and offset is not None and not inverted:
            offsets.append(offset)
            rule = (
                f"no<={boundary['highest_unheard_dbfs']:.2f}  "
                f"yes>={boundary['lowest_heard_dbfs']:.2f}"
            )
            text = f"{boundary['boundary_dbfs']:.2f} dBFS  {offset:+.2f} dB"
        elif boundary["kind"] == "boundary":
            rule = "NON-MONOTONIC responses, midpoint unreliable"
            text = f"{boundary['boundary_dbfs']:.2f} dBFS  (excluded)"
        elif boundary["kind"] == "capped-above":
            rule = "no 'yes' response"
            text = f">={boundary['boundary_dbfs']:.2f} dBFS"
        elif boundary["kind"] == "capped-below":
            rule = "no 'no' response"
            text = f"<={boundary['boundary_dbfs']:.2f} dBFS"
        else:
            rule = "no usable responses"
            text = "-"
        print(f"  {hz_key:>6}  {text:>28}  {rule}")
    if inverted_count:
        print(
            f"\n  warning: {inverted_count} freq(s) answered 'yes' below an unheard level "
            "(false alarm or lapse); excluded from the mean."
        )
    if offsets:
        mean = sum(offsets) / len(offsets)
        spread = (
            sum((v - mean) ** 2 for v in offsets) / (len(offsets) - 1)
        ) ** 0.5
        print(f"\n  mean offset {mean:+.2f} dB, spread (sd) {spread:.2f} dB over {len(offsets)} freqs")
        print("  negative offset = model over-predicts masking; positive = under-predicts.")
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(scored, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nwritten: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
