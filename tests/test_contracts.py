"""Regression: DatasetSplit must behave like enum.StrEnum on Python 3.10."""

from a2t.data.contracts import AudioAsset, DatasetSplit


def test_dataset_split_str_returns_value() -> None:
    assert str(DatasetSplit.TRAIN) == "train"
    assert f"{DatasetSplit.VALIDATION}" == "validation"
    assert DatasetSplit("test") is DatasetSplit.TEST


def test_audio_asset_requires_license_reference() -> None:
    base = dict(
        asset_id="a1",
        source_id="s1",
        session_id="s1",
        split=DatasetSplit.TRAIN,
        category="rain_stable",
        sample_rate=44100,
        channels=1,
        duration_seconds=8.0,
        processing_version="raw-v1",
    )
    asset = AudioAsset(**base, license_reference="CC0")
    asset.validate()
    missing = AudioAsset(**base, license_reference="")
    try:
        missing.validate()
    except ValueError:
        return
    raise AssertionError("empty license_reference must raise ValueError")
