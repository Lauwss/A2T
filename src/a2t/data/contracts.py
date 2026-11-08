"""Source-aware metadata contracts used before slicing and feature caching."""

from dataclasses import dataclass
from enum import StrEnum


class DatasetSplit(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


@dataclass(frozen=True)
class AudioAsset:
    """An immutable source reference; processed artifacts inherit its split."""

    asset_id: str
    source_id: str
    session_id: str
    split: DatasetSplit
    category: str
    sample_rate: int
    channels: int
    duration_seconds: float
    processing_version: str
    license_reference: str

    def validate(self) -> None:
        if not self.asset_id or not self.source_id:
            raise ValueError("asset_id and source_id are required")
        if self.sample_rate <= 0 or self.channels <= 0:
            raise ValueError("sample_rate and channels must be positive")
        if self.duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive")
        if not self.license_reference:
            raise ValueError("license_reference is required")
