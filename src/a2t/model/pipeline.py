"""Model pipeline orchestration without binding to a specific ML framework."""

from dataclasses import dataclass
from typing import Any, Mapping

from .interfaces import AudioCodec, ConditionAdapter, Denoiser, Sampler, TargetEncoder


@dataclass(frozen=True)
class GenerationRequest:
    """Explicit generation conditions; target audio is not mixed into output."""

    target_audio: Any
    target_sample_rate: int
    output_sample_rate: int
    conditions: Mapping[str, Any]
    latent_shape: tuple[int, ...]
    seed: int | None = None


class EnvironmentalAudioGenerator:
    """Connect the six model boundaries while keeping evaluation separate."""

    def __init__(
        self,
        target_encoder: TargetEncoder,
        condition_adapter: ConditionAdapter,
        codec: AudioCodec,
        denoiser: Denoiser,
        sampler: Sampler,
    ) -> None:
        self.target_encoder = target_encoder
        self.condition_adapter = condition_adapter
        self.codec = codec
        self.denoiser = denoiser
        self.sampler = sampler

    def generate(self, request: GenerationRequest) -> Any:
        if request.target_sample_rate <= 0 or request.output_sample_rate <= 0:
            raise ValueError("sample rates must be positive")
        if not request.latent_shape or any(dimension <= 0 for dimension in request.latent_shape):
            raise ValueError("latent_shape must contain positive dimensions")
        features = self.target_encoder.encode(
            request.target_audio,
            sample_rate=request.target_sample_rate,
        )
        condition = self.condition_adapter.adapt(features, request.conditions)
        latent = self.sampler.sample(
            self.denoiser,
            request.latent_shape,
            condition,
            seed=request.seed,
        )
        return self.codec.decode(latent, sample_rate=request.output_sample_rate)
