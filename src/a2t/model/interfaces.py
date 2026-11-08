"""Framework-neutral contracts for the model pipeline.

Concrete BEATs, DAC, and diffusion adapters belong in separate modules after
third-party APIs and checkpoints have been verified.
"""

from abc import ABC, abstractmethod
from typing import Any, Mapping, Sequence


class TargetEncoder(ABC):
    """Encode target audio into conditional features."""

    @abstractmethod
    def encode(self, audio: Any, *, sample_rate: int) -> Any:
        """Return features and preserve any padding or time-layout metadata."""
        raise NotImplementedError


class ConditionAdapter(ABC):
    """Combine target features with explicit generation conditions."""

    @abstractmethod
    def adapt(
        self,
        target_features: Any,
        conditions: Mapping[str, Any],
    ) -> Any:
        raise NotImplementedError


class AudioCodec(ABC):
    """Encode and decode environmental audio representations."""

    @abstractmethod
    def encode(self, audio: Any, *, sample_rate: int) -> Any:
        raise NotImplementedError

    @abstractmethod
    def decode(self, latent: Any, *, sample_rate: int) -> Any:
        raise NotImplementedError


class Denoiser(ABC):
    """Predict the configured diffusion training target in latent space."""

    @abstractmethod
    def predict(self, noisy_latent: Any, timestep: int, condition: Any) -> Any:
        raise NotImplementedError


class Sampler(ABC):
    """Iteratively denoise environmental-audio latents."""

    @abstractmethod
    def sample(
        self,
        denoiser: Denoiser,
        shape: Sequence[int],
        condition: Any,
        *,
        seed: int | None = None,
    ) -> Any:
        raise NotImplementedError
