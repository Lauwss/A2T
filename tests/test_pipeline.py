from a2t.model import (
    AudioCodec,
    ConditionAdapter,
    Denoiser,
    EnvironmentalAudioGenerator,
    GenerationRequest,
    Sampler,
    TargetEncoder,
)


class FakeTargetEncoder(TargetEncoder):
    def encode(self, audio, *, sample_rate):
        return (audio, sample_rate)


class FakeConditionAdapter(ConditionAdapter):
    def adapt(self, target_features, conditions):
        return target_features, conditions


class FakeCodec(AudioCodec):
    def encode(self, audio, *, sample_rate):
        return audio

    def decode(self, latent, *, sample_rate):
        return latent, sample_rate


class FakeDenoiser(Denoiser):
    def predict(self, noisy_latent, timestep, condition):
        return noisy_latent


class FakeSampler(Sampler):
    def sample(self, denoiser, shape, condition, *, seed=None):
        return shape, condition, seed


def test_generator_keeps_target_conditioning_separate_from_output() -> None:
    generator = EnvironmentalAudioGenerator(
        FakeTargetEncoder(),
        FakeConditionAdapter(),
        FakeCodec(),
        FakeDenoiser(),
        FakeSampler(),
    )

    output = generator.generate(
        GenerationRequest(
            target_audio=[0.1, 0.2],
            target_sample_rate=16_000,
            output_sample_rate=24_000,
            conditions={"category": "rain"},
            latent_shape=(2, 4),
            seed=7,
        )
    )

    assert output[1] == 24_000
    assert output[0][0] == (2, 4)
    assert output[0][2] == 7
