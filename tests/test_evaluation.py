from a2t.evaluation import EvaluationInput, MaskingEvaluator


def test_masking_evaluator_returns_frame_level_proxy_scores() -> None:
    frame_size = 8
    target = [1.0] * frame_size
    masker = [2.0] * frame_size
    mixture = [3.0] * frame_size

    scores = MaskingEvaluator(frame_size=frame_size, hop_size=frame_size).score(
        EvaluationInput(target, masker, mixture, sample_rate=24_000)
    )

    assert len(scores.threshold_exposure_db) == 1
    assert scores.threshold_exposure_db[0] < 0
    assert scores.representation_distance_db[0] > 0


def test_masking_evaluator_rejects_misaligned_signals() -> None:
    evaluator = MaskingEvaluator(frame_size=4, hop_size=2)

    try:
        evaluator.score(EvaluationInput([0.0] * 4, [0.0] * 3, [0.0] * 4, 24_000))
    except ValueError as error:
        assert "equal lengths" in str(error)
    else:
        raise AssertionError("misaligned signals should be rejected")
