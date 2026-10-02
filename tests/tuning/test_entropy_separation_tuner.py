import json
import math

import pytest

from astrouda.config import Config
from astrouda.tuning import EntropySeparationTuner, TunerPhase


def feed_non_improving(tuner: EntropySeparationTuner, number_of_epochs: int, loss: float = 10.0) -> None:
    for _ in range(number_of_epochs):
        tuner.update(loss)


def make_primed_tuner(**overrides: object) -> EntropySeparationTuner:
    tuner: EntropySeparationTuner = EntropySeparationTuner(Config(**overrides))
    tuner.update(1.0)

    return tuner


def test_defaults_three_classes() -> None:
    tuner: EntropySeparationTuner = EntropySeparationTuner(Config(number_of_classes=3))

    assert tuner.entropy_boundary == pytest.approx(math.log(3) / 2)
    assert tuner.confidence_margin == 0.4
    assert type(tuner.entropy_boundary) is float
    assert type(tuner.confidence_margin) is float


def test_defaults_ten_classes() -> None:
    tuner: EntropySeparationTuner = EntropySeparationTuner(Config(number_of_classes=10))

    assert tuner.entropy_boundary == pytest.approx(math.log(10) / 2)
    assert tuner.confidence_margin == 1.3


def test_margin_default_boundary_at_five_and_six_classes() -> None:
    assert EntropySeparationTuner(Config(number_of_classes=5)).confidence_margin == 0.4
    assert EntropySeparationTuner(Config(number_of_classes=6)).confidence_margin == 1.3


def test_config_overrides() -> None:
    tuner: EntropySeparationTuner = EntropySeparationTuner(
        Config(entropy_boundary_initial=0.7, confidence_margin_initial=0.9)
    )

    assert tuner.entropy_boundary == 0.7
    assert tuner.confidence_margin == 0.9


def test_improving_losses_never_change_parameters() -> None:
    tuner: EntropySeparationTuner = EntropySeparationTuner(Config())
    initial_boundary: float = tuner.entropy_boundary
    initial_margin: float = tuner.confidence_margin

    for epoch_loss in [5.0 - 0.1 * epoch for epoch in range(50)]:
        tuner.update(epoch_loss)

    assert tuner.entropy_boundary == initial_boundary
    assert tuner.confidence_margin == initial_margin
    assert tuner.phase == TunerPhase.ADJUST_BOUNDARY


def test_boundary_fires_on_sixth_not_fifth_non_improving_epoch() -> None:
    tuner: EntropySeparationTuner = make_primed_tuner()
    initial_boundary: float = tuner.entropy_boundary

    feed_non_improving(tuner, 5)
    assert tuner.entropy_boundary == initial_boundary
    assert tuner.phase == TunerPhase.ADJUST_BOUNDARY

    feed_non_improving(tuner, 1)
    assert tuner.entropy_boundary == pytest.approx(initial_boundary + 0.3)
    assert tuner.phase == TunerPhase.ADJUST_MARGIN


def test_margin_fires_on_third_not_second_further_epoch() -> None:
    tuner: EntropySeparationTuner = make_primed_tuner()
    feed_non_improving(tuner, 6)
    initial_margin: float = tuner.confidence_margin

    feed_non_improving(tuner, 2)
    assert tuner.confidence_margin == initial_margin
    assert tuner.phase == TunerPhase.ADJUST_MARGIN

    feed_non_improving(tuner, 1)
    assert tuner.confidence_margin == pytest.approx(initial_margin + 0.3)
    assert tuner.phase == TunerPhase.DOUBLE_MARGIN


def test_margin_doubles_on_third_further_epoch_and_step_advances() -> None:
    tuner: EntropySeparationTuner = make_primed_tuner()
    feed_non_improving(tuner, 9)
    margin_after_step: float = tuner.confidence_margin

    feed_non_improving(tuner, 2)
    assert tuner.confidence_margin == margin_after_step
    assert tuner.step_index == 0

    feed_non_improving(tuner, 1)
    assert tuner.confidence_margin == pytest.approx(margin_after_step * 2)
    assert tuner.step_index == 1
    assert tuner.phase == TunerPhase.ADJUST_BOUNDARY


def test_phases_visited_in_order() -> None:
    tuner: EntropySeparationTuner = make_primed_tuner()
    feed_non_improving(tuner, 12)

    visited_phases: list[str] = []
    for record in tuner.history:
        if not visited_phases or visited_phases[-1] != record[4]:
            visited_phases.append(record[4])

    assert visited_phases == ["adjust_boundary", "adjust_margin", "double_margin", "adjust_boundary"]


def test_step_index_wraps_after_fourth_step() -> None:
    tuner: EntropySeparationTuner = make_primed_tuner(entropy_boundary_initial=1.0, confidence_margin_initial=1.0)

    feed_non_improving(tuner, 12 * 4)
    assert tuner.step_index == 0

    boundary_before_wrap: float = tuner.entropy_boundary
    feed_non_improving(tuner, 6)
    assert tuner.entropy_boundary == pytest.approx(boundary_before_wrap + 0.3)


def test_full_cycle_values() -> None:
    tuner: EntropySeparationTuner = make_primed_tuner(entropy_boundary_initial=1.0, confidence_margin_initial=1.0)

    feed_non_improving(tuner, 12)
    assert tuner.entropy_boundary == pytest.approx(1.3)
    assert tuner.confidence_margin == pytest.approx(2.6)

    feed_non_improving(tuner, 12)
    assert tuner.entropy_boundary == pytest.approx(1.0)
    assert tuner.confidence_margin == pytest.approx(4.6)


def test_improvement_mid_cycle_resets_counter_but_keeps_phase() -> None:
    tuner: EntropySeparationTuner = make_primed_tuner()
    initial_boundary: float = tuner.entropy_boundary

    feed_non_improving(tuner, 5)
    tuner.update(0.5)
    assert tuner.non_improving_epochs == 0

    feed_non_improving(tuner, 5)
    assert tuner.entropy_boundary == initial_boundary

    feed_non_improving(tuner, 1)
    assert tuner.entropy_boundary == pytest.approx(initial_boundary + 0.3)

    feed_non_improving(tuner, 2)
    tuner.update(0.1)
    feed_non_improving(tuner, 2)
    assert tuner.phase == TunerPhase.ADJUST_MARGIN


def test_equal_loss_is_non_improving() -> None:
    tuner: EntropySeparationTuner = make_primed_tuner()

    tuner.update(1.0)

    assert tuner.non_improving_epochs == 1


def test_first_epoch_is_improving() -> None:
    tuner: EntropySeparationTuner = EntropySeparationTuner(Config())

    tuner.update(3.0)

    assert tuner.minimum_loss == 3.0
    assert tuner.non_improving_epochs == 0


def test_nan_loss_is_non_improving() -> None:
    tuner: EntropySeparationTuner = make_primed_tuner()

    tuner.update(float("nan"))

    assert tuner.non_improving_epochs == 1
    assert tuner.minimum_loss == 1.0


def test_boundary_floor_clamps_negative_step() -> None:
    tuner: EntropySeparationTuner = make_primed_tuner(
        entropy_boundary_initial=0.1, entropy_step_values=[-0.5], entropy_boundary_floor=0.0
    )

    feed_non_improving(tuner, 6)

    assert tuner.entropy_boundary == 0.0


def test_margin_floor_clamps_negative_step() -> None:
    tuner: EntropySeparationTuner = make_primed_tuner(
        confidence_margin_initial=0.1, entropy_step_values=[-0.5], confidence_margin_floor=0.05
    )

    feed_non_improving(tuner, 9)

    assert tuner.confidence_margin == 0.05


def test_initial_values_are_clamped() -> None:
    tuner: EntropySeparationTuner = EntropySeparationTuner(Config(number_of_classes=1, confidence_margin_initial=-1.0))

    assert tuner.entropy_boundary == 0.0
    assert tuner.confidence_margin == Config().confidence_margin_floor


def test_custom_patience_thresholds() -> None:
    tuner: EntropySeparationTuner = make_primed_tuner(
        entropy_boundary_patience_epochs=1, confidence_margin_patience_epochs=0
    )
    initial_boundary: float = tuner.entropy_boundary

    feed_non_improving(tuner, 1)
    assert tuner.entropy_boundary == initial_boundary

    feed_non_improving(tuner, 1)
    assert tuner.entropy_boundary == pytest.approx(initial_boundary + 0.3)

    feed_non_improving(tuner, 1)
    assert tuner.phase == TunerPhase.DOUBLE_MARGIN


def test_state_dict_round_trip_reproduces_future_trajectory() -> None:
    scripted_prefix: list[float] = [3.0, 2.0, 2.5, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1]
    scripted_suffix: list[float] = [3.0] * 15 + [1.0] + [5.0] * 30

    original_tuner: EntropySeparationTuner = EntropySeparationTuner(Config())
    for epoch_loss in scripted_prefix:
        original_tuner.update(epoch_loss)

    restored_tuner: EntropySeparationTuner = EntropySeparationTuner(Config())
    restored_tuner.load_state_dict(json.loads(json.dumps(original_tuner.state_dict())))

    for epoch_loss in scripted_suffix:
        original_tuner.update(epoch_loss)
        restored_tuner.update(epoch_loss)
        assert restored_tuner.entropy_boundary == original_tuner.entropy_boundary
        assert restored_tuner.confidence_margin == original_tuner.confidence_margin

    assert restored_tuner.history == original_tuner.history
    assert restored_tuner.state_dict() == original_tuner.state_dict()


def test_state_dict_before_first_update_is_json_serialisable() -> None:
    tuner: EntropySeparationTuner = EntropySeparationTuner(Config())

    restored_tuner: EntropySeparationTuner = EntropySeparationTuner(Config())
    restored_tuner.load_state_dict(json.loads(json.dumps(tuner.state_dict())))
    restored_tuner.update(1.0)

    assert restored_tuner.minimum_loss == 1.0


def test_history_records_every_epoch() -> None:
    tuner: EntropySeparationTuner = EntropySeparationTuner(Config())

    tuner.update(2.0)
    tuner.update(3.0)

    assert [record[0] for record in tuner.history] == [1, 2]
    assert tuner.history[1][1] == 3.0
    assert len(tuner.history[0]) == 5
