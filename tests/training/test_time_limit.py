import os
from types import SimpleNamespace
from typing import Any, Callable

import pytest
import torch

from astrouda.cli import main
from astrouda.config import Config
from astrouda.exit_codes import EX_TEMPFAIL
from astrouda.training import Trainer
from astrouda.training import trainer as trainer_module

ConfigFactory = Callable[..., Config]
COMMON_OVERRIDES: dict[str, Any] = dict(early_stopping_patience_epochs=100, entropy_boundary_patience_epochs=0)
SIMULATED_EPOCH_SECONDS: float = 10.0


class FakeClock:
    def __init__(self) -> None:
        self.current_seconds: float = 0.0

    def monotonic(self) -> float:
        return self.current_seconds


def install_fake_clock(trainer: Trainer, monkeypatch: pytest.MonkeyPatch, clock: FakeClock) -> None:
    "Every training epoch takes SIMULATED_EPOCH_SECONDS on the fake clock, no sleeping."
    monkeypatch.setattr(trainer_module, "time", SimpleNamespace(monotonic=clock.monotonic))
    original_train_epoch = trainer._train_epoch

    def timed_train_epoch() -> dict[str, float]:
        clock.current_seconds += SIMULATED_EPOCH_SECONDS

        return original_train_epoch()

    monkeypatch.setattr(trainer, "_train_epoch", timed_train_epoch)


def test_time_limit_defaults_preserve_current_behaviour() -> None:
    config: Config = Config()

    assert config.maximum_wall_time_seconds is None
    assert config.wall_time_epoch_margin == 1.5
    assert config.auto_resume is False


def test_finished_is_true_after_maximum_epochs_and_never_time_limited(make_tiny_config: ConfigFactory) -> None:
    trainer: Trainer = Trainer(make_tiny_config(maximum_epochs=1))

    assert trainer.finished is False
    trainer.train()

    assert trainer.finished is True
    assert trainer.stopped_for_time_limit is False


def test_finished_is_true_after_early_stopping(make_tiny_config: ConfigFactory) -> None:
    trainer: Trainer = Trainer(make_tiny_config(maximum_epochs=50, early_stopping_patience_epochs=1))
    trainer.train()

    assert trainer.early_stopping.should_stop
    assert trainer.completed_epochs < 50
    assert trainer.finished is True


def test_time_limited_chain_matches_uninterrupted_run(
    make_tiny_config: ConfigFactory, tmp_path: "os.PathLike[str]", monkeypatch: pytest.MonkeyPatch
) -> None:
    uninterrupted_trainer: Trainer = Trainer(
        make_tiny_config(maximum_epochs=5, output_directory=str(tmp_path / "uninterrupted"), **COMMON_OVERRIDES)
    )
    uninterrupted_history: list[dict[str, float]] = uninterrupted_trainer.train()

    epochs_completed_per_invocation: list[int] = []
    chained_trainer: Trainer
    while True:
        chained_trainer = Trainer(
            make_tiny_config(
                maximum_epochs=5,
                output_directory=str(tmp_path / "chained"),
                auto_resume=True,
                maximum_wall_time_seconds=25.0,
                checkpoint_every_epochs=3,
                **COMMON_OVERRIDES,
            )
        )
        epochs_before_invocation: int = chained_trainer.completed_epochs
        install_fake_clock(chained_trainer, monkeypatch, FakeClock())
        chained_trainer.train()
        epochs_completed_per_invocation.append(chained_trainer.completed_epochs - epochs_before_invocation)

        if chained_trainer.finished:
            break
        assert chained_trainer.stopped_for_time_limit is True
        assert len(epochs_completed_per_invocation) < 10

    assert epochs_completed_per_invocation == [2, 2, 1]
    assert chained_trainer.stopped_for_time_limit is False
    assert chained_trainer.history == uninterrupted_history
    for parameter_name, parameter in uninterrupted_trainer.model.state_dict().items():
        assert torch.equal(parameter, chained_trainer.model.state_dict()[parameter_name]), parameter_name
    assert torch.equal(uninterrupted_trainer.bank.probabilities, chained_trainer.bank.probabilities)
    assert uninterrupted_trainer.tuner.state_dict() == chained_trainer.tuner.state_dict()
    assert uninterrupted_trainer.early_stopping.state_dict() == chained_trainer.early_stopping.state_dict()


def test_stop_checkpoints_last_epoch_regardless_of_cadence(
    make_tiny_config: ConfigFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    config: Config = make_tiny_config(
        maximum_epochs=5, maximum_wall_time_seconds=25.0, checkpoint_every_epochs=100, **COMMON_OVERRIDES
    )
    trainer: Trainer = Trainer(config)
    install_fake_clock(trainer, monkeypatch, FakeClock())
    trainer.train()

    checkpoint: dict[str, Any] = torch.load(trainer.checkpoint_path, weights_only=False)
    assert trainer.stopped_for_time_limit is True
    assert checkpoint["epoch"] == trainer.completed_epochs == 2


def test_resuming_a_finished_run_returns_without_training(
    make_tiny_config: ConfigFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    config: Config = make_tiny_config(maximum_epochs=2, auto_resume=True)
    first_history: list[dict[str, float]] = Trainer(config).train()

    resumed_trainer: Trainer = Trainer(config)

    def fail_if_called() -> dict[str, float]:
        raise AssertionError("a finished run must not train again")

    monkeypatch.setattr(resumed_trainer, "_train_epoch", fail_if_called)

    assert resumed_trainer.finished is True
    assert resumed_trainer.train() == first_history
    assert resumed_trainer.stopped_for_time_limit is False


def test_auto_resume_without_checkpoint_starts_from_scratch(make_tiny_config: ConfigFactory) -> None:
    trainer: Trainer = Trainer(make_tiny_config(maximum_epochs=1, auto_resume=True))

    assert trainer.completed_epochs == 0


def test_a_budget_smaller_than_one_epoch_still_makes_progress(make_tiny_config: ConfigFactory) -> None:
    trainer: Trainer = Trainer(make_tiny_config(maximum_epochs=3, maximum_wall_time_seconds=0.0, **COMMON_OVERRIDES))
    trainer.train()

    assert trainer.completed_epochs == 1
    assert trainer.stopped_for_time_limit is True


def test_train_cli_exit_codes(tiny_config_path: Any) -> None:
    base_arguments: list[str] = ["train", "--config", str(tiny_config_path), "--set", "maximum_epochs=2"]

    assert main([*base_arguments, "--set", "maximum_wall_time_seconds=0", "--set", "auto_resume=true"]) == EX_TEMPFAIL == 75
    assert main([*base_arguments, "--set", "maximum_wall_time_seconds=0", "--set", "auto_resume=true"]) == 0
    assert main([*base_arguments, "--set", "auto_resume=true"]) == 0
    assert main([*base_arguments, "--set", "not_a_key=1"]) == 1
