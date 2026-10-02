import json
import os
from typing import Callable

import torch

from astrouda.config import Config
from astrouda.training import Trainer

ConfigFactory = Callable[..., Config]


def test_resume_reproduces_uninterrupted_run(make_tiny_config: ConfigFactory, tmp_path: "os.PathLike[str]") -> None:
    common: dict[str, object] = dict(early_stopping_patience_epochs=100, entropy_boundary_patience_epochs=0)

    uninterrupted_trainer: Trainer = Trainer(
        make_tiny_config(maximum_epochs=4, output_directory=str(tmp_path / "uninterrupted"), **common)
    )
    uninterrupted_history: list[dict[str, float]] = uninterrupted_trainer.train()

    Trainer(make_tiny_config(maximum_epochs=2, output_directory=str(tmp_path / "first_half"), **common)).train()
    resumed_trainer: Trainer = Trainer(
        make_tiny_config(
            maximum_epochs=4,
            output_directory=str(tmp_path / "second_half"),
            resume_from_checkpoint=str(tmp_path / "first_half" / "checkpoint.pt"),
            **common,
        )
    )
    assert resumed_trainer.completed_epochs == 2
    resumed_history: list[dict[str, float]] = resumed_trainer.train()

    assert resumed_history == uninterrupted_history
    for parameter_name, parameter in uninterrupted_trainer.model.state_dict().items():
        assert torch.equal(parameter, resumed_trainer.model.state_dict()[parameter_name]), parameter_name
    assert torch.equal(uninterrupted_trainer.bank.probabilities, resumed_trainer.bank.probabilities)
    assert uninterrupted_trainer.tuner.state_dict() == resumed_trainer.tuner.state_dict()
    assert uninterrupted_trainer.early_stopping.state_dict() == resumed_trainer.early_stopping.state_dict()


def test_run_writes_all_outputs(make_tiny_config: ConfigFactory) -> None:
    config: Config = make_tiny_config(maximum_epochs=2)
    Trainer(config).train()

    for file_name in ("config.json", "history.json", "checkpoint.pt", "best_model.pt"):
        assert os.path.exists(os.path.join(config.output_directory, file_name)), file_name

    with open(os.path.join(config.output_directory, "config.json")) as config_file:
        assert json.load(config_file) == config.to_dictionary()
    with open(os.path.join(config.output_directory, "history.json")) as history_file:
        assert len(json.load(history_file)) == 2


def test_checkpoint_contains_every_component(make_tiny_config: ConfigFactory) -> None:
    config: Config = make_tiny_config(maximum_epochs=1)
    Trainer(config).train()
    checkpoint: dict = torch.load(os.path.join(config.output_directory, "checkpoint.pt"), weights_only=False)

    expected_keys: set[str] = {
        "model", "optimizer", "scheduler", "tuner", "bank", "epoch", "early_stopping", "history", "random_states", "config",
    }
    assert expected_keys <= set(checkpoint)
    assert checkpoint["epoch"] == 1


def test_checkpoint_every_epochs_controls_cadence(make_tiny_config: ConfigFactory, monkeypatch) -> None:
    trainer: Trainer = Trainer(make_tiny_config(maximum_epochs=4, checkpoint_every_epochs=3, early_stopping_patience_epochs=100))
    saved_epochs: list[int] = []
    monkeypatch.setattr(trainer, "_save", lambda: saved_epochs.append(trainer.completed_epochs))
    trainer.train()

    assert saved_epochs == [3, 4]
