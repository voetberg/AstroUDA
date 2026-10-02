from typing import Any, Callable

import pytest
import torch

from astrouda.config import Config
from astrouda.losses import adaptive_clustering_loss, compute_class_weights, weighted_cross_entropy_loss
from astrouda.training import EarlyStopping, PredictionSet, Trainer, build_optimizer, build_scheduler, resolve_device

ConfigFactory = Callable[..., Config]


def test_optimizer_matches_paper(make_tiny_config: ConfigFactory) -> None:
    config: Config = make_tiny_config(learning_rate=0.002, momentum=0.8, weight_decay=0.01, use_nesterov=True)
    optimizer: torch.optim.SGD = build_optimizer(torch.nn.Linear(2, 2), config)

    assert isinstance(optimizer, torch.optim.SGD)
    assert optimizer.defaults["nesterov"] is True
    assert optimizer.defaults["momentum"] == 0.8
    assert optimizer.defaults["weight_decay"] == 0.01
    assert optimizer.param_groups[0]["lr"] == 0.002


def test_step_lr_decays_on_schedule(make_tiny_config: ConfigFactory) -> None:
    config: Config = make_tiny_config(learning_rate=0.1, learning_rate_decay_every_epochs=3, learning_rate_decay_factor=0.1)
    optimizer: torch.optim.SGD = build_optimizer(torch.nn.Linear(2, 2), config)
    scheduler: torch.optim.lr_scheduler.StepLR = build_scheduler(optimizer, config)

    learning_rates: list[float] = []
    for _ in range(7):
        learning_rates.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()

    assert learning_rates == pytest.approx([0.1, 0.1, 0.1, 0.01, 0.01, 0.01, 0.001])


def test_trainer_steps_scheduler_once_per_epoch(make_tiny_config: ConfigFactory) -> None:
    trainer: Trainer = Trainer(
        make_tiny_config(maximum_epochs=4, learning_rate_decay_every_epochs=2, early_stopping_patience_epochs=100)
    )
    history: list[dict[str, float]] = trainer.train()

    assert [record["learning_rate"] for record in history] == pytest.approx([1e-3, 1e-3, 1e-4, 1e-4])


def test_device_auto_resolves_to_available_device() -> None:
    expected_type: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    assert resolve_device("auto").type == expected_type
    assert resolve_device("cpu").type == "cpu"


def test_mixed_precision_disabled_off_cuda(make_tiny_config: ConfigFactory, caplog: pytest.LogCaptureFixture) -> None:
    trainer: Trainer = Trainer(make_tiny_config(use_mixed_precision=True, device="cpu"))

    assert trainer.mixed_precision_enabled is False
    assert "only supported on cuda" in caplog.text


def test_class_weights_come_from_source_training_counts(
    make_tiny_config: ConfigFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    trainer: Trainer = Trainer(make_tiny_config(maximum_epochs=1))
    counts: torch.Tensor = trainer.data_loaders.source_train_class_counts
    expected_weights: torch.Tensor = compute_class_weights(
        torch.repeat_interleave(torch.arange(len(counts)), counts), len(counts)
    )
    weights_seen: list[torch.Tensor] = []

    def spy(logits: torch.Tensor, labels: torch.Tensor, class_weights: torch.Tensor) -> torch.Tensor:
        weights_seen.append(class_weights.clone())

        return weighted_cross_entropy_loss(logits, labels, class_weights)

    monkeypatch.setattr("astrouda.training.trainer.weighted_cross_entropy_loss", spy)
    trainer.train()

    assert len(weights_seen) > 0
    assert all(torch.allclose(weights, expected_weights) for weights in weights_seen)
    assert torch.allclose(trainer.class_weights.cpu(), expected_weights)


def test_bank_is_updated_after_loss_computation(make_tiny_config: ConfigFactory, monkeypatch: pytest.MonkeyPatch) -> None:
    trainer: Trainer = Trainer(make_tiny_config(maximum_epochs=1, bank_size=10_000))
    observations: list[tuple[int, int]] = []

    def spy(*arguments: Any, **keyword_arguments: Any) -> torch.Tensor:
        observations.append((arguments[1].shape[0], len(trainer.bank)))

        return adaptive_clustering_loss(*arguments, **keyword_arguments)

    monkeypatch.setattr("astrouda.training.trainer.adaptive_clustering_loss", spy)
    trainer.train()

    terms_per_batch: int = 3
    additions_per_batch: int = 2 * trainer.config.batch_size * 3
    assert len(observations) % terms_per_batch == 0
    for call_index, (rows_seen_by_loss, bank_length_at_call) in enumerate(observations):
        batch_index: int = call_index // terms_per_batch
        assert rows_seen_by_loss == bank_length_at_call == batch_index * additions_per_batch
    assert len(trainer.bank) == (len(observations) // terms_per_batch) * additions_per_batch


def test_validation_leaves_bank_and_tuner_untouched(make_tiny_config: ConfigFactory) -> None:
    trainer: Trainer = Trainer(make_tiny_config(maximum_epochs=2))
    trainer.train()

    bank_before: torch.Tensor = trainer.bank.probabilities.clone()
    tuner_before: dict[str, Any] = trainer.tuner.state_dict()
    trainer._validate()
    trainer.evaluate("test")
    trainer.collect_predictions("validation")

    assert torch.equal(trainer.bank.probabilities, bank_before)
    assert trainer.tuner.state_dict() == tuner_before


def test_source_only_mode_has_no_adaptation_activity(make_tiny_config: ConfigFactory) -> None:
    trainer: Trainer = Trainer(make_tiny_config(enable_domain_adaptation=False, maximum_epochs=2))
    history: list[dict[str, float]] = trainer.train()

    assert len(trainer.bank) == 0
    assert trainer.tuner.epoch_count == 0
    assert all(record["adaptive_clustering_loss"] == 0.0 and record["entropy_separation_loss"] == 0.0 for record in history)
    assert all(record["total_loss"] == pytest.approx(record["cross_entropy_loss"]) for record in history)


def test_source_only_target_evaluation_uses_source_domain_index(
    make_tiny_config: ConfigFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    trainer: Trainer = Trainer(make_tiny_config(enable_domain_adaptation=False))
    domain_indices: list[int] = []
    original_forward = trainer.model.forward

    def spy(images: torch.Tensor, domain_index: int) -> Any:
        domain_indices.append(domain_index)

        return original_forward(images, domain_index)

    monkeypatch.setattr(trainer.model, "forward", spy)
    trainer.evaluate("validation")

    assert set(domain_indices) == {0}


def test_domain_adaptation_target_evaluation_uses_target_domain_index(
    make_tiny_config: ConfigFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    trainer: Trainer = Trainer(make_tiny_config())
    domain_indices: list[int] = []
    original_forward = trainer.model.forward

    def spy(images: torch.Tensor, domain_index: int) -> Any:
        domain_indices.append(domain_index)

        return original_forward(images, domain_index)

    monkeypatch.setattr(trainer.model, "forward", spy)
    trainer.evaluate("test")

    assert set(domain_indices) == {0, 1}


def test_early_stopping_class_counts_strict_improvements() -> None:
    early_stopping: EarlyStopping = EarlyStopping("target_validation_accuracy", 2)

    assert early_stopping.update(0.5, 1) is True
    assert early_stopping.update(0.5, 2) is False
    assert early_stopping.should_stop is False
    assert early_stopping.update(0.4, 3) is False
    assert early_stopping.should_stop is True
    assert early_stopping.best_epoch == 1


@pytest.mark.parametrize("metric_name", ["target_validation_accuracy", "source_validation_accuracy"])
def test_trainer_stops_on_scripted_metric(
    make_tiny_config: ConfigFactory, monkeypatch: pytest.MonkeyPatch, metric_name: str
) -> None:
    trainer: Trainer = Trainer(
        make_tiny_config(maximum_epochs=10, early_stopping_patience_epochs=2, early_stopping_metric=metric_name)
    )
    scripted_values: list[float] = [0.5, 0.6, 0.55, 0.55, 0.99, 0.99]
    scripted_iterator = iter(scripted_values)

    def scripted_validation() -> dict[str, float]:
        value: float = next(scripted_iterator)

        return {
            "source_validation_accuracy": value,
            "target_validation_accuracy": value,
            "validation_cross_entropy_loss": 1.0,
        }

    monkeypatch.setattr(trainer, "_validate", scripted_validation)
    history: list[dict[str, float]] = trainer.train()

    assert len(history) == 4
    assert trainer.early_stopping.best_epoch == 2
    assert trainer.early_stopping.best_value == 0.6


def test_unsupported_early_stopping_metric_raises(make_tiny_config: ConfigFactory) -> None:
    with pytest.raises(ValueError):
        Trainer(make_tiny_config(early_stopping_metric="loss"))


def test_history_holds_plain_floats(make_tiny_config: ConfigFactory) -> None:
    history: list[dict[str, float]] = Trainer(make_tiny_config(maximum_epochs=2)).train()

    assert len(history) == 2
    for record in history:
        assert all(isinstance(value, (int, float)) and not isinstance(value, torch.Tensor) for value in record.values())
        assert all(value == value for value in record.values())


def test_collect_predictions_shapes(make_tiny_config: ConfigFactory) -> None:
    trainer: Trainer = Trainer(make_tiny_config())
    predictions: PredictionSet = trainer.collect_predictions("test")

    assert predictions.source_probabilities.shape == (len(predictions.source_labels), trainer.config.number_of_classes)
    assert predictions.target_probabilities.shape == (len(predictions.target_labels), trainer.config.number_of_classes)
    assert predictions.source_probabilities.sum(axis=1) == pytest.approx(1.0, abs=1e-5)
    assert set(trainer.evaluate("test")) == {"source_accuracy", "target_accuracy"}

    with pytest.raises(ValueError):
        trainer.collect_predictions("train")
