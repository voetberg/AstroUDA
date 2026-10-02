import json
import os
import random
import shutil
import time
from typing import Any, ContextManager, Optional

import numpy as np
import torch
import torch.nn.functional as functional
from torch import Tensor

from astrouda.config import Config
from astrouda.data import DataLoaders, DomainAdaptationBatch, DomainAdaptationLoader, build_data_loaders
from astrouda.logging_utils import get_logger
from astrouda.losses import (
    ProbabilityBank,
    adaptive_clustering_loss,
    compute_class_weights,
    entropy_separation_loss,
    total_loss,
    weighted_cross_entropy_loss,
)
from astrouda.models import DomainAdaptationNetwork, build_model
from astrouda.training.checkpoint import capture_random_states, load_checkpoint, restore_random_states, save_checkpoint
from astrouda.training.device import resolve_device
from astrouda.training.early_stopping import EarlyStopping
from astrouda.training.optimization import build_optimizer, build_scheduler
from astrouda.training.predictions import PredictionSet
from astrouda.tuning import EntropySeparationTuner

SOURCE_DOMAIN_INDEX: int = 0
TARGET_DOMAIN_INDEX: int = 1
SUPPORTED_EARLY_STOPPING_METRICS: tuple[str, ...] = ("target_validation_accuracy", "source_validation_accuracy")
SUPPORTED_SPLITS: tuple[str, ...] = ("validation", "test")


class Trainer:
    """
    Trains the network with the paper's recipe: SGD (Nesterov) + StepLR, weighted CE on the source and, in domain
    adaptation mode, lambda * (L_AC + L_ES) on the target.

    Domain adaptation step (enable_domain_adaptation True), per batch:
        1. source forward with domain index 0, target forward with domain index 1. When augmented views exist the
           target views go through domain index 1 with gradients, the source views through domain index 0 without
           gradients (they only feed the bank and the source batch norm statistics).
        2. L_CE is the weighted cross entropy of the un-augmented source batch only.
        3. L_AC and L_ES are each computed on the un-augmented target probabilities and on every target view's
           probabilities, then averaged with equal weight over those 1 + (number of views) terms. L_AC compares each
           set against the ProbabilityBank, L_ES uses the tuner's current rho and m.
        4. All L_AC terms are computed BEFORE anything is added to the bank for this batch. Afterwards the detached
           source, target and view probabilities are added.
        5. L = L_CE + domain_adaptation_weight * (L_AC + L_ES), then backward and one SGD step.

    Source-only mode (enable_domain_adaptation False): L = L_CE, no bank, no tuner activity, target data is never
    forwarded in training. Target batch norm is never trained then, so target accuracy at evaluation uses domain
    index 0 (the source statistics).

    Per epoch: train, tuner.update(mean training total loss) in domain adaptation mode, validate (eval mode, no
    gradients, bank and tuner untouched), scheduler.step(), early stopping, checkpointing.
    Only python floats are accumulated from per-batch losses, so no autograd graph outlives its batch.

    Files in output_directory: config.json, history.json, checkpoint.pt (latest, resumable) and best_model.pt.
    """

    def __init__(self, config: Config) -> None:
        self.config: Config = config
        self.logger = get_logger(__name__, config)

        if self.config.early_stopping_metric not in SUPPORTED_EARLY_STOPPING_METRICS:
            raise ValueError(
                f"early_stopping_metric must be one of {SUPPORTED_EARLY_STOPPING_METRICS}, "
                f"got '{self.config.early_stopping_metric}'"
            )

        self._seed_everything()
        self.device: torch.device = resolve_device(self.config.device)
        self.mixed_precision_enabled: bool = self._resolve_mixed_precision()

        self.data_loaders: DataLoaders = build_data_loaders(self.config)
        self.model: DomainAdaptationNetwork = build_model(self.config).to(self.device)
        self.class_weights: Tensor = compute_class_weights(
            torch.repeat_interleave(
                torch.arange(self.config.number_of_classes), self.data_loaders.source_train_class_counts
            ),
            self.config.number_of_classes,
        ).to(self.device)
        self.bank: ProbabilityBank = ProbabilityBank(self.config.bank_size, self.config.number_of_classes, self.device)
        self.tuner: EntropySeparationTuner = EntropySeparationTuner(self.config)
        self.optimizer: torch.optim.SGD = build_optimizer(self.model, self.config)
        self.scheduler: torch.optim.lr_scheduler.StepLR = build_scheduler(self.optimizer, self.config)
        self.gradient_scaler: torch.amp.GradScaler = torch.amp.GradScaler("cuda", enabled=self.mixed_precision_enabled)
        self.early_stopping: EarlyStopping = EarlyStopping(
            self.config.early_stopping_metric, self.config.early_stopping_patience_epochs
        )

        self.completed_epochs: int = 0
        self.history: list[dict[str, float]] = []

        os.makedirs(self.config.output_directory, exist_ok=True)
        if self.config.resume_from_checkpoint is not None:
            self._restore(self.config.resume_from_checkpoint)

    @property
    def checkpoint_path(self) -> str:
        return os.path.join(self.config.output_directory, "checkpoint.pt")

    @property
    def best_model_path(self) -> str:
        return os.path.join(self.config.output_directory, "best_model.pt")

    @property
    def history_path(self) -> str:
        return os.path.join(self.config.output_directory, "history.json")

    def _seed_everything(self) -> None:
        random.seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)
        torch.manual_seed(self.config.random_seed)
        self.logger.debug(f"Seeded python, numpy and torch with {self.config.random_seed}")

    def _resolve_mixed_precision(self) -> bool:
        if self.config.use_mixed_precision and self.device.type != "cuda":
            self.logger.warning(f"use_mixed_precision is only supported on cuda, disabling it on {self.device.type}")

            return False

        return self.config.use_mixed_precision

    def _autocast(self) -> ContextManager[None]:
        return torch.autocast(device_type=self.device.type, enabled=self.mixed_precision_enabled)

    def _to_device(self, tensor: Optional[Tensor]) -> Optional[Tensor]:
        if tensor is None:
            return None

        return tensor.to(self.device, non_blocking=self.config.pin_memory)

    def _logits(self, images: Tensor, domain_index: int) -> Tensor:
        with self._autocast():
            logits: Tensor = self.model(images, domain_index).logits

        return logits.float()

    def _domain_adaptation_losses(self, target_probability_sets: list[Tensor]) -> tuple[Tensor, Tensor]:
        "Mean of the L_AC and of the L_ES terms over the un-augmented target probabilities and each view."
        clustering_terms: list[Tensor] = []
        separation_terms: list[Tensor] = []

        for probabilities in target_probability_sets:
            clustering_terms.append(
                adaptive_clustering_loss(
                    probabilities, self.bank.probabilities, self.config.top_k, self.config.probability_epsilon
                )
            )
            separation_terms.append(
                entropy_separation_loss(
                    probabilities,
                    self.tuner.entropy_boundary,
                    self.tuner.confidence_margin,
                    self.config.probability_epsilon,
                )
            )

        return torch.stack(clustering_terms).mean(), torch.stack(separation_terms).mean()

    def _train_step(self, batch: DomainAdaptationBatch) -> dict[str, float]:
        source_images: Tensor = self._to_device(batch.source_images)
        source_labels: Tensor = self._to_device(batch.source_labels)

        self.optimizer.zero_grad(set_to_none=True)
        source_logits: Tensor = self._logits(source_images, SOURCE_DOMAIN_INDEX)
        cross_entropy: Tensor = weighted_cross_entropy_loss(source_logits, source_labels, self.class_weights)

        if not self.config.enable_domain_adaptation:
            combined_loss: Tensor = cross_entropy
            clustering_value: float = 0.0
            separation_value: float = 0.0
        else:
            target_probabilities: Tensor = functional.softmax(
                self._logits(self._to_device(batch.target_images), TARGET_DOMAIN_INDEX), dim=1
            )
            target_probability_sets: list[Tensor] = [target_probabilities]
            bank_additions: list[Tensor] = [functional.softmax(source_logits.detach(), dim=1), target_probabilities.detach()]

            if batch.target_view_one is not None and batch.target_view_two is not None:
                for target_view in (batch.target_view_one, batch.target_view_two):
                    target_probability_sets.append(
                        functional.softmax(self._logits(self._to_device(target_view), TARGET_DOMAIN_INDEX), dim=1)
                    )
                bank_additions.extend(probabilities.detach() for probabilities in target_probability_sets[1:])
            if batch.source_view_one is not None and batch.source_view_two is not None:
                with torch.no_grad():
                    for source_view in (batch.source_view_one, batch.source_view_two):
                        bank_additions.append(
                            functional.softmax(self._logits(self._to_device(source_view), SOURCE_DOMAIN_INDEX), dim=1)
                        )

            clustering_loss, separation_loss = self._domain_adaptation_losses(target_probability_sets)
            combined_loss = total_loss(
                cross_entropy, clustering_loss, separation_loss, self.config.domain_adaptation_weight
            )
            clustering_value = clustering_loss.item()
            separation_value = separation_loss.item()

            for probabilities in bank_additions:
                self.bank.add(probabilities)

        self.gradient_scaler.scale(combined_loss).backward()
        self.gradient_scaler.step(self.optimizer)
        self.gradient_scaler.update()

        return {
            "cross_entropy_loss": cross_entropy.item(),
            "adaptive_clustering_loss": clustering_value,
            "entropy_separation_loss": separation_value,
            "total_loss": combined_loss.item(),
        }

    def _train_epoch(self) -> dict[str, float]:
        self.model.train()
        self.data_loaders.train.set_epoch(self.completed_epochs)

        loss_sums: dict[str, float] = {
            "cross_entropy_loss": 0.0,
            "adaptive_clustering_loss": 0.0,
            "entropy_separation_loss": 0.0,
            "total_loss": 0.0,
        }
        batch_count: int = 0
        for batch in self.data_loaders.train:
            step_losses: dict[str, float] = self._train_step(batch)
            for loss_name, loss_value in step_losses.items():
                loss_sums[loss_name] += loss_value
            batch_count += 1
            self.logger.debug(f"Epoch {self.completed_epochs + 1} batch {batch_count}: {step_losses}")

        return {loss_name: loss_sum / batch_count for loss_name, loss_sum in loss_sums.items()}

    @property
    def _target_evaluation_domain_index(self) -> int:
        return TARGET_DOMAIN_INDEX if self.config.enable_domain_adaptation else SOURCE_DOMAIN_INDEX

    def _loader_for_split(self, split: str) -> DomainAdaptationLoader:
        if split not in SUPPORTED_SPLITS:
            raise ValueError(f"split must be one of {SUPPORTED_SPLITS}, got '{split}'")

        return getattr(self.data_loaders, split)

    def _run_inference(self, split: str) -> tuple[PredictionSet, float]:
        "Eval mode, no gradients. Touches neither the bank nor the tuner. Returns predictions and mean source CE."
        self.model.eval()
        collected: dict[str, list[Tensor]] = {
            name: []
            for name in (
                "source_labels",
                "source_probabilities",
                "target_labels",
                "target_probabilities",
            )
        }
        source_cross_entropy_sum: float = 0.0

        with torch.no_grad():
            for batch in self._loader_for_split(split):
                if batch.source_images.shape[0] > 0:
                    source_logits: Tensor = self._logits(self._to_device(batch.source_images), SOURCE_DOMAIN_INDEX)
                    source_labels: Tensor = self._to_device(batch.source_labels)
                    source_cross_entropy_sum += functional.cross_entropy(source_logits, source_labels, reduction="sum").item()
                    collected["source_labels"].append(source_labels.cpu())
                    collected["source_probabilities"].append(functional.softmax(source_logits, dim=1).cpu())
                if batch.target_images.shape[0] > 0:
                    target_logits: Tensor = self._logits(
                        self._to_device(batch.target_images), self._target_evaluation_domain_index
                    )
                    collected["target_labels"].append(batch.target_labels.cpu())
                    collected["target_probabilities"].append(functional.softmax(target_logits, dim=1).cpu())

        source_probabilities: np.ndarray = torch.cat(collected["source_probabilities"]).numpy()
        target_probabilities: np.ndarray = torch.cat(collected["target_probabilities"]).numpy()
        prediction_set: PredictionSet = PredictionSet(
            source_labels=torch.cat(collected["source_labels"]).numpy(),
            source_predicted_classes=source_probabilities.argmax(axis=1).astype(np.int64),
            source_probabilities=source_probabilities,
            target_labels=torch.cat(collected["target_labels"]).numpy(),
            target_predicted_classes=target_probabilities.argmax(axis=1).astype(np.int64),
            target_probabilities=target_probabilities,
        )
        mean_source_cross_entropy: float = source_cross_entropy_sum / len(prediction_set.source_labels)
        self.logger.debug(
            f"Inference on {split}: {len(prediction_set.source_labels)} source, {len(prediction_set.target_labels)} target samples, "
            f"source accuracy {prediction_set.source_accuracy:.4f}, target accuracy {prediction_set.target_accuracy:.4f}, "
            f"source CE {mean_source_cross_entropy:.4f}"
        )

        return prediction_set, mean_source_cross_entropy

    def collect_predictions(self, split: str) -> PredictionSet:
        "Labels, predicted classes and softmax probabilities for 'validation' or 'test', source and target."
        prediction_set, _ = self._run_inference(split)

        return prediction_set

    def evaluate(self, split: str) -> dict[str, float]:
        prediction_set: PredictionSet = self.collect_predictions(split)

        return {"source_accuracy": prediction_set.source_accuracy, "target_accuracy": prediction_set.target_accuracy}

    def _validate(self) -> dict[str, float]:
        prediction_set, mean_source_cross_entropy = self._run_inference("validation")

        return {
            "source_validation_accuracy": prediction_set.source_accuracy,
            "target_validation_accuracy": prediction_set.target_accuracy,
            "validation_cross_entropy_loss": mean_source_cross_entropy,
        }

    def train(self) -> list[dict[str, float]]:
        "Run until maximum_epochs or early stopping. Returns the per-epoch history."
        self.logger.debug(
            f"Training from epoch {self.completed_epochs + 1} to at most {self.config.maximum_epochs} on {self.device}, "
            f"domain_adaptation={self.config.enable_domain_adaptation}"
        )
        self.config.to_json(os.path.join(self.config.output_directory, "config.json"))

        while self.completed_epochs < self.config.maximum_epochs and not self.early_stopping.should_stop:
            epoch_start_time: float = time.time()
            epoch_record: dict[str, float] = {
                "epoch": self.completed_epochs + 1,
                "learning_rate": self.optimizer.param_groups[0]["lr"],
                "entropy_boundary": self.tuner.entropy_boundary,
                "confidence_margin": self.tuner.confidence_margin,
            }
            epoch_record.update(self._train_epoch())

            if self.config.enable_domain_adaptation:
                self.tuner.update(epoch_record["total_loss"])

            epoch_record.update(self._validate())
            self.scheduler.step()
            self.completed_epochs += 1

            if self.early_stopping.update(epoch_record[self.config.early_stopping_metric], self.completed_epochs):
                torch.save(self.model.state_dict(), self.best_model_path)
                self.logger.debug(f"New best model saved to {self.best_model_path}")

            self.history.append(epoch_record)
            self._write_history()
            self.logger.info(
                f"Epoch {self.completed_epochs}/{self.config.maximum_epochs} "
                f"CE={epoch_record['cross_entropy_loss']:.4f} AC={epoch_record['adaptive_clustering_loss']:.4f} "
                f"ES={epoch_record['entropy_separation_loss']:.4f} total={epoch_record['total_loss']:.4f} "
                f"lr={epoch_record['learning_rate']:.2e} rho={epoch_record['entropy_boundary']:.3f} "
                f"m={epoch_record['confidence_margin']:.3f} "
                f"source_acc={epoch_record['source_validation_accuracy']:.4f} "
                f"target_acc={epoch_record['target_validation_accuracy']:.4f} "
                f"({time.time() - epoch_start_time:.1f}s)"
            )

            if self.completed_epochs % self.config.checkpoint_every_epochs == 0:
                self._save()

        if self.early_stopping.should_stop:
            self.logger.info(
                f"Early stopping after epoch {self.completed_epochs}: no {self.config.early_stopping_metric} "
                f"improvement for {self.early_stopping.epochs_without_improvement} epochs, "
                f"best {self.early_stopping.best_value:.4f} at epoch {self.early_stopping.best_epoch}"
            )

        self._save()

        return self.history

    def load_best_model(self) -> None:
        "Replace the current weights with those saved at the best epoch (best_model.pt)."
        if not os.path.exists(self.best_model_path):
            raise FileNotFoundError(f"No best model at {self.best_model_path}, train first")

        self.model.load_state_dict(torch.load(self.best_model_path, map_location=self.device, weights_only=True))
        self.logger.debug(f"Loaded best model weights from {self.best_model_path} (epoch {self.early_stopping.best_epoch})")

    def _write_history(self) -> None:
        with open(self.history_path, "w") as history_file:
            json.dump(self.history, history_file, indent=2)

    def _save(self) -> None:
        save_checkpoint(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "gradient_scaler": self.gradient_scaler.state_dict(),
                "tuner": self.tuner.state_dict(),
                "bank": self.bank.state_dict(),
                "epoch": self.completed_epochs,
                "early_stopping": self.early_stopping.state_dict(),
                "history": self.history,
                "random_states": capture_random_states(),
                "config": self.config.to_dictionary(),
            },
            self.checkpoint_path,
        )

    def _restore(self, checkpoint_path: str) -> None:
        checkpoint: dict[str, Any] = load_checkpoint(checkpoint_path)

        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.gradient_scaler.load_state_dict(checkpoint["gradient_scaler"])
        self.tuner.load_state_dict(checkpoint["tuner"])
        self.bank.load_state_dict(checkpoint["bank"])
        self.early_stopping.load_state_dict(checkpoint["early_stopping"])
        self.completed_epochs = int(checkpoint["epoch"])
        self.history = list(checkpoint["history"])
        restore_random_states(checkpoint["random_states"])

        resumed_best_model_path: str = os.path.join(os.path.dirname(checkpoint_path), "best_model.pt")
        if os.path.exists(resumed_best_model_path) and not os.path.exists(self.best_model_path):
            shutil.copy(resumed_best_model_path, self.best_model_path)
            self.logger.debug(f"Copied best model from {resumed_best_model_path}")

        self.logger.info(f"Resumed from {checkpoint_path} at epoch {self.completed_epochs}")
