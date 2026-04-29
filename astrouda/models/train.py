
import json

import torch

from astrouda.models.losses import CrossEntropyLoss, AdaptiveClusteringLoss, EntropySeparationLoss
from astrouda.models.models import AdaptiveModel
from astrouda.data.data import DADataLoader

from astrouda.utils import utils
from astrouda.utils.latent_visual import LatentVisualizer


class TrainDA: 
    def __init__(self, config): 
        self.config = config
        self.logger = utils.make_logger(config)
        self.device = utils.get_device(config)

        self.base_model =  AdaptiveModel(self.config)

        self.feature_model = self.base_model.feature_model
        self.feature_model.eval()

        self.classifier_model = self.base_model.classifier_model

        self.optimizer = self.base_model.optimizer
        self.scheduler = self.base_model.scheduler

        self._lambda = torch.Tensor([config.get('lambda', 0.005)], device=self.device)

        self.bound_p = torch.Tensor([config.get('p_init', 0.5)], device=self.device)
        self.bound_m = torch.Tensor([config.get('m_init', 0.5)], device=self.device)  
        self.n_es_updates = 0
        self.n_es_epochs = 0
        self.n_es_steps = 0

        self.ce_loss_fn = CrossEntropyLoss(config)
        self.ac_loss_fn = AdaptiveClusteringLoss(config)
        self.es_loss_fn = EntropySeparationLoss(config)

        self.loss_history = {"CE_loss": [], "AC_loss": [], "ES_loss": [], "Total_loss": []}
        self.val_history = {"loss": [], "accuracy": []}

    def update_ES_loss_params(self) -> None:
        """Logic from https://arxiv.org/pdf/2302.02005 algorithm 1"""
        step_size = [
            torch.tensor(0.3, device=self.device), 
            torch.tensor(-0.3, device=self.device),
            torch.tensor(0.5, device=self.device), 
            torch.tensor(-0.5, device=self.device)
        ]
        unchanged_message = f"ES loss parameters remain unchanged: p={self.bound_p.data}, m={self.bound_m.data}"
        if self.loss_history["Total_loss"][-1] < self.config.get("es_loss_threshold", 0.1):
            self.n_epochs += 1
            if (self.n_es_updates == 0) and (self.n_es_epochs > 5):
                self.bound_p += step_size[self.n_es_steps]
                self.n_es_epochs = 0
                self.n_es_updates += 1
            elif self.n_es_updates == 1 and (self.n_es_epochs >2):
                self.bound_m += step_size[self.n_es_steps]
                self.n_es_epochs = 0
                self.n_es_updates += 1
            elif self.n_es_updates == 2 and (self.n_es_epochs > 2):
                self.bound_m *= torch.tensor(2, device=self.device)
                self.n_es_epochs = 0
                self.n_es_updates += 1
            elif self.n_es_updates == 3 and (self.n_es_epochs > 2):
                self.n_es_steps += 1
                self.n_es_updates = 0
            else: 
                self.logger.info(unchanged_message)
                return
            
            self.logger.info(f"Updated ES loss parameters: p={self.bound_p.data}, m={self.bound_m.data}")
        else: 
            self.logger.info(unchanged_message)


    def calculate_loss(self, source, source_labels, target) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        source_prediction = self.classifier_model(source).to(float)
        CE_loss = self.ce_loss_fn.forward(source_prediction, source_labels)

        # Non-labeled loss
        target_prediction = self.classifier_model(target)
        ES_loss = self.es_loss_fn.forward(target_prediction, self.bound_p, self.bound_m)

        # Combined loss with target and source domains
        AC_loss = self.ac_loss_fn.forward(
            feature_model=self.feature_model, 
            classifier_model=self.classifier_model, 
            source_images=source, 
            target_images=target, 
            source_labels=source_labels
        )

        return CE_loss, AC_loss, ES_loss


    def train_epoch(self, data_loader):
        self.classifier_model.train()
        batch_loss = []
        ce_batch, ac_batch, es_batch = torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device)
        for source, source_labels, target in data_loader:

            CE_loss, AC_loss, ES_loss = self.calculate_loss(source, source_labels, target)
            
            loss = CE_loss + self._lambda * (AC_loss + ES_loss)

            batch_loss.append(loss)
            ce_batch += CE_loss.item()
            ac_batch += AC_loss.item()
            es_batch += ES_loss.item()

            self.logger.info(
                f"Training \n\
                CE Loss: {CE_loss}\n\
                AC Loss: {AC_loss}\n\
                ES Loss: {ES_loss.item()}\n\
                Total Loss: {loss.item()}"
            )
        
        loss = torch.stack(batch_loss).mean()
        loss.backward()

        n_batches = len(data_loader)
        self.loss_history["CE_loss"].append(ce_batch.item()/n_batches)
        self.loss_history["AC_loss"].append(ac_batch.item()/n_batches)
        self.loss_history["ES_loss"].append(es_batch.item()/n_batches)
        self.loss_history["Total_loss"].append(loss.item())

    def val_epoch(self, data_loader):
        self.classifier_model.eval()
        batch_loss = []
        batch_accuracy = {index: [] for index in range(self.config.get("num_classes", 10))}

        for source, source_labels, target in data_loader:

            # Loss
            CE_loss, AC_loss, ES_loss = self.calculate_loss(source, source_labels, target)
            loss = CE_loss + self._lambda * (AC_loss + ES_loss)
            batch_loss.append(loss)

            self.logger.info(
                f"Validation \n\
                CE Loss: {CE_loss}\n\
                AC Loss: {AC_loss}\n\
                ES Loss: {ES_loss}\n\
                Total Loss: {loss.item()}"
            )

            # Calculate accuracy
            # accuracy = utils.calculate_accuracy(source_labels, self.classifier_model(source))
            # for key, acc in accuracy.items():
            #     batch_accuracy[key].append(acc)
        
        loss = torch.stack(batch_loss).mean()
        accuracy = {
            #index: torch.stack(batch_accuracy).mean() for index, batch_accuracy in batch_accuracy.items()
        }
        
        self.val_history["loss"].append(loss.item())
        self.val_history["accuracy"].append(accuracy)


    def train(self):
        loader = DADataLoader(self.config)
        train_data_loader = loader.get_data_loader(train=True)
        val_data_loader = loader.get_data_loader(train=False)

        for epoch in range(self.config.get("num_epochs", 100)):
            self.logger.info(f"Starting epoch {epoch}")
            self.train_epoch(train_data_loader)
            self.optimizer.step()

            self.val_epoch(val_data_loader)
            
            self.scheduler.step()
            self.update_ES_loss_params()

            self.logger.info(f"Updated learning rate: {self.scheduler.get_last_lr()}")
            self.logger.info(f"Updated lambda value: {self._lambda}")

    def test(self):
        self.config['test'] = True
        loader = DADataLoader(self.config)
        loader = loader.get_data_loader()
        self.classifier_model.eval()
        accuracy = {
            "source": [],
            "target": []
        }
        for source, labels, target in loader:
            source_prediction = self.classifier_model(source)
            target_prediction = self.classifier_model(target)

            acc_source = utils.calculate_accuracy(labels, source_prediction)
            acc_target = utils.calculate_accuracy(None, target_prediction)
            accuracy["source"].append(acc_source)
            accuracy["target"].append(acc_target)
            
        self.logger.info(
                f"Test Loss \n \
                (source): {torch.mean(accuracy['source'])} \n \
                (target): {torch.mean(accuracy['target'])}"
            )
        LatentVisualizer(self.config, self.feature_model, loader)()


    def save(self):
        self.base_model.save_results()

        save_path = f"{self.base_model.output_path}/training_history.json"
        with open(save_path, 'w') as f:
            json.dump({
                "loss_history": self.loss_history,
                "val_history": self.val_history
            }, f)
        self.logger.info(f"Saved training history to {save_path}")

        utils.plot_loss(self.loss_history, self.config)
        self.logger.info(f"Plotted loss history to {self.base_model.output_path}/loss_history.png")
        utils.plot_accuracy(self.val_history, self.config)
        self.logger.info(f"Plotted accuracy history to {self.base_model.output_path}/accuracy_[datasets].png")


    def __call__(self):
        self.train()
        self.save()