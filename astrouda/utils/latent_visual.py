# Uses a T-SNE Plot to show the data in the latent space
# Uses isomap decomposition

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


class LatentVisualizer:
    def __init__(self, config, feature_model, data_loader): 
        self.results_dir = config.get("results_dir", "./results")
        self.feature_model = feature_model
        self.data_loader = data_loader


    def get_latent(self):
        self.feature_model.eval()
        latent_representations_source = []
        latent_representations_target = []
        labels = []
        for source, label, target in self.data_loader:
            with torch.no_grad():
                latent_representations_source.append(self.feature_model(source).cpu().numpy())
                latent_representations_target.append(self.feature_model(target).cpu().numpy())
                labels.append(label.cpu().numpy())

        return np.concatenate(labels), np.concatenate(latent_representations_source), np.concatenate(latent_representations_target)
    
    def perform_tsne(self, latent_source, latent_target):
        tsne = TSNE(n_components=2, random_state=42)
        latent_combined = np.concatenate([latent_source, latent_target], axis=0)
        tsne_result = tsne.fit_transform(latent_combined)
        return tsne_result[:len(latent_source)], tsne_result[len(latent_source):]

    def plot_latent(self, labels, latent_source, latent_target):
        # Use the labels to color the points, and use different markers for source and target
        plt.figure(figsize=(10, 6))
        for label in np.unique(labels):
            label_mask = labels == label
            plt.scatter(latent_source[label_mask, 0], latent_source[label_mask, 1], label=f"Source Class {label}", alpha=0.5)
            plt.scatter(latent_target[label_mask, 0], latent_target[label_mask, 1], label=f"Target Class {label}", alpha=0.5, marker='x')
        plt.title("T-SNE of Latent Space")
        plt.xlabel("T-SNE Dimension 1")
        plt.ylabel("T-SNE Dimension 2")
        plt.legend()
        plt.savefig(f"{self.results_dir}/latent_space_tsne.png")
        plt.close()


    def __call__(self,):
        labels, latent_source, latent_target = self.get_latent()
        latent_source_tsne, latent_target_tsne = self.perform_tsne(latent_source, latent_target)
        self.plot_latent(labels, latent_source_tsne, latent_target_tsne)