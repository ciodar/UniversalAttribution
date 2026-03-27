import numpy as np
import torch
import torch.nn as nn

import faiss


_CPU_DEVICE = torch.device("cpu")


class FaissKNNModule(nn.Module):
    """K-Nearest Neighbors classifier using Faiss IndexFlatIP (inner product).

    When used with L2-normalized features, inner product is equivalent to
    cosine similarity. The module interface mirrors LogRegModule: fit() to
    train and forward() to predict, returning probability distributions
    compatible with the evaluation pipeline (compute_oscr, evaluate_multiclass).
    """

    def __init__(self, k=1, num_classes=None):
        super().__init__()
        self.k = k
        self.num_classes = num_classes
        self.index = None
        self.labels = None

    def fit(self, train_features, train_labels):
        """Build the Faiss index from training features and labels.

        Args:
            train_features: Tensor of shape (n_samples, n_features), should be L2-normalized.
            train_labels: Tensor of shape (n_samples,) with integer class labels.
        """
        features_np = train_features.numpy().astype(np.float32)
        labels_np = train_labels.numpy().astype(np.int64)

        if self.num_classes is None:
            self.num_classes = int(labels_np.max()) + 1

        self.labels = labels_np
        self.index = faiss.IndexFlatIP(features_np.shape[1])
        self.index.add(features_np)

    def forward(self, samples, targets):
        """Predict class probability distributions for input samples.

        Returns a dict with:
            - "preds": Tensor (n_samples, num_classes) of class vote proportions
            - "target": the input targets, passed through unchanged

        The probability for each class is the fraction of k neighbors
        belonging to that class (weighted by similarity scores).
        """
        samples_device = samples.device
        samples_np = samples.cpu().numpy().astype(np.float32)

        distances, indices = self.index.search(samples_np, self.k)
        neighbor_labels = self.labels[indices]

        # Build probability distribution from neighbor votes
        n_samples = samples_np.shape[0]
        probas = np.zeros((n_samples, self.num_classes), dtype=np.float64)
        for i in range(n_samples):
            for j in range(self.k):
                label = neighbor_labels[i, j]
                probas[i, label] += distances[i, j]

        # Normalize to get proper probability distributions
        row_sums = probas.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        probas = probas / row_sums

        return {
            "preds": torch.from_numpy(probas).to(samples_device),
            "target": targets,
        }
