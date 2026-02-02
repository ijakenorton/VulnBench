"""
Gradient Boosting Model for Vulnerability Detection

This model uses bag-of-words features with gradient boosting to test
whether datasets contain spurious correlations. Inspired by Risse et al. (2025).
"""

import torch
import torch.nn as nn
import numpy as np
from collections import Counter
from sklearn.ensemble import HistGradientBoostingClassifier
from scipy.sparse import csr_matrix, vstack
import pickle
from pathlib import Path


class GradientBoostingModel(nn.Module):
    """
    Wrapper for sklearn's HistGradientBoostingClassifier to fit the PyTorch model interface.

    This model:
    - Uses bag-of-words features (token counts) instead of embeddings
    - Trains a gradient boosting classifier on these features
    - Tests for spurious correlations in datasets
    """

    def __init__(self, encoder, config, tokenizer, args):
        super(GradientBoostingModel, self).__init__()
        self.encoder = encoder  # Not used, kept for interface compatibility
        self.config = config
        self.tokenizer = tokenizer
        self.args = args

        # Initialize gradient boosting classifier
        # Using hyperparameters from Risse et al. (2025)
        self.clf = HistGradientBoostingClassifier(
            learning_rate=getattr(args, 'gb_learning_rate', 0.3),
            max_depth=getattr(args, 'gb_max_depth', 10),
            max_iter=getattr(args, 'gb_max_iter', 200),
            min_samples_leaf=getattr(args, 'gb_min_samples_leaf', 20),
            random_state=getattr(args, 'seed', 42)
        )

        # Cache for vocabulary size
        self.vocab_size = len(tokenizer)
        self.max_vocab_index = max(tokenizer.get_vocab().values())

        # Training data accumulator
        self.training_features = []
        self.training_labels = []
        self.is_fitted = False

    def _encode_to_bow(self, input_ids, return_sparse=False):
        """
        Convert token IDs to bag-of-words feature vectors.

        Args:
            input_ids: Tensor of shape [batch_size, seq_len] with token IDs
            return_sparse: If True, return scipy sparse matrix instead of dense numpy array

        Returns:
            numpy array or scipy sparse matrix of shape [batch_size, vocab_size] with token counts
        """
        batch_size = input_ids.shape[0]
        features = np.zeros((batch_size, self.max_vocab_index + 1), dtype=np.float32)

        # Convert to numpy for easier processing
        input_ids_np = input_ids.cpu().numpy()

        for i in range(batch_size):
            # Count tokens in this sequence
            token_ids = input_ids_np[i]
            # Filter out padding tokens (assuming padding token is 1)
            token_ids = token_ids[token_ids != 1]

            # Count occurrences
            counts = Counter(token_ids)
            for token_id, count in counts.items():
                if token_id <= self.max_vocab_index:
                    features[i, token_id] = count

        if return_sparse:
            return csr_matrix(features)
        return features

    def accumulate_training_data(self, input_ids, labels):
        """
        Accumulate training data for batch training.

        Since sklearn models need to fit on all data at once (or use partial_fit),
        we accumulate batches and fit at the end.

        Note: Only accumulates data once (from first epoch). Subsequent calls are ignored
        if data is already accumulated, to avoid memory explosion from multiple epochs.

        Uses sparse matrices to save memory for large datasets.
        """
        # Skip accumulation if already fitted or if we're re-seeing data
        if self.is_fitted:
            return

        # Convert to sparse matrix to save memory (bag-of-words is typically 95%+ sparse)
        features_sparse = self._encode_to_bow(input_ids, return_sparse=True)
        labels_np = labels.cpu().numpy()

        self.training_features.append(features_sparse)
        self.training_labels.append(labels_np)

    def fit_accumulated_data(self):
        """
        Fit the gradient boosting model on all accumulated training data.

        Uses sparse matrix operations to efficiently handle large datasets.
        """
        if not self.training_features:
            raise ValueError("No training data accumulated")

        # Concatenate all batches using sparse vstack
        X = vstack(self.training_features)
        y = np.concatenate(self.training_labels)

        print(f"Fitting gradient boosting on {X.shape[0]} samples with {X.shape[1]} features...")
        print(f"Matrix density: {X.nnz / (X.shape[0] * X.shape[1]) * 100:.2f}% (using sparse matrix)")

        # HistGradientBoostingClassifier can handle sparse matrices directly
        self.clf.fit(X, y)
        self.is_fitted = True

        # Clear accumulated data to free memory
        self.training_features = []
        self.training_labels = []

        print("Gradient boosting model fitted successfully!")

    def forward(self, input_ids=None, labels=None):
        """
        Forward pass compatible with PyTorch training loop.

        During training: accumulates data and returns dummy loss
        During inference: returns predictions
        """
        if self.training:
            # Training mode: accumulate data
            if labels is not None:
                self.accumulate_training_data(input_ids, labels)

            # Return dummy loss (actual training happens in fit_accumulated_data)
            # This keeps the training loop happy
            dummy_loss = torch.tensor(0.0, device=input_ids.device, requires_grad=True)
            dummy_probs = torch.zeros((input_ids.shape[0], 1), device=input_ids.device)
            return dummy_loss, dummy_probs
        else:
            # Inference mode: make predictions
            if not self.is_fitted:
                raise ValueError("Model not fitted yet. Call fit_accumulated_data() first.")

            features = self._encode_to_bow(input_ids)

            # Get predictions
            probs = self.clf.predict_proba(features)[:, 1:2]  # Get probability of class 1

            # Convert to torch tensor
            probs_tensor = torch.tensor(probs, dtype=torch.float32, device=input_ids.device)

            if labels is not None:
                # Calculate loss for evaluation
                labels_np = labels.cpu().numpy()
                # Binary cross-entropy loss
                epsilon = 1e-10
                loss = -np.mean(
                    labels_np * np.log(probs[:, 0] + epsilon) +
                    (1 - labels_np) * np.log(1 - probs[:, 0] + epsilon)
                )
                loss_tensor = torch.tensor(loss, dtype=torch.float32, device=input_ids.device)
                return loss_tensor, probs_tensor
            else:
                return probs_tensor

    def save_pretrained(self, save_directory):
        """Save the gradient boosting model."""
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        # Save the sklearn model
        model_path = save_directory / "gradient_boosting.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(self.clf, f)

        # Save metadata
        metadata = {
            'vocab_size': self.vocab_size,
            'max_vocab_index': self.max_vocab_index,
            'is_fitted': self.is_fitted
        }
        metadata_path = save_directory / "metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)

    def load_pretrained(self, load_directory):
        """Load the gradient boosting model."""
        load_directory = Path(load_directory)

        # Load the sklearn model
        model_path = load_directory / "gradient_boosting.pkl"
        with open(model_path, 'rb') as f:
            self.clf = pickle.load(f)

        # Load metadata
        metadata_path = load_directory / "metadata.pkl"
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)

        self.is_fitted = metadata['is_fitted']
