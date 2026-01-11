import numpy as np
import os
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config.settings import SAVED_MODELS_DIR

class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Convert labels from {0, 1} to {-1, 1} for hinge loss
        y_ = np.where(y <= 0, -1, 1)

        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    # No misclassification
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    # Misclassification
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]

    def predict(self, X):
        linear_output = np.dot(X, self.w) - self.b
        return np.sign(linear_output)
    
    def save_model(self, folder=None):
        """Save the SVM model weights and bias"""
        if folder is None:
            folder = SAVED_MODELS_DIR / "breast_cancer"
        os.makedirs(str(folder), exist_ok=True)
        np.save(str(Path(folder) / "svm_weights.npy"), self.w)
        np.save(str(Path(folder) / "svm_bias.npy"), self.b)
        # Save hyperparameters
        np.save(str(Path(folder) / "svm_params.npy"), np.array([self.lr, self.lambda_param, self.n_iters]))
    
    def load_model(self, folder=None):
        """Load the SVM model weights and bias"""
        if folder is None:
            folder = SAVED_MODELS_DIR / "breast_cancer"
        self.w = np.load(str(Path(folder) / "svm_weights.npy"))
        self.b = np.load(str(Path(folder) / "svm_bias.npy"))
        params = np.load(str(Path(folder) / "svm_params.npy"))
        self.lr = params[0]
        self.lambda_param = params[1]
        self.n_iters = int(params[2])

