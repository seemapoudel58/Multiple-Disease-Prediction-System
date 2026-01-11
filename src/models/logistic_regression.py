import numpy as np
import os
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config.settings import SAVED_MODELS_DIR

class LogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=2000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        
        self.weights = 0.0
        self.bias = 0.0

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def compute_loss(self, y, y_pred):
        return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for epoch in range(self.epochs):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(linear_model)

            # Compute gradients
            dw = np.dot(X.T, (y_pred - y)) / n_samples
            db = np.mean(y_pred - y)

           
            # Update weights
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            if epoch % 100 == 0:
                loss = self.compute_loss(y, y_pred)
                #print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(linear_model)
        return (y_pred >= 0.5).astype(int)

    def save_model(self, folder=None):
        if folder is None:
            folder = SAVED_MODELS_DIR
        os.makedirs(str(folder), exist_ok=True)
        np.save(str(Path(folder) / "weights.npy"), self.weights)
        np.save(str(Path(folder) / "bias.npy"), self.bias)
        #print("✅ Model trained and saved.")
