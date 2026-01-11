import numpy as np
import pandas as pd
import os
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.config.settings import PROCESSED_DATA_DIR

class DataPreprocessor:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.X = None
        self.y = None

    def load_data(self):
        df = pd.read_csv(self.dataset_path)
        self.X = df.drop(columns=['target']).values
        self.y = df['target'].values

    def normalize_data(self):
        self.X = (self.X - np.mean(self.X, axis=0)) / np.std(self.X, axis=0)

    def split_data(self, train_ratio=0.7, val_ratio=0.15):
        np.random.seed(42)
        indices = np.random.permutation(len(self.X))
        self.X, self.y = self.X[indices], self.y[indices]

        train_size = int(train_ratio * len(self.X))
        val_size = int(val_ratio * len(self.X))

        X_train, y_train = self.X[:train_size], self.y[:train_size]
        X_val, y_val = self.X[train_size:train_size + val_size], self.y[train_size:train_size + val_size]
        X_test, y_test = self.X[train_size + val_size:], self.y[train_size + val_size:]

        return X_train, y_train, X_val, y_val, X_test, y_test

    def save_data(self, X_train, y_train, X_val, y_val, X_test, y_test):
        os.makedirs(str(PROCESSED_DATA_DIR), exist_ok=True)
        np.save(str(PROCESSED_DATA_DIR / "X_train.npy"), X_train)
        np.save(str(PROCESSED_DATA_DIR / "y_train.npy"), y_train)
        np.save(str(PROCESSED_DATA_DIR / "X_val.npy"), X_val)
        np.save(str(PROCESSED_DATA_DIR / "y_val.npy"), y_val)
        np.save(str(PROCESSED_DATA_DIR / "X_test.npy"), X_test)
        np.save(str(PROCESSED_DATA_DIR / "y_test.npy"), y_test)

        #print("✅ Data preprocessing complete. Saved in 'processed_data' folder.")
