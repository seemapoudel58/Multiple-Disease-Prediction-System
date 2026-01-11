import numpy as np

class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        self.tree = self.build_tree(X, y)

    def predict(self, X):
        return np.array([self.predict_sample(x, self.tree) for x in X])

    def gini_impurity(self, y):
        classes, counts = np.unique(y, return_counts=True)
        p = counts / counts.sum()
        return 1 - np.sum(p**2)

    def split(self, X_column, threshold):
        left_mask = X_column <= threshold
        right_mask = ~left_mask
        return left_mask, right_mask

    def information_gain(self, y, left_mask, right_mask):
        n = len(y)
        n_left, n_right = left_mask.sum(), right_mask.sum()
        if n_left == 0 or n_right == 0:
            return 0
        gini_parent = self.gini_impurity(y)
        gini_left = self.gini_impurity(y[left_mask])
        gini_right = self.gini_impurity(y[right_mask])
        weighted_gini = (n_left / n) * gini_left + (n_right / n) * gini_right
        return gini_parent - weighted_gini

    def best_split(self, X, y):
        best_gain = 0
        best_split = None
        for feature_index in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                left_mask, right_mask = self.split(X[:, feature_index], threshold)
                gain = self.information_gain(y, left_mask, right_mask)
                if gain > best_gain:
                    best_gain = gain
                    best_split = {
                        "feature_index": feature_index,
                        "threshold": threshold,
                        "left_mask": left_mask,
                        "right_mask": right_mask,
                    }
        return best_split, best_gain

    def build_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        if num_samples <= 1 or len(np.unique(y)) == 1 or (self.max_depth is not None and depth >= self.max_depth):
            leaf_value = self.most_common_label(y)
            return {"type": "leaf", "value": leaf_value}

        split, gain = self.best_split(X, y)
        if gain == 0:
            leaf_value = self.most_common_label(y)
            return {"type": "leaf", "value": leaf_value}

        left_tree = self.build_tree(X[split["left_mask"]], y[split["left_mask"]], depth + 1)
        right_tree = self.build_tree(X[split["right_mask"]], y[split["right_mask"]], depth + 1)
        return {
            "type": "node",
            "feature_index": split["feature_index"],
            "threshold": split["threshold"],
            "left": left_tree,
            "right": right_tree,
        }

    def most_common_label(self, y):
        classes, counts = np.unique(y, return_counts=True)
        return classes[np.argmax(counts)]

    def predict_sample(self, x, tree):
        if tree["type"] == "leaf":
            return tree["value"]
        feature_value = x[tree["feature_index"]]
        if feature_value <= tree["threshold"]:
            return self.predict_sample(x, tree["left"])
        else:
            return self.predict_sample(x, tree["right"])
