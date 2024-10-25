# Hunt's algorithm simplified in Python
class HuntDecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        num_samples, num_features = X.shape
        unique_classes = set(y)

        # Stopping condition: if all labels are the same or we hit max depth
        if len(unique_classes) == 1 or (self.max_depth and depth == self.max_depth):
            return self._create_leaf(y)

        # Select the best feature to split on
        best_feature, best_threshold = self._find_best_split(X, y)

        if best_feature is None:
            return self._create_leaf(y)

        # Split dataset
        left_indices = X[:, best_feature] < best_threshold
        right_indices = ~left_indices

        # Recursively build the tree
        left_subtree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        return {"feature": best_feature, "threshold": best_threshold, "left": left_subtree, "right": right_subtree}

    def _find_best_split(self, X, y):
        num_samples, num_features = X.shape
        best_feature, best_threshold = None, None
        best_gini = 1.0  # Initial worst case Gini

        for feature in range(num_features):
            thresholds = set(X[:, feature])
            for threshold in thresholds:
                gini = self._calculate_gini(X, y, feature, threshold)
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _calculate_gini(self, X, y, feature, threshold):
        left_indices = X[:, feature] < threshold
        right_indices = ~left_indices

        left_gini = self._gini(y[left_indices])
        right_gini = self._gini(y[right_indices])

        # Weighted average of left and right splits
        total_gini = (sum(left_indices) * left_gini + sum(right_indices) * right_gini) / len(y)
        return total_gini

    def _gini(self, y):
        # Calculate Gini Impurity
        unique_classes, counts = np.unique(y, return_counts=True)
        probs = counts / counts.sum()
        return 1 - sum(probs ** 2)

    def _create_leaf(self, y):
        # Create a leaf node by selecting the most common class
        unique_classes, counts = np.unique(y, return_counts=True)
        return unique_classes[np.argmax(counts)]

    def predict(self, X):
        return [self._predict_one(x, self.tree) for x in X]

    def _predict_one(self, x, tree):
        if isinstance(tree, dict):
            feature, threshold = tree["feature"], tree["threshold"]
            if x[feature] < threshold:
                return self._predict_one(x, tree["left"])
            else:
                return self._predict_one(x, tree["right"])
        else:
            return tree


# Example usage:
import numpy as np
from sklearn.datasets import load_iris

# Load Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Initialize and train Hunt's decision tree
tree = HuntDecisionTree(max_depth=3)
tree.fit(X, y)

# Make predictions
predictions = tree.predict(X)
print(predictions)
