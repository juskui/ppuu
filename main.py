# Importing the required libraries
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

# Loading a dataset (Iris dataset)
iris = load_iris()
X, y = iris.data, iris.target

# Initializing the DecisionTreeClassifier
clf = DecisionTreeClassifier()

# Fitting the model
clf.fit(X, y)

# Plotting the decision tree
plt.figure(figsize=(15, 12))
plot_tree(
    clf, feature_names=iris.feature_names, class_names=iris.target_names, filled=True
)
plt.title("Decision Tree Example - Iris Dataset")
plt.show()
