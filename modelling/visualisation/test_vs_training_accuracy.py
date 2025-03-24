"""
Script to look at model outputs, plotting testing accuracy against training accuracy
"""

import numpy as np
import matplotlib.pyplot as plt

labels = [
    "n=100, d=20",
    "n=100, d=15",
    "n=100, d=10",
    "n=100, d=None",
    "n=200, d=15",
    "n=500, d=15",
]

# 5 fold cross validation
# Formatted as (ACC, ERR)
testing_accuracies = [
    (0.98035, 0.0006),
    (0.97216, 0.0005),
    (0.93980, 0.0008),
    (0.98253, 0.0003),
    (0.97232, 0.0006),
    (0.97224, 0.0007),
]

training_accuracies = [
    (0.99756, 0.0001),
    (0.98561, 0.0002),
    (0.94697, 0.0007),
    (0.99999, 0.0000),
    (0.98584, 0.0002),
    (0.98596, 0.0001)
]


fig, ax = plt.subplots()

for test_accuracy, training_accuracy, label in zip(testing_accuracies, training_accuracies, labels):

    ax.errorbar(training_accuracy[0], test_accuracy[0], xerr=training_accuracy[1], yerr=test_accuracy[1], marker=".", capsize=3, lw=1, label=label)


x_range = np.linspace(0.935, 1)

ax.plot(x_range, x_range, color="grey", ls="dashed", label="Training Accuracy == Testing Accuracy")

ax.legend()
ax.set_xlabel("Training Accuracy")
ax.set_ylabel("Testing Accuracy")

ax.set_xlim(0.935, 1.0005)
ax.set_ylim(0.935, 1)

ax.set_title("Training-Testing Accuracy Comparison for Multiple Models")

plt.show()
