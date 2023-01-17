import numpy as np


# Probabilities of 3 samples
softmax_outputs = np.array([
    [0.7, 0.2, 0.1],
    [0.5, 0.1, 0.4],
    [0.02, 0.9, 0.08]
    ]).T

# Target (ground-truth) labels for 3 samples
class_targets = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 1, 0]
    ]).T

# Calculate values along second axis (axis of index 1)
predictions = np.argmax(softmax_outputs, axis=0)
true = np.argmax(class_targets, axis= 0)
# If targets are one-hot encoded - convert them

print(predictions)
print(true)
print(np.mean(predictions == true))

