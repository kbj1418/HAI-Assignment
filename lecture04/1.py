import numpy as np

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))

def knn_predict(x,X_train, y_train, k):
    distances = [euclidean_distance(x,xi) for xi in X_train]
    k_indices = np.argsort(distances)[:k]
    k_labels = y_train[k_indices]

    labels, counts = np.unique(k_labels, return_counts = True)
    return labels[np.argmax(counts)]

def compute_accuracy(X_val, y_val, x_train, y_train, k):
    correct  = 0
    for x, y in zip(X_val,y_val):
        pred = knn_predict(x, x_train, y_train, k)
        if pred == y:
            correct += 1
    return correct / len(y_val)

x = np.array([
    [1.0, 2.0], [1.5, 1.8], [5.0, 8.0],
    [6.0, 9.0], [1.0, 0.6], [9.0, 11.0]
    [8.0, 2.0], [10.0, 2.0], [9.0, 3.0]
])
y = np.array(['A', 'A', 'B', 'B', 'A', 'B', 'B', 'B', 'B'])

np.random.seed(42)
indices = np.random.permutation(len(x))
split = int(0.7 * len(x))

train_idx = indices[:split]
val_idx = indices[split:]

