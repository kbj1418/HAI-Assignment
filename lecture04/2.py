import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

# 데이터 준비
iris = load_iris()
X = iris.data[:130, [2, 3]]  # 꽃잎 길이, 너비
y = iris.target[:130]

# --------------------------------
# 1. 유클리드 거리 함수
# --------------------------------
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

# --------------------------------
# 2. KNN 예측 함수
# --------------------------------
def knn_predict(x, X_train, y_train, k):
    distances = [euclidean_distance(x, xi) for xi in X_train]
    k_indices = np.argsort(distances)[:k]
    k_labels = y_train[k_indices]

    # 다수결
    labels, counts = np.unique(k_labels, return_counts=True)
    return labels[np.argmax(counts)]

# --------------------------------
# 5. 훈련/검증 데이터 분할
# --------------------------------
np.random.seed(45)
indices = np.random.permutation(len(X))
split = int(0.7 * len(X))

train_idx = indices[:split]
val_idx = indices[split:]

X_train, y_train = X[train_idx], y[train_idx]
X_val, y_val = X[val_idx], y[val_idx]

# --------------------------------
# 6. k값 최적화
# --------------------------------
k_candidates = range(1, 11)
best_k = None
best_acc = 0.0

print("k값별 정확도:")
for k in k_candidates:
    acc = compute_accuracy(X_val, y_val, X_train, y_train, k)
    print(f"k = {k} -> 정확도: {acc:.2f}")
    
    if acc > best_acc:
        best_acc = acc
        best_k = k

print(f"\n✅ 최적의 k값은: {best_k} (정확도: {best_acc:.2f})")
