import numpy as np
import pandas as pd
import time
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize

X = pd.read_csv("data.csv").values
y = pd.read_csv("label.csv").values.flatten()

K = len(np.unique(y))

#Distance Functions
def euclidean_distance(a, b):
    return np.linalg.norm(a - b)

def cosine_distance(a, b):
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)

def jaccard_distance(a, b):
    intersection = np.sum(np.minimum(a, b))
    union = np.sum(np.maximum(a, b)) + 1e-10
    return 1 - intersection / union


def kmeans(X, K, distance_func, max_iter=500):
    n_samples, n_features = X.shape

    np.random.seed(42)
    centroids = X[np.random.choice(n_samples, K, replace=False)]

    prev_sse = float("inf")

    for iteration in range(max_iter):
        clusters = [[] for _ in range(K)]

        for i in range(n_samples):
            distances = [distance_func(X[i], c) for c in centroids]
            cluster_idx = np.argmin(distances)
            clusters[cluster_idx].append(i)

        new_centroids = np.zeros((K, n_features))
        for idx, cluster in enumerate(clusters):
            if len(cluster) > 0:
                new_centroids[idx] = np.mean(X[cluster], axis=0)
            else:
                new_centroids[idx] = centroids[idx]

        sse = 0 #Computing SSE
        for idx, cluster in enumerate(clusters):
            for i in cluster:
                sse += euclidean_distance(X[i], new_centroids[idx]) ** 2

        if np.allclose(centroids, new_centroids): #Stopping conditions
            return clusters, new_centroids, sse, iteration+1

        if sse > prev_sse:
            return clusters, new_centroids, sse, iteration+1

        prev_sse = sse
        centroids = new_centroids

    return clusters, centroids, sse, max_iter


def get_cluster_labels(clusters, y): #Majority Voting for Labels
    labels = np.zeros(len(y))
    for cluster_idx, cluster in enumerate(clusters):
        if len(cluster) == 0:
            continue
        majority_label = np.bincount(y[cluster]).argmax()
        for i in cluster:
            labels[i] = majority_label
    return labels


def run_experiment(X, y, name, distance_func):
    print(f"\n===== {name} =====")

    start = time.time()

    clusters, centroids, sse, iterations = kmeans(
    X, K, distance_func, max_iter=500
    )

    end = time.time()

    pred_labels = get_cluster_labels(clusters, y)
    acc = accuracy_score(y, pred_labels)

    print(f"SSE: {sse}")
    print(f"Accuracy: {acc}")
    print(f"Iterations: {iterations}")
    print(f"Time: {end - start:.2f} sec")

    return sse, acc, iterations, end - start

#Preprocessing for Cosine & Jaccard
X_cosine = normalize(X)

X_jaccard = (X > np.mean(X)).astype(int)


results = {} #Running all 3 methods

results["Euclidean"] = run_experiment(X, y, "Euclidean K-Means", euclidean_distance)

results["Cosine"] = run_experiment(X_cosine, y, "Cosine K-Means", cosine_distance)

results["Jaccard"] = run_experiment(X_jaccard, y, "Jaccard K-Means", jaccard_distance)


print("\n===== FINAL COMPARISON =====") #Comparing the results
for key, val in results.items():
    print(f"{key}: SSE={val[0]}, Accuracy={val[1]}, Iter={val[2]}, Time={val[3]:.2f}")
