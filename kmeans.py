# kmeans.py
import numpy as np
import random

def initialize_centroids(data, k, method="random"):
    if method == "random":
        return data[random.sample(range(len(data)), k)]
    elif method == "farthest_first":
        centroids = [data[random.randint(0, len(data) - 1)]]
        for _ in range(1, k):
            dist = np.array([min([np.linalg.norm(x - c) for c in centroids]) for x in data])
            next_centroid = data[np.argmax(dist)]
            centroids.append(next_centroid)
        return np.array(centroids)
    elif method == "kmeans++":
        centroids = [data[random.randint(0, len(data) - 1)]]
        for _ in range(1, k):
            dist_sq = np.array([min([np.linalg.norm(x - c)**2 for c in centroids]) for x in data])
            probs = dist_sq / dist_sq.sum()
            cumulative_probs = probs.cumsum()
            r = random.random()
            for i, p in enumerate(cumulative_probs):
                if r < p:
                    centroids.append(data[i])
                    break
        return np.array(centroids)
    else:
        raise ValueError("Unknown initialization method")

def assign_clusters(data, centroids):
    return np.argmin(np.linalg.norm(data[:, np.newaxis] - centroids, axis=2), axis=1)

def update_centroids(data, clusters, k):
    new_centroids = np.array([data[clusters == i].mean(axis=0) if len(data[clusters == i]) > 0 else data[random.randint(0, len(data) - 1)] for i in range(k)])
    return new_centroids

def kmeans(data, k, method="random", initial_centroids=None, max_iters=100):
    if initial_centroids is not None:
        centroids = initial_centroids
    else:
        centroids = initialize_centroids(data, k, method)
    
    for _ in range(max_iters):
        clusters = assign_clusters(data, centroids)
        new_centroids = update_centroids(data, clusters, k)
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    return centroids, clusters
