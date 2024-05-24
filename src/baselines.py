import numpy as np
from sklearn.cluster import KMeans
import random

def kmeans_sampling(embeddings, n_clusters=10, n_samples=100):
    embedding_array = np.array([embedding.cpu() for embedding in embeddings.values()])
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(embedding_array)
    cluster_indices = {i: [] for i in range(n_clusters)}
    for idx, label in zip(embeddings.keys(), kmeans.labels_):
        cluster_indices[label].append(idx)

    samples_per_cluster = n_samples // n_clusters
    sampled_indices = []
    for indices in cluster_indices.values():
        if len(indices) >= samples_per_cluster:
            sampled_indices.extend(random.sample(indices, samples_per_cluster))
        else:
            sampled_indices.extend(indices)
    
    sampled_embeddings = {idx: embeddings[idx] for idx in sampled_indices}
    return sampled_embeddings, sampled_indices

def random_sampling(embeddings, n_samples=100):
    sampled_indices = random.sample(list(embeddings.keys()), n_samples)
    sampled_embeddings = {idx: embeddings[idx] for idx in sampled_indices}
    return sampled_embeddings, sampled_indices