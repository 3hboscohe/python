import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

def generate_data(n_samples=300, n_features=2, n_clusters=3, random_state=42):
    data, labels = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_clusters, random_state=random_state)
    return data, labels

def apply_kmeans(data, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(data)
    return kmeans

def plot_clusters(data, kmeans):
    plt.scatter(data[:, 0], data[:, 1], c=kmeans.labels_, cmap='viridis', alpha=0.6, edgecolors='k')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red', marker='X', label='Centroids')
    plt.title("K-Means Clustering")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.show()

def main():
    data, _ = generate_data()
    kmeans = apply_kmeans(data)
    plot_clusters(data, kmeans)

if __name__ == "__main__":
    main()
