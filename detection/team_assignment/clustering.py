"""
Team Clustering Module

Clusters player embeddings using KMeans (runs ONCE for entire video).
Matches project-RG clustering approach.
"""

import numpy as np
from collections import Counter
from typing import Tuple, Dict, List
import os
import contextlib

# Workaround for threadpoolctl bug on Windows (Anaconda) where a BLAS DLL's
# get_config() returns None, causing .split() to crash inside KMeans.
# We replace threadpool_limits with a no-op so sklearn skips the broken path entirely.
try:
    import threadpoolctl as _tpctl

    @contextlib.contextmanager
    def _noop_threadpool_limits(*args, **kwargs):
        yield

    _tpctl.threadpool_limits = _noop_threadpool_limits

    # Also patch sklearn's own reference if it was already imported
    try:
        import sklearn.utils.fixes as _skfix
        _skfix.threadpool_limits = _noop_threadpool_limits
    except Exception:
        pass
except Exception:
    pass

from sklearn.cluster import KMeans


class TeamClusterer:
    """
    Cluster player embeddings to identify teams.
    Runs KMeans ONCE on all collected embeddings.
    """
    
    def __init__(self, n_clusters: int = 3, random_state: int = 0):
        """
        Initialize clusterer.
        
        Args:
            n_clusters: Number of clusters (default: 3 for Team A, Team B, Others)
            random_state: Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = None
        self.cluster_labels = None
        self.cluster_counter = None
    
    def fit(self, embeddings: np.ndarray) -> 'TeamClusterer':
        """
        Fit KMeans on all collected embeddings.
        
        Args:
            embeddings: All embeddings [N_total_players, embedding_dim]
        
        Returns:
            self
        """
        if embeddings.shape[0] == 0:
            raise ValueError("Cannot fit with zero embeddings")
        
        print(f"[TeamClusterer] Fitting KMeans on {embeddings.shape[0]} players...")
        
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10,
        )
        
        self.kmeans.fit(embeddings)
        
        # Store cluster labels for training data
        self.cluster_labels = self.kmeans.labels_
        
        # Count clusters (matching main_onnx.py line 400)
        self.cluster_counter = Counter(self.cluster_labels)
        
        print(f"[TeamClusterer] Cluster distribution: {self.cluster_counter.most_common()}")
        
        return self
    
    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Predict cluster assignments for new embeddings.
        
        Args:
            embeddings: Embeddings [N, embedding_dim]
        
        Returns:
            cluster_ids: [N] cluster assignments
        """
        if self.kmeans is None:
            raise ValueError("Must call fit() before predict()")
        
        if embeddings.shape[0] == 0:
            return np.array([], dtype=np.int32)
        
        return self.kmeans.predict(embeddings)

    def predict_with_outliers(
        self,
        embeddings: np.ndarray,
        sigma: float = 2.0,
    ) -> np.ndarray:
        """
        Predict cluster assignments and mark distant embeddings as outliers (-1).

        A player is marked -1 when its distance to its assigned cluster center
        exceeds  mean_dist + sigma * std_dist  computed per-cluster on the
        same batch.  This lets n_clusters=2 still produce -1 / 0 / 1 output.

        Args:
            embeddings: [N, D] float32
            sigma:      Standard-deviation multiplier for the per-cluster
                        outlier threshold.  Higher = fewer outliers.
                        Pass sigma < 0 to disable (identical to predict()).

        Returns:
            cluster_ids: [N] int32 — cluster label, or -1 for outliers
        """
        if self.kmeans is None:
            raise ValueError("Must call fit() before predict_with_outliers()")

        if embeddings.shape[0] == 0:
            return np.array([], dtype=np.int32)

        if sigma < 0:
            return self.kmeans.predict(embeddings).astype(np.int32)

        centers = self.kmeans.cluster_centers_           # [K, D]
        raw_ids = self.kmeans.predict(embeddings)        # [N]

        # Distance from each embedding to its own cluster center
        assigned_centers = centers[raw_ids]              # [N, D]
        dists = np.linalg.norm(embeddings - assigned_centers, axis=1)  # [N]

        result = raw_ids.copy().astype(np.int32)

        for cluster_id in range(self.n_clusters):
            mask = raw_ids == cluster_id
            if not np.any(mask):
                continue
            d = dists[mask]
            threshold = d.mean() + sigma * d.std()
            outlier_mask = mask & (dists > threshold)
            result[outlier_mask] = -1
            n_out = int(outlier_mask.sum())
            if n_out:
                print(
                    f"[TeamClusterer] Cluster {cluster_id}: "
                    f"{n_out} outliers (dist > {threshold:.4f}, sigma={sigma})"
                )

        return result
    
    def get_cluster_centers(self) -> np.ndarray:
        """
        Get cluster centers.
        
        Returns:
            centers: [n_clusters, embedding_dim]
        """
        if self.kmeans is None:
            raise ValueError("Must call fit() first")
        
        return self.kmeans.cluster_centers_
    
    def get_cluster_sizes(self) -> Dict[int, int]:
        """
        Get cluster sizes (number of players in each cluster).
        
        Returns:
            Dict mapping cluster_id → count
        """
        if self.cluster_counter is None:
            raise ValueError("Must call fit() first")
        
        return dict(self.cluster_counter)

