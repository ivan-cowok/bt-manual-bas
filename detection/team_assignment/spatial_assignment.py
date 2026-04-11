"""
Spatial Team Assignment Module

Assigns teams based on spatial position (left=Team A, right=Team B).
Key insight: Calculate center x-coordinate for each cluster, then assign based on position.
"""

import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict


class SpatialTeamAssigner:
    """
    Assign teams based on spatial clustering.
    
    Algorithm:
    1. For each cluster, calculate mean x-coordinate of all player bboxes
    2. Sort clusters by x-coordinate (left to right)
    3. Assign: leftmost = Team A (0), rightmost = Team B (1), middle = Other (-1)
    """
    
    def __init__(self, exclude_smallest_cluster: bool = True):
        """
        Initialize spatial assigner.
        
        Args:
            exclude_smallest_cluster: Treat smallest cluster as "other" (referees)
        """
        self.exclude_smallest_cluster = exclude_smallest_cluster
        self.cluster_to_team = None
        self.cluster_spatial_centers = None
    
    def fit(
        self,
        cluster_ids: np.ndarray,
        bboxes: List[Tuple[int, int, int, int]],
        cluster_sizes: Dict[int, int]
    ) -> 'SpatialTeamAssigner':
        """
        Determine team assignments based on spatial positions.
        
        Args:
            cluster_ids: Cluster ID for each player [N_players]
            bboxes: Bounding boxes for each player [[x1,y1,x2,y2], ...]
            cluster_sizes: Size of each cluster {cluster_id: count}
        
        Returns:
            self
        """
        if len(cluster_ids) != len(bboxes):
            raise ValueError("cluster_ids and bboxes must have same length")
        
        print(f"[SpatialAssigner] Analyzing {len(cluster_ids)} players...")
        
        # Calculate center x-coordinate for each cluster
        cluster_x_coords = defaultdict(list)
        
        for cluster_id, bbox in zip(cluster_ids, bboxes):
            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) / 2
            cluster_x_coords[int(cluster_id)].append(center_x)
        
        # Calculate mean x-coordinate for each cluster
        self.cluster_spatial_centers = {}
        for cluster_id, x_coords in cluster_x_coords.items():
            self.cluster_spatial_centers[cluster_id] = np.mean(x_coords)
        
        print(f"[SpatialAssigner] Cluster spatial centers (x-coord):")
        for cid, center_x in sorted(self.cluster_spatial_centers.items()):
            print(f"  Cluster {cid}: x={center_x:.1f} (n={cluster_sizes[cid]})")
        
        # Identify smallest cluster (likely referees/others).
        # Only meaningful when there are >2 clusters; with exactly 2 clusters
        # both must be team clusters — excluding one would leave only one team.
        smallest_cluster = None
        if self.exclude_smallest_cluster and len(cluster_sizes) > 2:
            smallest_cluster = min(cluster_sizes.items(), key=lambda x: x[1])[0]
            print(f"[SpatialAssigner] Smallest cluster: {smallest_cluster} → 'other' (-1)")
        
        # Sort clusters by x-coordinate (left to right)
        sorted_clusters = sorted(
            self.cluster_spatial_centers.items(),
            key=lambda x: x[1]
        )
        
        # Assign teams
        self.cluster_to_team = {}
        
        # Exclude smallest cluster from team assignment
        team_clusters = [
            (cid, x) for cid, x in sorted_clusters 
            if smallest_cluster is None or cid != smallest_cluster
        ]
        
        if len(team_clusters) >= 2:
            # Leftmost = Team A (0), Rightmost = Team B (1)
            self.cluster_to_team[team_clusters[0][0]] = 0  # Team A (left)
            self.cluster_to_team[team_clusters[-1][0]] = 1  # Team B (right)
            
            print(f"[SpatialAssigner] Team assignments:")
            print(f"  Cluster {team_clusters[0][0]} (x={team_clusters[0][1]:.1f}) → Team A (0)")
            print(f"  Cluster {team_clusters[-1][0]} (x={team_clusters[-1][1]:.1f}) → Team B (1)")
            
            # Middle clusters (if any) → other
            for cid, x in team_clusters[1:-1]:
                self.cluster_to_team[cid] = -1
                print(f"  Cluster {cid} (x={x:.1f}) → Other (-1)")
        
        elif len(team_clusters) == 1:
            # Only one team cluster - assign as Team A
            self.cluster_to_team[team_clusters[0][0]] = 0
            print(f"[SpatialAssigner] Only one team cluster → Team A (0)")
        
        # Assign smallest cluster to "other"
        if smallest_cluster is not None:
            self.cluster_to_team[smallest_cluster] = -1
        
        # Handle any unassigned clusters
        for cluster_id in self.cluster_spatial_centers.keys():
            if cluster_id not in self.cluster_to_team:
                self.cluster_to_team[cluster_id] = -1
                print(f"[SpatialAssigner] Cluster {cluster_id} → Other (-1) (unassigned)")
        
        return self
    
    def predict(self, cluster_ids: np.ndarray) -> np.ndarray:
        """
        Get team assignments for cluster IDs.
        
        Args:
            cluster_ids: Cluster IDs [N]
        
        Returns:
            team_ids: [N] team assignments (0=Team A, 1=Team B, -1=Other)
        """
        if self.cluster_to_team is None:
            raise ValueError("Must call fit() first")
        
        team_ids = np.array([
            self.cluster_to_team.get(int(cid), -1)
            for cid in cluster_ids
        ], dtype=np.int32)
        
        return team_ids
    
    def get_cluster_to_team_mapping(self) -> Dict[int, int]:
        """
        Get cluster → team mapping.
        
        Returns:
            Dict: {cluster_id: team_id}
                  team_id: 0=Team A, 1=Team B, -1=Other
        """
        if self.cluster_to_team is None:
            raise ValueError("Must call fit() first")
        
        return self.cluster_to_team.copy()

