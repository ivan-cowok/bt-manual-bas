"""
Team Assignment Module

Two-pass team classification:
1. Extract features from ALL frames
2. Cluster ONCE, then assign teams spatially (left=Team A, right=Team B)

Encoder backends:
  - TeamFeatureEncoder  (original autoencoder, team.onnx)
  - ReIDFeatureEncoder  (OSNet ReID model,     osnet.onnx)

Preprocessor backends:
  - TeamCropPreprocessor  (64×64 center-crop, NHWC)
  - ReIDCropPreprocessor  (256×128 full-crop, NCHW, ImageNet norm)
"""

from .preprocessing import TeamCropPreprocessor, ReIDCropPreprocessor
from .encoder import TeamFeatureEncoder, ReIDFeatureEncoder
from .clustering import TeamClusterer
from .spatial_assignment import SpatialTeamAssigner

__all__ = [
    'TeamCropPreprocessor',
    'ReIDCropPreprocessor',
    'TeamFeatureEncoder',
    'ReIDFeatureEncoder',
    'TeamClusterer',
    'SpatialTeamAssigner',
]

