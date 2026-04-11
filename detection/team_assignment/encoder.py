"""
Team Feature Encoder Module

Two encoder backends:

TeamFeatureEncoder (original / autoencoder):
  - Model: team.onnx  (convolutional autoencoder)
  - Input: [B, 64, 64, 3] float32 NHWC, values in [0, 1]
  - Output: 256-dim embeddings (global average pooling applied internally)

ReIDFeatureEncoder (OSNet ReID):
  - Model: osnet.onnx  (torchreid OSNet)
  - Input: [B, 3, 256, 128] float32 NCHW, ImageNet-normalised
  - Output: D-dim L2-normalised embeddings (D = model-dependent, typically 512)

Both classes expose the same public interface:
  encode_batch(batch: np.ndarray) -> np.ndarray
  get_embedding_dim() -> int
"""

import onnxruntime as ort
import numpy as np
from typing import List, Optional


class TeamFeatureEncoder:
    """
    Feature encoder for team classification.
    Runs ONNX encoder model to extract embeddings from player crops.
    """
    
    def __init__(
        self,
        model_path: str,
        providers: Optional[List[str]] = None
    ):
        """
        Initialize feature encoder.
        
        Args:
            model_path: Path to ONNX encoder model
            providers: ONNX Runtime providers (default: auto-detect)
        """
        self.model_path = model_path
        
        # Setup providers
        if providers is None:
            # Auto-detect: try GPU first, fallback to CPU
            available = ort.get_available_providers()
            if 'CUDAExecutionProvider' in available:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            else:
                providers = ['CPUExecutionProvider']
        
        self.providers = providers
        
        # Create session
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=providers
        )
        
        # Get input/output info
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_name = self.session.get_outputs()[0].name
        
        print(f"[TeamEncoder] Loaded: {model_path}")
        print(f"[TeamEncoder] Input shape: {self.input_shape}")
        print(f"[TeamEncoder] Providers: {providers}")
    
    def encode_batch(self, crops_normalized: np.ndarray) -> np.ndarray:
        """
        Extract features from normalized crops.
        
        Args:
            crops_normalized: Batch [B, 64, 64, 3] float32 (NHWC format)
        
        Returns:
            embeddings: [B, 256] float32 (after global average pooling)
        """
        if crops_normalized.shape[0] == 0:
            # Empty batch
            return np.array([], dtype=np.float32).reshape(0, 256)
        
        # Run inference
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: crops_normalized}
        )
        
        # Get embeddings: [B, 16, 16, 256]
        embeddings = outputs[0]
        
        # Apply Global Average Pooling: [B, 16, 16, 256] -> [B, 256]
        # Average over spatial dimensions (H=16, W=16)
        if len(embeddings.shape) == 4:
            # Shape is [B, H, W, C] - apply global average pooling
            embeddings = embeddings.mean(axis=(1, 2))  # Average over H and W
            print(f"[TeamEncoder] Applied global average pooling: {embeddings.shape}")
        else:
            # Fallback: flatten if shape is unexpected
            embeddings = embeddings.reshape(embeddings.shape[0], -1)
        
        return embeddings
    
    def get_embedding_dim(self) -> int:
        """Get embedding dimensionality (256 after global average pooling)"""
        # After global average pooling: [B, 16, 16, 256] -> [B, 256]
        return 256


# =============================================================================
# ReID Feature Encoder  (OSNet / torchreid)
# =============================================================================

class ReIDFeatureEncoder:
    """
    Feature encoder for team classification using OSNet ReID model.

    Runs osnet.onnx via ONNX Runtime.
    Input:  [B, 3, 256, 128] float32 NCHW, ImageNet-normalised
    Output: [B, D] float32 L2-normalised embeddings
    """

    def __init__(
        self,
        model_path: str,
        providers: Optional[List[str]] = None,
    ):
        """
        Args:
            model_path: Path to osnet.onnx
            providers:  ONNX Runtime execution providers (auto-detected when None)
        """
        self.model_path = model_path

        if providers is None:
            available = ort.get_available_providers()
            if 'CUDAExecutionProvider' in available:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            else:
                providers = ['CPUExecutionProvider']

        self.providers = providers

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=providers,
        )

        self.input_name  = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_name = self.session.get_outputs()[0].name

        # Infer embedding dimension from model metadata
        # Run a dummy forward pass with batch=1 to get output dim
        self._embedding_dim: Optional[int] = None

        print(f"[ReIDEncoder] Loaded: {model_path}")
        print(f"[ReIDEncoder] Input shape: {self.input_shape}")
        print(f"[ReIDEncoder] Providers: {providers}")

    def encode_batch(self, crops_nchw: np.ndarray) -> np.ndarray:
        """
        Extract L2-normalised ReID features.

        Args:
            crops_nchw: [B, 3, H, W] float32, ImageNet-normalised

        Returns:
            embeddings: [B, D] float32, L2-normalised
        """
        if crops_nchw.shape[0] == 0:
            dim = self._embedding_dim if self._embedding_dim else 512
            return np.empty((0, dim), dtype=np.float32)

        outputs = self.session.run(
            [self.output_name],
            {self.input_name: crops_nchw},
        )
        feats = outputs[0].astype(np.float32)  # [B, D]

        # Cache dimension
        if self._embedding_dim is None:
            self._embedding_dim = feats.shape[1]

        # L2 normalise (mirrors extract_feature_vector_onnx.py)
        norms = np.linalg.norm(feats, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return feats / norms

    def get_embedding_dim(self) -> int:
        """Return embedding dimensionality (resolved after first batch or from model)."""
        if self._embedding_dim is not None:
            return self._embedding_dim
        # Read from model output shape if static
        out_shape = self.session.get_outputs()[0].shape
        if len(out_shape) == 2 and isinstance(out_shape[1], int):
            self._embedding_dim = out_shape[1]
            return self._embedding_dim
        return 512  # OSNet default fallback

