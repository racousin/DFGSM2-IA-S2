"""Data generation utilities for L2Math practicals."""

import numpy as np


def generate_linear_data(n_samples=100, n_features=1, noise=1.0, bias=5.0, weight_scale=3.0, seed=42):
    """
    Generate synthetic data for linear regression.

    Args:
        n_samples: Number of samples
        n_features: Number of input features
        noise: Standard deviation of Gaussian noise
        bias: Bias value
        weight_scale: Scale of the weights
        seed: Random seed for reproducibility

    Returns:
        X: Input array of shape (n_samples, n_features)
        y: Target array of shape (n_samples, 1)
        true_weights: True weights used for generation
        true_bias: True bias used for generation
    """
    np.random.seed(seed)

    # Generate true parameters
    true_weights = np.random.randn(n_features, 1) * weight_scale
    true_bias = np.array([bias])

    # Generate input data
    X = np.random.randn(n_samples, n_features) * 2

    # Generate targets with noise
    y = X @ true_weights + true_bias + np.random.randn(n_samples, 1) * noise

    print(f"Generated linear dataset:")
    print(f"  - Samples: {n_samples}")
    print(f"  - Features: {n_features}")
    print(f"  - X shape: {X.shape}")
    print(f"  - y shape: {y.shape}")
    print(f"  - True weights: {true_weights.flatten()}")
    print(f"  - True bias: {true_bias[0]:.4f}")

    return X, y, true_weights, true_bias
