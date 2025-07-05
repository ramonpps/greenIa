import numpy as np

def record_based_encoding(features, num_dimensions=100):
    """
    Aplica codificação record-based no vetor de features.
    """
    random_projection = np.random.randn(features.shape[1], num_dimensions)
    encoded = np.dot(features, random_projection)
    return (encoded > 0).astype(int)  # Binariza os valores