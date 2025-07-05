import numpy as np
import torch

def thermometer_encoding(features, num_levels=3):
    """
    Aplica codificação termométrica no vetor de features usando PyTorch.
    """
    # Garante que as features estejam no dispositivo correto
    device = features.device if isinstance(features, torch.Tensor) else "cpu"
    features = torch.tensor(features, device=device) if not isinstance(features, torch.Tensor) else features

    # Normaliza as features para o intervalo [0, 1]
    min_val, max_val = features.min(), features.max()
    normalized = (features - min_val) / (max_val - min_val)

    # Codificação termométrica
    thresholds = torch.linspace(0, 1, num_levels, device=device)
    encoded = (normalized.unsqueeze(-1) >= thresholds).float()

    # Achata a última dimensão para retornar um tensor bidimensional
    encoded = encoded.view(features.shape[0], -1)

    return encoded