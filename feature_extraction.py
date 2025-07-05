from ultralytics import YOLO
import torch
import cv2
import numpy as np

def extract_features(model_path, image_path):
    """
    Extrai o vetor de features da última camada convolucional do modelo YOLO.
    """
    model = YOLO(model_path)
    model.eval()

    # Carrega a imagem e redimensiona para o tamanho esperado pelo modelo
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (640, 640))  # Substitua pelo tamanho esperado pelo modelo
    image = image / 255.0  # Normaliza os valores dos pixels para [0, 1]
    image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float()  # Formata para [1, C, H, W]

    # Passa a imagem pelo modelo e obtém as features
    with torch.no_grad():
        outputs = model.model(image)  # Passa a imagem pelo modelo
        features = outputs[0]  # Obtém as features da última camada convolucional

    return features.cpu().numpy()