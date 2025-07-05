import os
from feature_extraction import extract_features
from wisard_preprocessing import thermometer_encoding
from hdc_preprocessing import record_based_encoding
from wisard_classifier import evaluate_wisard
from hdc_classifier import train_hdc, evaluate_hdc
import numpy as np
from joblib import Parallel, delayed
import torch

def main():
    # Configura o dispositivo (CPU ou GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # Caminhos dos modelos e diretórios
    model_path = 'runs/train/yolov11-greenIA2/weights/best.pt'
    image_dir = 'valid/images'  # Diretório contendo as imagens
    label_dir = 'valid/labels'  # Diretório contendo os arquivos de rótulos

    # Listagem de imagens
    print("Listando arquivos de imagem...")
    image_paths = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith(('.jpg', '.png'))]
    print(f"Total de imagens encontradas: {len(image_paths)}")

    # Listagem de rótulos
    print("Listando arquivos de rótulos...")
    label_paths = [os.path.join(label_dir, os.path.basename(img).replace('.jpg', '.txt').replace('.png', '.txt')) for img in image_paths]
    print(f"Total de rótulos esperados: {len(label_paths)}")

    # Lendo os rótulos
    print("Lendo rótulos dos arquivos...")
    labels = []
    for label_path in label_paths:
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
                classes = [line.split()[0] for line in lines if line.strip()]
                if classes:
                    labels.append(classes[0])  # Usa a primeira classe como rótulo principal
                else:
                    print(f"Aviso: O arquivo de rótulo {label_path} está vazio ou mal formatado. Ignorando...")
                    labels.append("Unknown")
        else:
            print(f"Aviso: Arquivo de rótulo não encontrado: {label_path}")
            labels.append("Unknown")
    print(f"Total de rótulos lidos: {len(labels)}")

    # Verifica se o número de imagens corresponde ao número de rótulos
    if len(image_paths) != len(labels):
        raise ValueError("O número de imagens não corresponde ao número de rótulos.")

    # Processamento em lotes
    batch_size = 50  # Ajuste o tamanho do lote conforme necessário
    print(f"Processando imagens em lotes de tamanho {batch_size}...")

    all_features = []
    for i in range(0, len(image_paths), batch_size):
        batch_image_paths = image_paths[i:i + batch_size]
        print(f"Processando lote {i // batch_size + 1} com {len(batch_image_paths)} imagens...")

        # Extração de features para o lote atual
        batch_features = [extract_features(model_path, img) for img in batch_image_paths]
        batch_features = np.vstack(batch_features)  # Combina as features do lote em um único array
        print(f"Features extraídas para o lote {i // batch_size + 1}. Formato: {batch_features.shape}")

        # Converte as features para um tensor PyTorch e move para a GPU
        batch_features = torch.tensor(batch_features, device=device)
        batch_features = batch_features.view(batch_features.shape[0], -1)  # Achata as dimensões extras
        print(f"Features do lote {i // batch_size + 1} movidas para {device}. Formato: {batch_features.shape}")

        all_features.append(batch_features)

    # Combina todos os lotes em um único tensor
    features = torch.cat(all_features, dim=0)
    print(f"Formato final das features combinadas: {features.shape}")

    # WiSARD
    print("Iniciando codificação termométrica para WiSARD...")
    encoded_features_wisard = torch.cat(
        [thermometer_encoding(features[i:i + 1]) for i in range(features.shape[0])]
    )
    print(f"Formato das features codificadas para WiSARD: {encoded_features_wisard.shape}")
    print("Avaliando WiSARD...")
    wisard_accuracy = evaluate_wisard(encoded_features_wisard.cpu().numpy(), labels)
    print(f"WiSARD Accuracy: {wisard_accuracy:.4f}")

    # HDC
    print("Iniciando codificação para HDC...")

    # Reduzindo o número de dimensões na codificação record-based
    num_dimensions = 500  # Reduzindo para 500 dimensões
    print(f"Usando {num_dimensions} dimensões para a codificação HDC.")

    # Processamento em lotes para HDC
    batch_size = 50  # Ajuste o tamanho do lote conforme necessário
    encoded_features_hdc = []

    for i in range(0, features.shape[0], batch_size):
        batch_features = features[i:i + batch_size].cpu().numpy()
        print(f"Codificando lote {i // batch_size + 1} para HDC...")
        batch_encoded = record_based_encoding(batch_features, num_dimensions=num_dimensions)
        # Converte o array NumPy retornado para um tensor PyTorch
        batch_encoded_tensor = torch.tensor(batch_encoded, device=device)
        encoded_features_hdc.append(batch_encoded_tensor)

    # Concatena os lotes em um único tensor
    encoded_features_hdc = torch.cat(encoded_features_hdc, dim=0)
    print(f"Formato das features codificadas para HDC: {encoded_features_hdc.shape}")

    print("Treinando modelo HDC...")
    hdc_model = train_hdc(encoded_features_hdc.cpu().numpy(), labels)
    print("Avaliando HDC...")
    hdc_accuracy = evaluate_hdc(hdc_model, encoded_features_hdc.cpu().numpy(), labels)
    print(f"HDC Accuracy: {hdc_accuracy:.4f}")

if __name__ == "__main__":
    main()