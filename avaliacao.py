from ultralytics import YOLO
import sys
import io
from datetime import datetime


def evaluate_model(model_path, data_path='data.yaml'):
    model = YOLO(model_path)

    buffer = io.StringIO()
    sys.stdout = buffer
    metrics = model.val(data=data_path, split='val', task='detect', workers=0)
    sys.stdout = sys.__stdout__

    # Extrai métricas
    precision = float(metrics.results_dict['metrics/precision(B)'])
    recall = float(metrics.results_dict['metrics/recall(B)'])
    map50 = float(metrics.results_dict['metrics/mAP50(B)'])
    map95 = float(metrics.results_dict['metrics/mAP50-95(B)'])
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)

    # Salva log em .txt, incluindo métricas resumidas
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = model_path.split('/')[-1].replace('.pt', '')
    log_file = f"log_{model_name}_{timestamp}.txt"
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(buffer.getvalue())
        f.write(f"\n📊 Avaliação do modelo: {model_name}\n")
        f.write(f"  Precision:  {precision:.4f}\n")
        f.write(f"  Recall:     {recall:.4f}\n")
        f.write(f"  mAP@50:     {map50:.4f}\n")
        f.write(f"  mAP@50-95:  {map95:.4f}\n")
        f.write(f"  F1 Score:   {f1_score:.4f}\n")

    # Exibe resumo no terminal
    print(f"\n📊 Avaliação do modelo: {model_name}")
    print(f"  Precision:  {precision:.4f}")
    print(f"  Recall:     {recall:.4f}")
    print(f"  mAP@50:     {map50:.4f}")
    print(f"  mAP@50-95:  {map95:.4f}")
    print(f"  F1 Score:   {f1_score:.4f}")
    print(f"  🔽 Log salvo em: {log_file}")


if __name__ == "__main__":
    print("Avaliando modelo original:")
    evaluate_model('runs/train/yolov11-greenIA2/weights/best.pt')

    print("\nAvaliando modelo podado:")
    evaluate_model('yolo11n_pruned.pt')

    print("\nAvaliando modelo quantizado:")
    evaluate_model('yolo11n_quantized.pt')
