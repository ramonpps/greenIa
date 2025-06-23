import torch
from ultralytics import YOLO

original_yolo_model = YOLO('runs/train/yolov11-greenIA2/weights/best.pt')
model_to_quantize = original_yolo_model.model

def quantization(model):
    model.eval()
    q_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
    )
    print("Quantização dinâmica aplicada.")
    return q_model

def save_quantized_model(yolo_object, path="yolo11n_quantized.pt"):
    yolo_object.model = quantized_pytorch_model
    yolo_object.save(path)
    print(f"Modelo quantizado salvo em {path}")


quantized_pytorch_model = quantization(model_to_quantize)
save_quantized_model(original_yolo_model, "yolo11n_quantized.pt")