import torch
from ultralytics import YOLO
from torch.nn.utils import prune

original_yolo_model = YOLO('runs/train/yolov11-greenIA2/weights/best.pt')
model_to_prune = original_yolo_model.model

def pruning(model, amount=0.2):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=amount)
            prune.remove(module, 'weight')
            print(f"Pruned {name} by {amount * 100}%")

    return model

def save_pruned_model(yolo_object, path="../modelos/yolo11n_pruned.pt"):
    yolo_object.model = model_to_prune
    yolo_object.save(path)
    print(f"Modelo podado salvo em {path}")

if __name__ == "__main__":
    pruned_pytorch_model = pruning(model_to_prune, amount=0.2)
    save_pruned_model(original_yolo_model)