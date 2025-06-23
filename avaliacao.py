from ultralytics import YOLO

def evaluate_model(model_path, data_path='data.yaml'):
    model = YOLO(model_path)
    metrics = model.val(data=data_path, split='val')
    print(metrics)

if __name__ == "__main__":
    print("Avaliando modelo original:")
    evaluate_model('runs/train/yolov11-greenIA2/weights/best.pt')

    print("Avaliando modelo podado:")
    evaluate_model('yolo11n_pruned.pt')

    print("Avaliando modelo quantizado:")
    evaluate_model('yolo11n_quantized.pt')