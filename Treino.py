from ultralytics import YOLO


def treinar_modelo():
    model = YOLO("yolo11n.pt")
    # Treinamento
    model.train(
        data='data.yaml',  # Caminho para o data.yaml
        epochs=50,
        imgsz=640,
        batch=-1,
        name='yolov11-greenIA',
        project='runs/train',
        workers=20,
        device = 'cuda'
    )
    
    # Avaliação no conjunto de teste
    metrics = model.val(
        data='dataset/data.yaml',
        split='test'
    )
if __name__ == "__main__":
    treinar_modelo()

