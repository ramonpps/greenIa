from ultralytics import YOLO


def treinar_modelo():
    model = YOLO("../modelos/yolo11n.pt")

    model.train(
        data='data.yaml',  
        epochs=50,
        imgsz=640,
        batch=-1,
        name='yolov11-greenIA',
        project='runs/train',
        workers=20,
        device = 'cuda'
    )
    

    metrics = model.val(
        data='dataset/data.yaml',
        split='test'
    )
if __name__ == "__main__":
    treinar_modelo()