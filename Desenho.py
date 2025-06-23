from ultralytics import YOLO
import cv2
import torch
import os
import matplotlib.pyplot as plt

model = YOLO('best.pt')

def predict_and_plot(image_path, conf_threshold=0.7):
    results = model(image_path)[0]

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    class_names = model.names
    print(class_names)

    for box in results.boxes:
        conf = box.conf[0].item()
        if conf >= conf_threshold:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            label = f"{class_names[cls_id]} {conf:.2f}"

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 0, 0), 2)

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')
    plt.title('Detecções (conf > {:.0f}%)'.format(conf_threshold * 100))
    plt.show()

predict_and_plot('dataset/train/images/20231214_145818_213_LWIR_R_jpg.rf.548e7579ecfd0be468930ce50e906889.jpg')
