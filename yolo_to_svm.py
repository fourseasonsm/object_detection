import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import hog

# 1. Создаем выходную папку с проверкой
output_dir = 'svm_input'
os.makedirs(output_dir, exist_ok=True)

def extract_hog_features(img):
    """Извлечение HOG-признаков"""
    resized = cv2.resize(img, (64, 64))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    features = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), feature_vector=True)
    return features


# Загрузка модели YOLO
net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

for img_name in os.listdir('test_img'):
    img_path = os.path.join('test_img', img_name)
    image = cv2.imread(img_path)
    if image is None:
        continue

    height, width = image.shape[:2]

    # Детекция объектов
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    svm_data = []
    for out in outs:
        for detection in out:
            # Извлечение confidence
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = float(scores[class_id])  # Важно: преобразование в float

            # Только объекты с low confidence
            if confidence < 0.7 and confidence > 0:
                # Преобразование координат
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = max(0, int(center_x - w / 2))
                y = max(0, int(center_y - h / 2))

                # Вырезаем ROI с проверкой границ
                roi = image[y:y + h, x:x + w]
                if roi.size == 0:
                    continue

                # Извлекаем признаки
                try:
                    hog_features = extract_hog_features(roi)
                    svm_data.append({
                        'image': img_name,
                        'class_id': int(class_id) + 1,
                        'yolo_confidence': confidence,
                        'features': ' '.join(map(str, hog_features))  # Оптимизированное сохранение
                    })
                except:
                    continue

    # Сохранение в CSV
    if svm_data:
        df = pd.DataFrame(svm_data)
        output_path = os.path.join('svm_input', f"{os.path.splitext(img_name)[0]}.csv")
        df.to_csv(output_path, index=False)
        print(f"Обработано {img_name}: {len(svm_data)} объектов с confidence < 0.7")