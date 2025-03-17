import tensorflow as tf
from tensorflow.keras import layers, models

def create_face_model(input_shape):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dense(2, activation='softmax')  # 2 class สำหรับ 2 คน (สามารถเพิ่มจำนวนได้ตามต้องการ)
    ])
    return model
