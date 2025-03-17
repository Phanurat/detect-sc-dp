import tensorflow as tf
import cv2
import numpy as np

# โหลดโมเดลที่ฝึกไว้ล่วงหน้า
model = tf.keras.applications.MobileNetV2(weights='imagenet', input_shape=(224, 224, 3))

# โหลดภาพ
image = cv2.imread('face/person.jpg')

# แปลงภาพเป็นรูปแบบที่โมเดลต้องการ
image_resized = cv2.resize(image, (224, 224))
image_resized = np.expand_dims(image_resized, axis=0)
image_resized = tf.keras.applications.mobilenet_v2.preprocess_input(image_resized)

# ทำการทำนาย
predictions = model.predict(image_resized)

# แสดงผลลัพธ์
decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions)
print(decoded_predictions)
