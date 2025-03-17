import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split

# ฟังก์ชันโหลดข้อมูลจากโฟลเดอร์
def load_images_from_folder(folder, label, size=(128, 128)):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            img = cv2.resize(img, size)  # ปรับขนาดภาพ
            images.append(img)
            labels.append(label)
    return images, labels

# โหลดข้อมูลจากโฟลเดอร์
folder_1 = 'dataset/nakoy'  # โฟลเดอร์ของบุคคลคนแรก
try:
    images_1, labels_1 = load_images_from_folder(folder_1, label=0)
    print("ข้อมูลถูกโหลดเรียบร้อยแล้ว")
except Exception as e:
    print(f"เกิดข้อผิดพลาด: {e}")

# รวมข้อมูลจากโฟลเดอร์
images = images_1
labels = labels_1  # labels_1 คือตัวแปรที่มี label สำหรับภาพจาก folder_1

# แปลงเป็น numpy array
images = np.array(images)
labels = np.array(labels)

# แบ่งข้อมูลเป็นชุดฝึก (train) และชุดทดสอบ (test)
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# ปรับค่าพิกเซลภาพให้อยู่ในช่วง 0-1
X_train, X_test = X_train / 255.0, X_test / 255.0
