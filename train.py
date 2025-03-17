import cv2
import face_recognition
import os
import pickle

# สร้างโฟลเดอร์เพื่อเก็บภาพถ่ายใบหน้า
if not os.path.exists('face_data'):
    os.makedirs('face_data')

# ฟังก์ชันเก็บ face encoding
def collect_face_encodings(name, image_path):
    # โหลดภาพ
    image = face_recognition.load_image_file(image_path)

    # หา encoding ของใบหน้า
    face_encodings = face_recognition.face_encodings(image)
    
    # ถ้าพบใบหน้า
    if face_encodings:
        encoding = face_encodings[0]  # ใช้ encoding ของใบหน้าตัวแรก
        return name, encoding
    else:
        return None, None

# เก็บข้อมูลใบหน้าของผู้ใช้
name = "nakoy_ink"
image_path = "face/person4.jpg"  # ใส่เส้นทางของภาพใบหน้าที่ต้องการเก็บ

name, encoding = collect_face_encodings(name, image_path)
if encoding is not None:
    # บันทึก encoding และชื่อ
    with open('face_data/encodings.pkl', 'ab') as f:
        pickle.dump((name, encoding), f)
    print(f"Encoding for {name} saved.")
else:
    print("No face detected in the image.")
