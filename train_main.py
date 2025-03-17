import cv2
import face_recognition
import numpy as np
import os

# 📂 โฟลเดอร์ที่เก็บภาพของบุคคลที่รู้จัก
KNOWN_FACES_DIR = "known_faces"
known_faces = []
known_names = []

# ✅ โหลดภาพทุกภาพในโฟลเดอร์
for name in os.listdir(KNOWN_FACES_DIR):  # วนลูปชื่อบุคคลในโฟลเดอร์
    person_dir = os.path.join(KNOWN_FACES_DIR, name)
    
    if not os.path.isdir(person_dir):
        continue  # ข้ามถ้าไม่ใช่โฟลเดอร์

    for filename in os.listdir(person_dir):  # วนลูปรูปภาพของบุคคลนี้
        filepath = os.path.join(person_dir, filename)
        
        image = face_recognition.load_image_file(filepath)
        encodings = face_recognition.face_encodings(image)
        
        if len(encodings) > 0:
            known_faces.append(encodings[0])  # เก็บ encoding ใบหน้า
            known_names.append(name)  # เก็บชื่อบุคคล

print(f"📌 โหลดรูปภาพ {len(known_faces)} ใบหน้าสำเร็จ!")

# 📷 เปิดกล้องหรือจับภาพหน้าจอ
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 🔍 ค้นหาใบหน้าในภาพ
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_faces, face_encoding)
        name = "Unknown"

        face_distances = face_recognition.face_distance(known_faces, face_encoding)
        best_match_index = np.argmin(face_distances)  # เลือกค่าที่ใกล้เคียงที่สุด

        if matches[best_match_index]:
            name = known_names[best_match_index]  # ใช้ชื่อของบุคคลที่ตรงมากที่สุด

        # ✨ แสดงผลบนหน้าจอ
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
