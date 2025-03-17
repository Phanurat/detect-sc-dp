import face_recognition
import pickle

# โหลดฐานข้อมูล encoding
with open('face_data/encodings.pkl', 'rb') as f:
    known_face_encodings = []
    known_face_names = []
    try:
        while True:
            name, encoding = pickle.load(f)
            known_face_encodings.append(encoding)
            known_face_names.append(name)
    except EOFError:
        pass  # หมดข้อมูลในไฟล์

# ฟังก์ชันในการรู้จำใบหน้าใหม่
def recognize_face(image_path):
    # โหลดภาพ
    image = face_recognition.load_image_file(image_path)

    # หา encoding ของใบหน้าภายในภาพ
    unknown_face_encodings = face_recognition.face_encodings(image)

    if unknown_face_encodings:
        for unknown_face_encoding in unknown_face_encodings:
            # เปรียบเทียบ encoding ของใบหน้าที่ไม่รู้จักกับฐานข้อมูล
            matches = face_recognition.compare_faces(known_face_encodings, unknown_face_encoding)

            name = "Unknown"  # ตั้งชื่อเป็น "Unknown" ถ้าไม่มีการจับคู่
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
            
            print(f"Face recognized as: {name}")
    else:
        print("No faces detected in the image.")

# เรียกใช้ฟังก์ชันเพื่อทำนาย
recognize_face('face/person5.jpg')  # ใส่เส้นทางของภาพใบหน้าที่ต้องการทำนาย
