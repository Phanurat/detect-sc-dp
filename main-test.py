import cv2
import numpy as np
import mss
from ultralytics import YOLO
import pytesseract
from PIL import Image
import time

# โหลดโมเดล YOLO
model = YOLO("yolov8n.pt")

# กำหนดพื้นที่ที่จะจับภาพหน้าจอ
monitor = {"top": 100, "left": 100, "width": 800, "height": 600}

# ตั้งค่า path ของ tesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# เริ่มการจับภาพหน้าจอและประมวลผล
with mss.mss() as sct:
    while True:
        # จับภาพหน้าจอจากพื้นที่ที่กำหนด
        screenshot = sct.grab(monitor)
        img = np.array(screenshot)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # ตรวจจับวัตถุด้วย YOLO
        results = model(img)

        # แปลงเป็นภาพที่สามารถใช้กับ Tesseract OCR
        image = Image.fromarray(img)
        
        # ใช้ Tesseract OCR ในการอ่านข้อความจากภาพ
        text = pytesseract.image_to_string(image, lang='tha')
        print(text)

        # ตรวจจับตัวอักษรในภาพและวาดกรอบสี่เหลี่ยมรอบข้อความ
        h, w, _ = img.shape
        boxes = pytesseract.image_to_boxes(image, lang='tha')

        # วาดกรอบสี่เหลี่ยมรอบข้อความที่ตรวจพบ
        for b in boxes.splitlines():
            b = b.split()
            x, y, x2, y2 = int(b[1]), int(b[2]), int(b[3]), int(b[4])
            cv2.rectangle(img, (x, h - y), (x2, h - y2), (0, 255, 0), 2)

        # ตรวจจับวัตถุด้วย YOLO และวาดกรอบสี่เหลี่ยมรอบวัตถุ
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                label = r.names[int(box.cls[0])]
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # แสดงภาพที่ประมวลผล
        cv2.imshow("YOLOv8 Screen Detection", img)

        # หยุดการทำงานเมื่อกดปุ่ม 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# ปิดหน้าต่าง
cv2.destroyAllWindows()