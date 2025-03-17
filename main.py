import cv2
import numpy as np
import mss
from ultralytics import YOLO

# โหลดโมเดล YOLOv8 ที่ผ่านการฝึกมาแล้ว (ใช้โมเดลที่พร้อมใช้งาน)
model = YOLO("yolov8n.pt")  # ใช้โมเดลเวอร์ชันเล็ก (nano) เพื่อลดการใช้ทรัพยากร

# กำหนดขนาดของพื้นที่จับภาพ
monitor = {"top": 100, "left": 100, "width": 800, "height": 600}

# เริ่มจับภาพหน้าจอแบบเรียลไทม์
with mss.mss() as sct:
    while True:
        # จับภาพหน้าจอ
        screenshot = sct.grab(monitor)
        img = np.array(screenshot)

        # แปลงจาก BGRA เป็น BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        # ตรวจจับวัตถุด้วย YOLO
        results = model(img)

        # แสดงผลการตรวจจับ
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # พิกัดของ bounding box
                conf = box.conf[0]  # ความมั่นใจของการตรวจจับ
                label = r.names[int(box.cls[0])]  # ประเภทของวัตถุที่ตรวจจับได้

                # วาดกรอบสี่เหลี่ยมรอบวัตถุ
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # แสดงภาพที่มีการตรวจจับ
        cv2.imshow("YOLOv8 Screen Detection", img)

        # กด 'q' เพื่อออกจากโปรแกรม
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# ปิดหน้าต่างทั้งหมด
cv2.destroyAllWindows()
