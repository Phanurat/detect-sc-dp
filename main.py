import cv2
import numpy as np
import mss
from ultralytics import YOLO
import pytesseract
from PIL import Image
import time

model = YOLO("yolov8n.pt")
monitor = {"top": 100, "left": 100, "width": 800, "height": 600}
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

with mss.mss() as sct:
    while True:
        screenshot = sct.grab(monitor)
        img = np.array(screenshot)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        results = model(img)
        time.sleep(2)
        image = Image.fromarray(img) 
        text = pytesseract.image_to_string(image, lang='tha')
        print(text)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                label = r.names[int(box.cls[0])]
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("YOLOv8 Screen Detection", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()
