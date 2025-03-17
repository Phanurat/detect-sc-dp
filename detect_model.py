import cv2

# โหลดโมเดล Haar Cascade ที่ฝึกไว้แล้ว
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# โหลดภาพที่ต้องการตรวจจับ
image = cv2.imread('face/person.jpg')

# แปลงภาพเป็นสีเทา
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# ตรวจจับใบหน้า
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

# วาดกรอบรอบใบหน้า
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

# แสดงภาพ
cv2.imshow('Detected Faces', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
