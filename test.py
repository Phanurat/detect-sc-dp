import cv2
import face_recognition

# โหลดภาพ
image = face_recognition.load_image_file('face/person.jpg')

# หาตำแหน่งใบหน้า
face_locations = face_recognition.face_locations(image)

# โหลดภาพใหม่เพื่อใช้วาดกรอบ
image_cv2 = cv2.imread('face/person.jpg')

# วาดกรอบรอบใบหน้า
for face_location in face_locations:
    top, right, bottom, left = face_location
    # วาดกรอบสีแดง
    cv2.rectangle(image_cv2, (left, top), (right, bottom), (0, 0, 255), 2)

# แสดงภาพที่มีกรอบ
cv2.imshow('Image with Face Detection', image_cv2)
cv2.waitKey(0)
cv2.destroyAllWindows()
