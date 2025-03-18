import pyautogui
import time
from pythainlp import word_tokenize
import random

# กดปุ่ม Win เพื่อเปิด Start Menu
pyautogui.press('win')
time.sleep(1)

# พิมพ์ "Microsoft Edge"
pyautogui.write('Microsoft Edge', interval=0.1)
time.sleep(1)

# กด Enter เพื่อเปิด Microsoft Edge
pyautogui.press('enter')
time.sleep(3)

# กด Ctrl + L เพื่อโฟกัสที่แถบ URL
pyautogui.hotkey('ctrl', 'l')
time.sleep(1)

# วางลิงก์ Facebook
pyautogui.write('https://www.facebook.com/?filter=favorites&sk=h_chr', interval=0.05)
time.sleep(1)

# กด Enter เพื่อไปที่ลิงก์
pyautogui.press('enter')
time.sleep(5)  # รอให้เว็บโหลดเสร็จ

# ใช้ปุ่มลูกศรลงเพื่อเลื่อนหน้าจอลง
while True:
    pyautogui.press('down') 
    time.sleep(2)
    #words = []

