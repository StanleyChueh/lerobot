import cv2

cap = cv2.VideoCapture(6)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("無法接收影格，正在退出...")
        break
    
    # 處理影像...
    cv2.imshow('frame', frame)
    
    if cv2.waitKey(1) == ord('q'):
        break

# 確保 release 發生在所有讀取行為之後
cap.release()
cv2.destroyAllWindows()
