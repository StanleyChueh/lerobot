import cv2

# 定義滑鼠回呼函式
def show_coords(event, x, y, flags, param):
    # 當滑鼠左鍵點擊時 (LBUTTONDOWN)
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"點擊座標: X={x}, Y={y}")
        # 在畫面上畫一個小圓點標示
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        # 在畫面上顯示座標文字
        cv2.putText(img, f"({x},{y})", (x + 10, y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# 1. 讀取 /dev/video4
cap = cv2.VideoCapture(4)

if not cap.isOpened():
    print("無法開啟 /dev/video4，請檢查權限或路徑。")
    exit()

# 2. 建立視窗並綁定滑鼠事件
cv2.namedWindow("Image")
cv2.setMouseCallback("Image", show_coords)

print("程式執行中。請在視窗上點擊滑鼠左鍵查看座標，按 'q' 鍵退出。")

while True:
    ret, img = cap.read()
    if not ret:
        break

    cv2.imshow("Image", img)

    # 按下 'q' 鍵退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()