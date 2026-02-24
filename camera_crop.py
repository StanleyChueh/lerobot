import cv2

def main():
    cap = cv2.VideoCapture(4)  # Change to your camera index

    if not cap.isOpened():
        print("Error: Cannot open camera")
        return

    roi_selected = False
    roi = None

    print("Press 's' to select ROI")
    print("Press 'r' to reselect ROI")
    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        display_frame = frame.copy()

        if roi_selected:
            x, y, w, h = roi

            # Draw rectangle on original camera window
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Crop ROI
            roi_frame = frame[y:y+h, x:x+w]

            # Resize to 640x480
            roi_resized = cv2.resize(roi_frame, (640, 480))

            # Show ROI in color (BGR format)
            cv2.imshow("ROI 640x480 RGB", roi_resized)

        cv2.imshow("Camera", display_frame)

        key = cv2.waitKey(1) & 0xFF

        # Select ROI
        if key == ord('s'):
            roi = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)
            cv2.destroyWindow("Select ROI")

            # Check if valid selection
            if roi[2] > 0 and roi[3] > 0:
                roi_selected = True
            else:
                roi_selected = False

        # Reselect ROI
        elif key == ord('r'):
            roi_selected = False
            cv2.destroyWindow("ROI 640x480 RGB")

        # Quit
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
