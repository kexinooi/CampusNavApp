import cv2
import numpy as np

# --- USER SETTINGS ---
KNOWN_QR_WIDTH_CM = 15.0      # Width of your printed QR code
KNOWN_DISTANCE_M = 0.5       # Measured distance between QR and camera during test

# --- INIT ---
cap = cv2.VideoCapture(0)  # Open webcam (change to video file or image if needed)
detector = cv2.QRCodeDetector()

print("Press 'C' to start capturing QR code.")

# Variable to track capture mode
capturing = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Show message when waiting for capture to start
    if not capturing:
        cv2.putText(frame, "Press 'C' to start capturing", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    else:
        # QR code detection
        data, points, _ = detector.detectAndDecode(frame)

        if points is not None:
            points = points[0]  # shape (4, 2)

            # Draw bounding box around the QR code
            for i in range(4):
                pt1 = tuple(points[i].astype(int))
                pt2 = tuple(points[(i + 1) % 4].astype(int))
                cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

            # Compute bounding box width (avg of top and bottom edges)
            top_width = np.linalg.norm(points[0] - points[1])
            bottom_width = np.linalg.norm(points[3] - points[2])
            bbox_width_px = (top_width + bottom_width) / 2

            cv2.putText(frame, f"Width(px): {bbox_width_px:.2f}", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        else:
            bbox_width_px = None

    cv2.imshow("QR Calibration", frame)
    key = cv2.waitKey(1)

    # Press 'C' to start capturing
    if key == 99:  # ASCII for 'C'
        capturing = True
        print("Started capturing QR code. Press 'SPACE' to calibrate.")

    # Press SPACE to capture and compute focal length
    elif key == 32 and capturing and bbox_width_px is not None:
        focal_length_px = (bbox_width_px * KNOWN_DISTANCE_M) / (KNOWN_QR_WIDTH_CM / 100)
        print(f"✅ Calibration Complete")
        print(f"  ➤ BBox width (pixels): {bbox_width_px:.2f}")
        print(f"  ➤ Focal Length (pixels): {focal_length_px:.2f}")
        print("You can now use this focal length in your distance estimation.")
        break

    # Press ESC to quit
    elif key == 27:
        print("❌ Calibration canceled.")
        break

cap.release()
cv2.destroyAllWindows()
