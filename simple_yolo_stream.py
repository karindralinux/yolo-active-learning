from ultralytics import YOLO
import cv2

# Load pre-trained YOLO model (otomatis di-download saat pertama kali jalan)
model = YOLO('yolov8n.pt')

# Inisialisasi webcam Mac (index 0 adalah kamera bawaan)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if success:
        # Jalankan inferensi YOLO pada frame
        results = model(frame)
        
        # Render bounding box hasil deteksi di atas frame
        annotated_frame = results[0].plot()
        
        # Tampilkan ke layar macOS
        cv2.imshow("YOLOv8 macOS Inference", annotated_frame)
        
        # Tekan tombol 'q' di keyboard untuk berhenti
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()