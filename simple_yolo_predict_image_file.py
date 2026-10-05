import os
import sys
from ultralytics import YOLO
import cv2

# Load pre-trained YOLO model (otomatis di-download saat pertama kali jalan)
# Ganti ke 'active_learning/best_al_model.pt' untuk memakai model hasil active learning
model = YOLO('yolov8n.pt')

# Ambil path gambar dari argumen CLI, atau minta input jika tidak diberikan
if len(sys.argv) > 1:
    image_path = sys.argv[1]
else:
    image_path = input("Masukkan path gambar: ").strip().strip("'\"")

image_path = os.path.expanduser(image_path)

if not os.path.isfile(image_path):
    sys.exit(f"File tidak ditemukan: {image_path}")

# Jalankan inferensi YOLO pada gambar
results = model(image_path)

# Cetak ringkasan deteksi
for box in results[0].boxes:
    cls_id = int(box.cls[0])
    print(f"{model.names[cls_id]}: {float(box.conf[0]):.2f}")

# Render bounding box hasil deteksi di atas gambar
annotated_image = results[0].plot()

# Tampilkan ke layar macOS, tekan tombol apa saja untuk menutup
cv2.imshow("YOLOv8 macOS Image Prediction", annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
