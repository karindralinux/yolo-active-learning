# 🎓 Panduan Belajar YOLOv8 & Active Learning di macOS

Repositori ini dirancang sebagai tempat belajar praktis mengenai **Deteksi Objek (YOLOv8)** dan konsep mutakhir **Active Learning (Human-in-the-Loop)** yang dioptimalkan untuk macOS (Apple Silicon / MPS).

---

## 🗺️ Peta Struktur Repositori

Untuk belajar secara efektif, disarankan membaca file dengan urutan berikut:

```text
yolo-mac-test/
├── 1. simple_yolo_stream.py      # [LANGKAH 1] Mulai dari sini: Inferensi real-time webcam
├── 2. datasets/                  # [LANGKAH 2] Pelajari format data & anotasi YOLO
│   └── coco8/labels/train/...    # Contoh format label .txt (class_id, x, y, w, h)
├── 3. active_learning/
│   ├── main.py                   # [LANGKAH 3] Inti Active Learning: Siklus simulasi otomatis
│   └── best_al_model.pt          # Model checkpoint hasil pelatihan Active Learning
├── active_learning_sim/          # Folder kerja simulasi (dihasilkan otomatis oleh main.py)
│   ├── dataset.yaml              # Konfigurasi data & pemetaan class 0 s/d 79
│   ├── images/ & labels/         # Data train (bertambah tiap siklus) & data val
│   └── oracle_vault/             # "Brankas rahasia" label asli yang disimulasikan
└── runs/                         # Log pelatihan, grafik loss, dan evaluasi mAP Ultralytics
```

---

## 📚 Tahapan Belajar

### Tahap 1: Inferensi Dasar dengan Webcam (`simple_yolo_stream.py`)
File ini adalah gerbang termudah untuk memahami cara YOLO bekerja:
1. **Memuat Model**: Menggunakan bobot nano `yolov8n.pt`.
2. **Kamera OpenCV**: Mengakses webcam Mac (`cv2.VideoCapture(0)`).
3. **Inferensi & Rendering**:
   ```python
   results = model(frame)         # Prediksi bounding box
   annotated_frame = results[0].plot()  # Menggambar kotak & label di layar
   ```
*Jalankan dengan:*
```bash
source venv/bin/activate
python simple_yolo_stream.py
```
*(Tekan tombol `q` untuk keluar dari layar webcam).*

---

### Tahap 2: Memahami Format Data & Anotasi YOLO
Buka contoh label di `datasets/coco8/labels/train/000000000009.txt`:
```text
45 0.479492 0.688771 0.955609 0.5955
```
Setiap baris mewakili 1 objek deteksi dengan 5 nilai:
1. **`class_id` (45)**: ID objek (di dataset COCO, `45` = `bowl` / mangkuk).
2. **`x_center` (0.479)**: Posisi X titik tengah kotak (relatif terhadap lebar gambar, 0.0 - 1.0).
3. **`y_center` (0.688)**: Posisi Y titik tengah kotak (relatif terhadap tinggi gambar, 0.0 - 1.0).
4. **`width` (0.955)**: Lebar kotak relatif terhadap lebar gambar.
5. **`height` (0.595)**: Tinggi kotak relatif terhadap tinggi gambar.

> **Tips:** Mengapa desimal normalisasi? Agar koordinat tetap valid meskipun gambar diubah resolusinya (misal di-resize ke 640x640 saat training).

---

### Tahap 3: Memahami Simulasi Active Learning (`active_learning/main.py`)

#### Mengapa Perlu Active Learning?
Dalam industri, melabeli 100.000 gambar membutuhkan biaya dan waktu yang sangat besar. **Active Learning (AL)** menyelesaikan masalah ini:
> *"Daripada melabeli semua data secara acak, biarkan model memilih hanya data yang membuatnya paling bingung (uncertainty) untuk dilabeli oleh manusia."*

#### Bagaimana Skrip Ini Bekerja Tanpa Repot (Simulated Oracle)?
Skrip `active_learning/main.py` dirancang **100% plug & play**:
1. **Auto-download & Setup**: Otomatis menyiapkan dataset `coco128` (128 gambar).
2. **Pemisahan Data**:
   - `Val Set` (28 gambar): Benchmark tetap untuk mengukur evaluasi akurasi mAP riil.
   - `Initial Seed Train` (10 gambar): Modal awal model di Siklus 0.
   - `Unlabeled Pool` (90 gambar): Gambar mentah yang pura-puranya belum memiliki label.
   - `Oracle Vault`: Tempat persembunyian label asli (mensimulasikan manusia yang tahu jawaban benar).
3. **Siklus Pembelajaran (Loop)**:
   - **Inference**: Model memprediksi sisa gambar di `Pool`.
   - **Uncertainty Scoring**: Gambar dengan prediksi paling tidak yakin (*least confidence*) atau tidak terdeteksi sama sekali diberi skor ketidakpastian tertinggi.
   - **Simulated Oracle**: Script mengambil 10 gambar paling membingungkan dan "membuka" label aslinya dari brankas ke folder latih.
   - **Retraining Inkremental**: Melatih model kembali dengan akselerasi Apple Silicon GPU (`mps`).
   - **Evaluasi**: Mengukur kenaikan $mAP_{50}$ pada `Val Set`.
4. **Hasil Akhir**: Model terbaik disimpan ke `active_learning/best_al_model.pt`.

*Jalankan dengan:*
```bash
source venv/bin/activate
python active_learning/main.py
```

---

### Tahap 4: Menguji Model Hasil Active Learning di Webcam
Setelah simulasi selesai dan menghasilkan `active_learning/best_al_model.pt`, Anda bisa mengujinya langsung:

Buka `simple_yolo_stream.py` dan ubah:
```python
# Ganti dari yolov8n.pt bawaan:
model = YOLO('active_learning/best_al_model.pt')
```
Lalu jalankan `python simple_yolo_stream.py`.

---

## ⚙️ Ringkasan Parameter Kunci di `active_learning/main.py`

| Parameter | Default | Fungsi |
|---|---|---|
| `ITERATIONS` | `3` | Banyaknya siklus Active Learning dijalankan. |
| `BUDGET_PER_ITER` | `10` | Jumlah gambar paling ragu yang dilabeli per siklus. |
| `EPOCHS_PER_ITER` | `3` | Jumlah epoch retraining per siklus (cepat di Mac M1/M2/M3). |
| `SEED_SIZE` | `10` | Jumlah gambar latih awal untuk baseline. |
| `VAL_SIZE` | `28` | Jumlah gambar validasi tetap untuk benchmark mAP. |
| `DEVICE` | `mps` | Akselerasi GPU Metal Apple Silicon. |

---

## 📈 Cara Membaca Metrik Evaluasi

Pada akhir proses simulasi, Anda akan melihat tabel:
- **`mAP@50`**: Mean Average Precision pada IoU threshold 0.50 (semakin mendekati 1.0, semakin akurat).
- **`mAP@50-95`**: Metrik komprehensif rata-rata pada IoU 0.50 hingga 0.95.
- Peningkatan nilai mAP dari **Baseline -> Siklus 1 -> Siklus 2** membuktikan bahwa Active Learning berhasil meningkatkan kecerdasan model secara efisien dengan data yang sedikit.
