"""
Active Learning Simulation for YOLOv8 (Single Self-Contained Script)
-------------------------------------------------------------------
Pendekatan "Paling Tidak Merepotkan" (Plug & Play):
1. Menggunakan dataset COCO128 bawaan Ultralytics (otomatis disiapkan jika belum ada).
2. Membagi data menjadi:
   - Val Set (28 gambar): Benchmark tetap untuk mengukur evaluasi akurasi riil.
   - Seed Train Set (10 gambar): Dataset latih awal (Cycle 0).
   - Unlabeled Pool (90 gambar): Gambar mentah yang disimulasikan belum berlabel.
   - Oracle Vault: Label ground-truth asli yang disimpan tersembunyi.
3. Siklus Active Learning (Uncertainty Sampling -> Simulated Oracle -> Retrain -> Evaluate).
4. Otomatis menggunakan akselerasi GPU Metal (MPS) pada macOS Apple Silicon.
5. Menyimpan model terbaik ke 'active_learning/best_al_model.pt' yang siap dipakai di webcam (simple_yolo.py).
"""

import os
import shutil
import glob
import torch
import yaml
from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset

# ==============================================================================
# 1. KONFIGURASI PARAMETER ACTIVE LEARNING
# ==============================================================================
WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(WORKSPACE_DIR)

# Folder kerja simulasi (terisolasi agar data asli aman & dapat di-run ulang)
SIM_DIR = os.path.join(PROJECT_ROOT, "active_learning_sim")
TRAIN_IMG_DIR = os.path.join(SIM_DIR, "images", "train")
TRAIN_LBL_DIR = os.path.join(SIM_DIR, "labels", "train")
VAL_IMG_DIR = os.path.join(SIM_DIR, "images", "val")
VAL_LBL_DIR = os.path.join(SIM_DIR, "labels", "val")
POOL_IMG_DIR = os.path.join(SIM_DIR, "pool", "images")
ORACLE_VAULT_DIR = os.path.join(SIM_DIR, "oracle_vault", "labels")
DATASET_YAML_PATH = os.path.join(SIM_DIR, "dataset.yaml")

# Parameter Simulasi
SEED_SIZE = 10         # Jumlah gambar awal untuk baseline training
VAL_SIZE = 28          # Jumlah gambar evaluasi tetap
ITERATIONS = 3         # Jumlah siklus Active Learning
BUDGET_PER_ITER = 10   # Jumlah gambar paling ragu (uncertain) yang dilabeli per siklus
EPOCHS_PER_ITER = 3    # Epoch retraining per siklus (3 epoch cepat & terukur di Mac)
IMGSZ = 640
BATCH_SIZE = 8

# Deteksi perangkat hardware (Apple Silicon MPS / CUDA / CPU)
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "0"
else:
    DEVICE = "cpu"


# ==============================================================================
# 2. PERSIAPAN DATASET SIMULASI (SIMULATED ORACLE SETUP)
# ==============================================================================
def prepare_simulation_environment():
    """
    Menyiapkan dataset COCO128 dan membuat pembagian folder yang bersih:
    - Train (Seed awal)
    - Val (Benchmark tetap)
    - Pool (Gambar mentah tanpa label yang dapat dilihat model)
    - Oracle Vault (Tempat penyimpanan rahasia label asli)
    """
    print("\n" + "="*60)
    print(" [1/4] MENYIAPKAN DATASET & ENVIRONMENT SIMULASI")
    print("="*60)
    
    # 1. Pastikan coco128 tersedia
    dataset_info = check_det_dataset("coco128.yaml")
    coco_path = dataset_info["path"]
    src_images = sorted(glob.glob(os.path.join(coco_path, "images", "train2017", "*.jpg")))
    src_labels_dir = os.path.join(coco_path, "labels", "train2017")
    
    total_found = len(src_images)
    print(f"-> Ditemukan {total_found} gambar dari COCO128.")
    
    # 2. Reset folder kerja simulasi agar selalu bersih & idempoten
    if os.path.exists(SIM_DIR):
        shutil.rmtree(SIM_DIR)
        
    for d in [TRAIN_IMG_DIR, TRAIN_LBL_DIR, VAL_IMG_DIR, VAL_LBL_DIR, POOL_IMG_DIR, ORACLE_VAULT_DIR]:
        os.makedirs(d, exist_ok=True)
        
    # 3. Pembagian Data:
    # - val: 0 s/d VAL_SIZE
    # - seed: VAL_SIZE s/d (VAL_SIZE + SEED_SIZE)
    # - pool: sisanya
    val_files = src_images[:VAL_SIZE]
    seed_files = src_images[VAL_SIZE:VAL_SIZE + SEED_SIZE]
    pool_files = src_images[VAL_SIZE + SEED_SIZE:]
    
    # Isi Val Set (beserta label)
    for img_path in val_files:
        base_name = os.path.basename(img_path)
        lbl_name = os.path.splitext(base_name)[0] + ".txt"
        shutil.copy(img_path, os.path.join(VAL_IMG_DIR, base_name))
        lbl_path = os.path.join(src_labels_dir, lbl_name)
        if os.path.exists(lbl_path):
            shutil.copy(lbl_path, os.path.join(VAL_LBL_DIR, lbl_name))
            
    # Isi Seed Train Set (beserta label)
    for img_path in seed_files:
        base_name = os.path.basename(img_path)
        lbl_name = os.path.splitext(base_name)[0] + ".txt"
        shutil.copy(img_path, os.path.join(TRAIN_IMG_DIR, base_name))
        lbl_path = os.path.join(src_labels_dir, lbl_name)
        if os.path.exists(lbl_path):
            shutil.copy(lbl_path, os.path.join(TRAIN_LBL_DIR, lbl_name))
            
    # Isi Pool (HANYA gambar) & simpan label asli di Oracle Vault
    for img_path in pool_files:
        base_name = os.path.basename(img_path)
        lbl_name = os.path.splitext(base_name)[0] + ".txt"
        shutil.copy(img_path, os.path.join(POOL_IMG_DIR, base_name))
        lbl_path = os.path.join(src_labels_dir, lbl_name)
        if os.path.exists(lbl_path):
            shutil.copy(lbl_path, os.path.join(ORACLE_VAULT_DIR, lbl_name))
            
    # 4. Buat dataset.yaml dinamis dengan 80 class COCO
    yaml_dict = {
        "path": os.path.abspath(SIM_DIR),
        "train": "images/train",
        "val": "images/val",
        "names": dataset_info["names"]
    }
    with open(DATASET_YAML_PATH, "w") as f:
        yaml.safe_dump(yaml_dict, f, sort_keys=False)
        
    print(f"-> Val Set            : {len(os.listdir(VAL_IMG_DIR))} gambar (Fixed Benchmark)")
    print(f"-> Initial Seed Train : {len(os.listdir(TRAIN_IMG_DIR))} gambar")
    print(f"-> Unlabeled Pool     : {len(os.listdir(POOL_IMG_DIR))} gambar")
    print(f"-> Perangkat Pelatihan: {DEVICE.upper()}")


# ==============================================================================
# 3. UNCERTAINTY SAMPLING STRATEGY
# ==============================================================================
def calculate_uncertainty(model, image_path):
    """
    Menghitung skor ketidakpastian (Uncertainty Score) model terhadap gambar.
    Strategi: Least Confidence
    - Jika tidak ada deteksi: Skor = 1.0 (potensi False Negative tinggi)
    - Jika ada deteksi: Skor = 1.0 - max(confidence_score)
    Semakin tinggi skor, semakin ragu model terhadap gambar tersebut.
    """
    results = model.predict(image_path, verbose=False, device=DEVICE)
    boxes = results[0].boxes
    if len(boxes) == 0:
        return 1.0
    max_conf = float(torch.max(boxes.conf).item())
    return 1.0 - max_conf


# ==============================================================================
# 4. ACTIVE LEARNING MAIN LOOP
# ==============================================================================
def run_active_learning_simulation():
    prepare_simulation_environment()
    
    # Inisialisasi model dari bobot nano pre-trained
    model_path = os.path.join(PROJECT_ROOT, "yolov8n.pt")
    if not os.path.exists(model_path):
        model_path = "yolov8n.pt"
    
    model = YOLO(model_path)
    
    # Riwayat metrik evaluasi
    history = []
    
    print("\n" + "="*60)
    print(" [2/4] EVALUASI BASELINE (SIKLUS 0 / SEED SAJA)")
    print("="*60)
    
    # Evaluasi performa awal model pada Validation Set
    val_results = model.val(data=DATASET_YAML_PATH, split="val", verbose=False, device=DEVICE)
    base_map50 = float(val_results.box.map50)
    base_map = float(val_results.box.map)
    train_count = len(os.listdir(TRAIN_IMG_DIR))
    pool_count = len(os.listdir(POOL_IMG_DIR))
    
    history.append({
        "cycle": "Baseline (Seed)",
        "train_samples": train_count,
        "pool_left": pool_count,
        "map50": base_map50,
        "map50_95": base_map
    })
    print(f"Baseline -> Train: {train_count} img | Pool: {pool_count} img | mAP50: {base_map50:.4f} | mAP50-95: {base_map:.4f}")
    
    # Mulai Siklus Active Learning
    best_weights_path = model_path
    
    for cycle in range(1, ITERATIONS + 1):
        print("\n" + "="*60)
        print(f" [3/4] ACTIVE LEARNING SIKLUS {cycle}/{ITERATIONS}")
        print("="*60)
        
        pool_images = sorted([f for f in os.listdir(POOL_IMG_DIR) if f.endswith(('.jpg', '.png'))])
        if len(pool_images) == 0:
            print("Pool data telah habis! Menghentikan siklus.")
            break
            
        # --- Tahap 1: Inferensi & Perhitungan Uncertainty di Pool Data ---
        print(f"-> Melakukan inferensi & scoring pada {len(pool_images)} gambar di pool...")
        scored_images = []
        for img_name in pool_images:
            img_path = os.path.join(POOL_IMG_DIR, img_name)
            score = calculate_uncertainty(model, img_path)
            scored_images.append((img_name, score))
            
        # Urutkan berdasarkan ketidakpastian tertinggi (descending)
        scored_images.sort(key=lambda x: x[1], reverse=True)
        
        # Ambil Top-K sampel sesuai anggaran (budget)
        selected_batch = scored_images[:BUDGET_PER_ITER]
        print(f"-> Memilih {len(selected_batch)} sampel paling meragukan (Uncertainty: {selected_batch[0][1]:.3f} ~ {selected_batch[-1][1]:.3f})")
        
        # --- Tahap 2: Simulasi Human-in-the-Loop (Oracle Vault) ---
        print("-> [Simulated Oracle] Memindahkan gambar & membuka anotasi ground-truth...")
        for img_name, score in selected_batch:
            # Pindahkan gambar dari Pool ke Train
            src_img = os.path.join(POOL_IMG_DIR, img_name)
            dst_img = os.path.join(TRAIN_IMG_DIR, img_name)
            shutil.move(src_img, dst_img)
            
            # Ambil label rahasia dari Oracle Vault ke Train
            lbl_name = os.path.splitext(img_name)[0] + ".txt"
            src_lbl = os.path.join(ORACLE_VAULT_DIR, lbl_name)
            dst_lbl = os.path.join(TRAIN_LBL_DIR, lbl_name)
            if os.path.exists(src_lbl):
                shutil.move(src_lbl, dst_lbl)
                
        new_train_count = len(os.listdir(TRAIN_IMG_DIR))
        new_pool_count = len(os.listdir(POOL_IMG_DIR))
        print(f"-> Data Train kini: {new_train_count} gambar | Sisa Pool: {new_pool_count} gambar")
        
        # --- Tahap 3: Retraining Inkremental ---
        print(f"-> Memulai retraining inkremental ({EPOCHS_PER_ITER} epoch)...")
        train_run = model.train(
            data=DATASET_YAML_PATH,
            epochs=EPOCHS_PER_ITER,
            imgsz=IMGSZ,
            batch=BATCH_SIZE,
            device=DEVICE,
            project=os.path.join(PROJECT_ROOT, "runs", "al_simulation"),
            name=f"cycle_{cycle}",
            exist_ok=True,
            verbose=False
        )
        
        # Checkpoint model terbaik siklus ini
        best_cycle_weight = os.path.join(PROJECT_ROOT, "runs", "al_simulation", f"cycle_{cycle}", "weights", "best.pt")
        if os.path.exists(best_cycle_weight):
            best_weights_path = best_cycle_weight
            model = YOLO(best_weights_path)
            
        # --- Tahap 4: Evaluasi pada Fixed Val Set ---
        val_res = model.val(data=DATASET_YAML_PATH, split="val", verbose=False, device=DEVICE)
        c_map50 = float(val_res.box.map50)
        c_map = float(val_res.box.map)
        
        history.append({
            "cycle": f"Siklus {cycle}",
            "train_samples": new_train_count,
            "pool_left": new_pool_count,
            "map50": c_map50,
            "map50_95": c_map
        })
        print(f"Siklus {cycle} Selesai -> mAP50: {c_map50:.4f} | mAP50-95: {c_map:.4f}")

    # ==============================================================================
    # 5. RINGKASAN HASIL & SIMPAN MODEL
    # ==============================================================================
    print("\n" + "="*60)
    print(" [4/4] RINGKASAN HASIL SIMULASI ACTIVE LEARNING")
    print("="*60)
    
    header = f"| {'Tahap':<18} | {'Train Img':<10} | {'Sisa Pool':<10} | {'mAP@50':<10} | {'mAP@50-95':<10} |"
    divider = "+" + "-"*20 + "+" + "-"*12 + "+" + "-"*12 + "+" + "-"*12 + "+" + "-"*12 + "+"
    print(divider)
    print(header)
    print(divider)
    for h in history:
        print(f"| {h['cycle']:<18} | {h['train_samples']:<10} | {h['pool_left']:<10} | {h['map50']:<10.4f} | {h['map50_95']:<10.4f} |")
    print(divider)
    
    # Simpan bobot final yang mudah diakses
    final_output_model = os.path.join(WORKSPACE_DIR, "best_al_model.pt")
    if os.path.exists(best_weights_path):
        shutil.copy(best_weights_path, final_output_model)
        print(f"\n[OK] Model Active Learning terbaik berhasil disimpan di:")
        print(f"     👉 {final_output_model}")
        print("\nUntuk menguji model ini secara live di webcam, buka simple_yolo.py dan ubah baris 5:")
        print("     model = YOLO('active_learning/best_al_model.pt')")
    print("="*60 + "\n")

if __name__ == "__main__":
    run_active_learning_simulation()