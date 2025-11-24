# 🧠 Brain Tumor Classification using CNN, ResNet50, and Vision Transformer  
_Praktikum Sains Data – Semester 1_

Proyek ini adalah implementasi Deep Learning untuk klasifikasi tumor otak menggunakan dataset MRI. Terdapat empat kelas utama:

1. **Glioma Tumor**
2. **Meningioma Tumor**
3. **Pituitary Tumor**
4. **No Tumor**

Model yang diimplementasikan:
- ⭐ **CNN Basic**
- 🚀 **CNN Advanced**
- 🏋️ **ResNet50 Transfer Learning**
- 🔭 **Vision Transformer (ViT)**

---

## 📂 Project Structure

```
📦 SainsDataPrak6
│
├── 01_cnn/
│   ├── cnn_basic_brain_tumor.py
│   ├── cnn_advanced.py
│   ├── resnet50_brain_tumor.py
│   ├── vit_brain_tumor.py
│   ├── utils/
│   └── (helper scripts lainnya)
│
├── saves_enhanced/
│   ├── training_CNN_Enhanced.csv
│   ├── training_ResNet50_Enhanced.csv
│   ├── training_ViT_Enhanced.csv
│   └── logs_*/ (TensorBoard logs)
│
├── brain_tumor_dataset/ (ignored)
├── README.md
└── .gitignore
```

> File `.h5` dan folder model **tidak** disimpan di GitHub karena ukurannya sangat besar (>100MB).

---

## ⚡ GPU Setup (RTX 4060 — TensorFlow 2.10)

Gunakan Python **3.10** dan TensorFlow **2.10 GPU** (versi terakhir yang mendukung GPU di Windows).

### 1️⃣ Buat Virtual Environment

```powershell
py -3.10 -m venv tf-gpu
```

### 2️⃣ Aktifkan (PowerShell)

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\tf-gpu\Scripts\Activate.ps1
```

### 3️⃣ Install Dependencies

```powershell
pip install tensorflow==2.10
pip install numpy==1.23.5
pip install matplotlib seaborn scikit-learn scipy opencv-python
```

---

## 🧪 Cara Menjalankan Model

### ▶ Jalankan CNN Basic
```powershell
python 01_cnn/cnn_basic_brain_tumor.py
```

### ▶ Jalankan CNN Advanced
```powershell
python 01_cnn/cnn_advanced.py
```

### ▶ Jalankan ResNet50
```powershell
python 01_cnn/resnet50_brain_tumor.py
```

### ▶ Jalankan Vision Transformer
```powershell
python 01_cnn/vit_brain_tumor.py
```

---

## 🧵 Verifikasi GPU

```powershell
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Output yang benar:

```
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

---

## 📊 Output Model

Setiap training menghasilkan:

- Grafik **Accuracy**
- Grafik **Loss**
- **Confusion Matrix**
- **Classification Report**
- CSV training log
- TensorBoard logs

---

## 📘 Dataset

Struktur dataset:

```
brain_tumor_dataset/
   ├── Training/
   │    ├── glioma_tumor/
   │    ├── meningioma_tumor/
   │    ├── pituitary_tumor/
   │    └── no_tumor/
   └── Testing/
```

Dataset dapat diunduh dari:

🔗 https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

---

## ⭐ Performa Model (Ringkasan)

| Model              | Akurasi  | Catatan                        |
|-------------------|----------|--------------------------------|
| CNN Basic         | 85–90%   | ringan dan cepat               |
| CNN Advanced      | 92–95%   | deep network, lebih stabil     |
| ResNet50          | 95–97%   | performa terbaik               |
| Vision Transformer| 94–97%   | efektif pada GPU               |

---

## 👤 Author

**Abd Rahman Dzaky – Magister Sains Data**  
GitHub: https://github.com/serbetwarung

---

## ⭐ Support

Jika repo ini membantu, jangan lupa kasih **⭐ Star** ya!
