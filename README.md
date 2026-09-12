# Proyek Akhir: Menyelesaikan Permasalahan Institusi Pendidikan

## Jaya Jaya Institut — Student Dropout Prediction

---

## Business Understanding

### Latar Belakang

Jaya Jaya Institut adalah institusi pendidikan perguruan tinggi yang berdiri sejak tahun 2000. Meski telah mencetak banyak lulusan dengan reputasi baik, institusi ini menghadapi masalah serius: **tingginya angka dropout mahasiswa** (~32% dari total mahasiswa). Angka ini berdampak negatif terhadap reputasi, pendapatan, dan kualitas pendidikan secara keseluruhan.

### Permasalahan Bisnis

1. Seberapa besar tingkat dropout mahasiswa di Jaya Jaya Institut?
2. Faktor-faktor apa saja yang paling berpengaruh terhadap keputusan mahasiswa untuk dropout?
3. Bagaimana cara memprediksi sedini mungkin mahasiswa yang berpotensi dropout sehingga dapat diberi bimbingan khusus?

### Cakupan Proyek

- Analisis eksploratif data (EDA) untuk memahami distribusi dan pola dropout
- Pembangunan model machine learning untuk memprediksi risiko dropout (Dropout vs Graduate)
- Pembuatan dashboard monitoring performa mahasiswa menggunakan Metabase
- Deployment prototype sistem prediksi berbasis Streamlit

---

## Persiapan

### Sumber Data

Dataset Students' Performance dari Dicoding Academy:  
[https://github.com/dicodingacademy/dicoding_dataset/blob/main/students_performance/README.md](https://github.com/dicodingacademy/dicoding_dataset/blob/main/students_performance/README.md)

- **Jumlah data:** 4,424 mahasiswa
- **Fitur:** 36 kolom (demografis, akademik, ekonomi makro)
- **Target:** Status mahasiswa (Dropout / Graduate / Enrolled)
- **Missing values:** Tidak ada

### Setup Environment

#### Setup Environment - Anaconda

```bash
conda create --name main-ds python=3.9
conda activate main-ds
pip install -r requirements.txt
```

#### Setup Environment - Shell/Terminal (pipenv)

```bash
pip install pipenv
pipenv install
pipenv shell
```

#### Menjalankan Notebook

```bash
jupyter notebook notebook.ipynb
```

---

## Business Dashboard

Dashboard dibuat menggunakan **Metabase** untuk memvisualisasikan dan memonitor performa mahasiswa Jaya Jaya Institut.

### Visualisasi pada Dashboard

1. **Distribusi Status Mahasiswa** — proporsi Dropout, Graduate, dan Enrolled
2. **Dropout Rate per Program Studi** — identifikasi prodi dengan tingkat dropout tertinggi
3. **Pengaruh Status SPP** — korelasi pembayaran SPP dengan risiko dropout
4. **Analisis Demografis** — distribusi usia, gender, dan beasiswa terhadap status
5. **Performa Akademik** — nilai dan unit semester vs status mahasiswa

### Menjalankan Dashboard Metabase

```bash
# Jalankan container Metabase
docker run -d -p 3000:3000 \
  -v "$(pwd)/metabase-data:/metabase.db" \
  --name metabase metabase/metabase

# Tunggu ~2 menit, lalu akses di browser:
# http://localhost:3000
```

Untuk me-restore dashboard yang sudah dibuat, copy file database ke container:

```bash
docker cp metabase.db.mv.db metabase:/metabase.db/metabase.db.mv.db
docker restart metabase
```

### Credential Metabase

| Field        | Value                   |
| ------------ | ----------------------- |
| **Email**    | `root@mail.com`         |
| **Password** | `root123`               |
| **URL**      | `http://localhost:3000` |

Screenshot dashboard tersedia di folder `pratama_dicoding-dashboard/`.

---

## Menjalankan Prototype Machine Learning

Prototype sistem prediksi dropout dibuat menggunakan Streamlit.

### Cara Menjalankan Lokal

```bash
# Dari folder submission/
# Pastikan environment sudah aktif

# Jika menggunakan conda:
conda activate main-ds
streamlit run app.py

# Jika menggunakan pipenv:
pipenv shell
streamlit run app.py
```

Aplikasi akan terbuka di browser: `http://localhost:8501`

### Link Streamlit Community Cloud

> **[https://jaya-jaya-institut-dropout-oeqbgsu2rukf4kjwvtc3ff.streamlit.app](https://jaya-jaya-institut-dropout-oeqbgsu2rukf4kjwvtc3ff.streamlit.app)**

### Link GitHub Repository

> **[https://github.com/josapratama/jaya-jaya-institut-dropout](https://github.com/josapratama/jaya-jaya-institut-dropout)**

### Fitur Prototype

- **🏠 Dashboard Overview** — KPI utama dan visualisasi ringkasan data
- **🔍 Prediksi Dropout** — Form input data mahasiswa + hasil prediksi probabilitas
- **📊 Analisis Data** — EDA interaktif, statistik per status, feature importance
- **ℹ️ Tentang Sistem** — Dokumentasi model dan action items

---

## Conclusion

Berdasarkan analisis data mahasiswa Jaya Jaya Institut:

### Catatan Penting tentang Data

Model machine learning **hanya dilatih menggunakan data mahasiswa berstatus Dropout dan Graduate**. Mahasiswa berstatus Enrolled **tidak diikutsertakan** dalam proses training karena outcome mereka belum diketahui — mereka masih aktif kuliah. Data Enrolled disimpan terpisah di `model/enrolled_for_prediction.csv` untuk digunakan sebagai data prediksi di masa depan.

| Kelompok Data | Jumlah | Keterangan                                               |
| ------------- | ------ | -------------------------------------------------------- |
| Graduate      | 2,209  | Digunakan untuk training (target = 0)                    |
| Dropout       | 1,421  | Digunakan untuk training (target = 1)                    |
| Enrolled      | 794    | **Tidak digunakan** — disimpan untuk prediksi masa depan |

### Temuan Utama

| Temuan                  | Detail                                                      |
| ----------------------- | ----------------------------------------------------------- |
| Dropout rate (DO vs GR) | **39.2%** dari subset Dropout+Graduate                      |
| Faktor terkuat          | Nilai & unit disetujui semester 1 dan 2                     |
| Faktor finansial        | Mahasiswa dengan SPP menunggak dropout rate 3x lebih tinggi |
| Faktor beasiswa         | Penerima beasiswa memiliki dropout rate jauh lebih rendah   |
| Program studi           | Beberapa prodi memiliki dropout rate > 40%                  |

### Performa Model (Random Forest — Dropout vs Graduate)

| Metrik        | Nilai                             |
| ------------- | --------------------------------- |
| Algoritma     | Random Forest Classifier          |
| Data training | Dropout + Graduate (3,630 sampel) |
| Target        | 1 = Dropout, 0 = Graduate         |
| Akurasi       | **92.56%**                        |
| F1 Score      | **0.9043**                        |
| ROC-AUC       | **0.9716**                        |

---

## Rekomendasi Action Items

### 1. 🚨 Implementasi Early Warning System

Gunakan model ini setiap awal semester untuk memprediksi seluruh mahasiswa aktif (Enrolled). Mahasiswa dengan probabilitas dropout > 60% segera mendapatkan intervensi.

### 2. 📚 Program Bimbingan Akademik Intensif

Mahasiswa dengan nilai semester 1 di bawah 10/20 atau gagal lebih dari 50% mata kuliah harus segera mendapat bimbingan intensif dari dosen wali. Intervensi dini di semester pertama sangat krusial.

### 3. 💰 Bantuan Finansial Proaktif

Identifikasi mahasiswa dengan tunggakan SPP sebelum semester baru. Tawarkan skema cicilan fleksibel atau rekomendasikan program beasiswa. Data menunjukkan mahasiswa dengan SPP tidak lunas memiliki risiko dropout 3x lebih tinggi.

### 4. 🎓 Evaluasi Program Studi Bermasalah

Program studi dengan dropout rate > 35% memerlukan evaluasi menyeluruh: review beban kurikulum, metode pengajaran, dan program pendampingan khusus.

### 5. 📈 Monitoring Dashboard Rutin

Jadikan dashboard Metabase sebagai alat monitoring wajib dalam rapat evaluasi bulanan. Pantau tren dropout per semester, per program studi, dan per kelompok demografis.

### 6. 🤝 Program Orientasi Khusus Mahasiswa Rentan

Mahasiswa berusia > 25 tahun saat enrollment, mahasiswa pindahan (displaced), dan mahasiswa tanpa beasiswa dengan kondisi finansial terbatas perlu program orientasi dan konseling khusus.

---

## Struktur Direktori

```
submission/
├── model/
│   ├── dropout_model.joblib         # Model Random Forest terlatih
│   ├── features.json                # Daftar fitur yang digunakan
│   └── enrolled_for_prediction.csv  # Data Enrolled untuk prediksi masa depan
├── pratama_dicoding-dashboard/      # Screenshot visualisasi dashboard
│   ├── target_distribution.png
│   ├── demographic_analysis.png
│   ├── academic_performance.png
│   ├── economic_factors.png
│   ├── correlation_heatmap.png
│   ├── dropout_by_course.png
│   ├── model_comparison.png
│   └── model_evaluation.png
├── notebook.ipynb                   # Jupyter Notebook analisis lengkap
├── app.py                           # Streamlit prototype
├── data.csv                         # Dataset Students' Performance
├── metabase.db.mv.db                # Database Metabase (dashboard)
├── requirements.txt                 # Daftar library
├── Pipfile                          # Pipenv dependencies
├── packages.txt                     # System packages (Streamlit Cloud)
└── README.md                        # Dokumentasi proyek (file ini)
```

---

_Proyek ini dikerjakan sebagai submission akhir kelas Belajar Penerapan Data Science — Dicoding Academy._
