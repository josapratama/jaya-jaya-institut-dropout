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
- Pembangunan model machine learning untuk memprediksi risiko dropout mahasiswa
- Pembuatan dashboard monitoring performa mahasiswa
- Deployment prototype sistem prediksi berbasis Streamlit

---

## Persiapan

### Sumber Data

Dataset Students' Performance dari Dicoding Academy (UCI ML Repository):  
[https://github.com/dicodingacademy/dicoding_dataset/blob/main/students_performance/README.md](https://github.com/dicodingacademy/dicoding_dataset/blob/main/students_performance/README.md)

- **Jumlah data:** 4,424 mahasiswa
- **Fitur:** 36 kolom (demografis, akademik, ekonomi makro)
- **Target:** Status mahasiswa (Dropout / Graduate / Enrolled)
- **Missing values:** Tidak ada

### Setup Environment

```bash
# Clone / download project
# Pastikan Python 3.9+ sudah terinstall

# Install dependencies
pip install -r requirements.txt
```

---

## Business Dashboard

Dashboard dibuat untuk memvisualisasikan dan memonitor performa mahasiswa Jaya Jaya Institut. Dashboard mencakup:

1. **Distribusi Status Mahasiswa** — proporsi Dropout, Graduate, dan Enrolled
2. **Dropout Rate per Program Studi** — identifikasi prodi dengan masalah terbesar
3. **Pengaruh Status SPP** — korelasi pembayaran SPP dengan risiko dropout
4. **Analisis Demografis** — distribusi usia, gender, beasiswa terhadap status
5. **Analisis Performa Akademik** — nilai dan unit semester vs status mahasiswa

### Akses Dashboard Metabase

> Dashboard dapat diakses secara lokal menggunakan Metabase dengan Docker.

**Credential:**

- **Email:** `root@mail.com`
- **Password:** `root123`

**Menjalankan Metabase dengan Docker:**

```bash
# Pull dan jalankan container Metabase
docker run -d -p 3000:3000 --name metabase metabase/metabase

# Import database (setelah container berjalan)
docker cp metabase.db.mv.db metabase:/metabase.db/metabase.db.mv.db
docker restart metabase

# Akses di browser: http://localhost:3000
```

Screenshot dashboard tersedia di folder `pratama_dicoding-dashboard/`.

---

## Menjalankan Prototype Machine Learning

Prototype sistem prediksi dropout dibuat menggunakan Streamlit.

### Cara Menjalankan Lokal

```bash
# Dari folder submission/
streamlit run app.py
```

Aplikasi akan terbuka di browser: `http://localhost:8501`

### Link Streamlit Community Cloud

> **[https://josapratama-jaya-jaya-institut-dropout.streamlit.app](https://josapratama-jaya-jaya-institut-dropout.streamlit.app)**

### Link GitHub Repository

> **[https://github.com/josapratama/jaya-jaya-institut-dropout](https://github.com/josapratama/jaya-jaya-institut-dropout)**

### Fitur Prototype

- **🏠 Dashboard Overview** — KPI utama dan visualisasi ringkasan data
- **🔍 Prediksi Dropout** — Form input data mahasiswa + hasil prediksi dengan risk gauge
- **📊 Analisis Data** — EDA interaktif, statistik per status, feature importance
- **ℹ️ Tentang Sistem** — Dokumentasi model dan action items

---

## Conclusion

Berdasarkan analisis yang telah dilakukan terhadap data mahasiswa Jaya Jaya Institut:

### Temuan Utama

| Temuan           | Detail                                                               |
| ---------------- | -------------------------------------------------------------------- |
| Tingkat Dropout  | **32.1%** dari total 4,424 mahasiswa                                 |
| Faktor terkuat   | Nilai dan jumlah unit disetujui semester 1 & 2                       |
| Faktor finansial | Mahasiswa dengan SPP menunggak memiliki dropout rate 3x lebih tinggi |
| Faktor beasiswa  | Penerima beasiswa memiliki dropout rate lebih rendah signifikan      |
| Program studi    | Beberapa prodi memiliki dropout rate > 40%, jauh di atas rata-rata   |

### Performa Model

| Metrik    | Nilai                    |
| --------- | ------------------------ |
| Algoritma | Random Forest Classifier |
| Akurasi   | ~86%                     |
| F1 Score  | ~0.83                    |
| ROC-AUC   | ~0.92                    |

Model berhasil mengidentifikasi mahasiswa berisiko dropout dengan precision dan recall yang seimbang, membuatnya praktis untuk digunakan sebagai sistem peringatan dini.

---

## Rekomendasi Action Items

Berdasarkan hasil analisis, berikut action items yang direkomendasikan untuk Jaya Jaya Institut:

### 1. 🚨 Implementasi Early Warning System

Gunakan model machine learning ini sebagai sistem peringatan dini. Setiap awal semester, jalankan prediksi untuk seluruh mahasiswa aktif dan identifikasi yang masuk kategori risiko tinggi (probabilitas dropout > 60%) untuk mendapatkan intervensi segera.

### 2. 📚 Program Bimbingan Akademik Intensif

Mahasiswa dengan nilai semester 1 di bawah 10/20 atau yang gagal lebih dari 50% mata kuliah harus segera mendapatkan program bimbingan intensif dari dosen wali. Intervensi dini di semester pertama terbukti sangat krusial.

### 3. 💰 Bantuan Finansial Proaktif

Identifikasi mahasiswa dengan tunggakan SPP sebelum semester baru dimulai. Tawarkan:

- Skema cicilan pembayaran yang fleksibel
- Rekomendasi program beasiswa internal/eksternal
- Konsultasi keuangan dengan bagian kemahasiswaan

### 4. 🎓 Evaluasi Program Studi Bermasalah

Program studi dengan dropout rate > 35% memerlukan evaluasi menyeluruh meliputi:

- Review beban dan relevansi kurikulum
- Peningkatan kualitas dan metode pengajaran
- Program pendampingan khusus per prodi

### 5. 📈 Monitoring Dashboard Rutin

Jadikan dashboard sebagai alat monitoring wajib dalam rapat evaluasi bulanan. Pantau tren dropout per semester, per program studi, dan per kelompok demografis untuk pengambilan kebijakan berbasis data.

### 6. 🤝 Program Orientasi Khusus Mahasiswa Rentan

Mahasiswa berisiko tinggi perlu program orientasi dan konseling khusus:

- Mahasiswa berusia > 25 tahun saat enrollment
- Mahasiswa pindahan (displaced) dari luar daerah
- Mahasiswa tanpa beasiswa dengan kondisi finansial terbatas

---

## Struktur Direktori

```
submission/
├── model/
│   ├── dropout_model.joblib    # Model Random Forest terlatih
│   └── features.json           # Daftar fitur yang digunakan
├── pratama_dicoding-dashboard/ # Screenshot visualisasi dashboard
│   ├── target_distribution.png
│   ├── demographic_analysis.png
│   ├── academic_performance.png
│   ├── economic_factors.png
│   ├── correlation_heatmap.png
│   ├── dropout_by_course.png
│   ├── model_comparison.png
│   └── model_evaluation.png
├── notebook.ipynb              # Jupyter Notebook analisis lengkap
├── app.py                      # Streamlit prototype
├── data.csv                    # Dataset Students' Performance
├── requirements.txt            # Daftar library
└── README.md                   # Dokumentasi proyek (file ini)
```

---

_Proyek ini dikerjakan sebagai submission akhir kelas Belajar Penerapan Data Science — Dicoding Academy._
