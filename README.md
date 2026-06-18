# 📊 Sales Prediction Dashboard

Aplikasi web untuk memprediksi penjualan toko menggunakan model **XGBoost Regression**, dibangun dengan **Streamlit**.

🔗 **Live Demo:** https://sales-prediction-app-bsqrz7yumyezg7waaux67l.streamlit.app/

---

## 📌 Tentang Project

Project ini memprediksi nilai *Sales* sebuah toko berdasarkan kondisi operasional harian (hari, promo, libur, dll) menggunakan model XGBoost yang dilatih dari data historis penjualan toko.

### Performa Model

| Metrik | Nilai |
|---|---|
| R² | 0.80 |
| MAE | 1159.25 |
| RMSE | 1717.23	 |
| MAPE | 21.00% |

---

## 🗂️ Struktur Project

```
sales_app/
├── app.py              # Aplikasi Streamlit
├── model_xgb.pkl        # Model XGBoost terlatih (joblib)
├── requirements.txt     # Daftar dependency
└── README.md
```

---

## ⚙️ Fitur Input

| Fitur | Keterangan |
|---|---|
| Store | ID toko (1–1115) |
| Day Of Week | Hari (Senin–Minggu) |
| Promo | Status promo aktif |
| State Holiday | Jenis hari libur nasional |
| School Holiday | Status libur sekolah |
| Month | Bulan (1–12) |
| Is Weekend | Otomatis dari Day Of Week |
| Is Payday Period | Periode gajian (manual) |
| Store Avg Customer | Rata-rata jumlah customer historis |

> `day_of_month` dan `week_of_year` diisi otomatis berdasarkan tanggal saat aplikasi dijalankan.

---

## 🚀 Cara Menjalankan Lokal

```bash
# 1. Clone repo
git clone https://github.com/Rangga-a/sales-prediction-app.git
cd sales-prediction-app

# 2. Buat virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

# 3. Install dependency
pip install -r requirements.txt

# 4. Jalankan aplikasi
streamlit run app.py
```

Aplikasi akan terbuka otomatis di `http://localhost:8501`.

---

## 🌐 Deploy

Aplikasi ini dideploy menggunakan **Streamlit Community Cloud**. Setiap `git push` ke branch `main` akan otomatis memicu redeploy.

---

## 🛠️ Tech Stack

- **Python**
- **XGBoost** — model regresi
- **Streamlit** — web app framework
- **Plotly** — visualisasi gauge chart
- **Pandas, Joblib** — pengolahan data & model serialization

---

## ⚠️ Catatan

- Model dilatih menggunakan data historis penjualan ritel.
- Beberapa fitur (`store_avg_customer`) menggunakan nilai input manual sebagai pendekatan, karena data historis real-time per toko tidak tersedia di aplikasi.
- Project ini dibuat untuk tujuan pembelajaran/akademik.
