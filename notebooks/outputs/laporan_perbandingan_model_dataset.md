# Perbandingan Model: Random Forest vs LSTM vs XGBoost pada Dataset v1 dan v2

**Project:** Prediksi Polusi PM2.5 di Jakarta Berdasarkan Faktor Cuaca Menggunakan Random Forest Regression (Praktikum 10 Data Mining, IPB Kelompok 10)

**Tujuan dokumen ini:** mendokumentasikan eksperimen perbandingan 3 model pada 2 dataset, beserta justifikasi mengapa dataset v2 dibuat di luar scope proposal.

---

## 1. Kenapa Dataset v2 Dibuat?

Audit terhadap `dataset_final_model.csv` (v1, sesuai proposal Bab 5) menemukan **3 masalah** yang menahan R² overall di ~0.67:

### 1.1 Setengah fitur cuaca prediktif tidak dipakai
Hasil korelasi Pearson pada data cuaca mentah vs ISPU PM2.5:

| Fitur | |r| | Dipakai di v1 (proposal)? |
|---|---|---|
| temp | 0.313 | ✅ |
| **cloudcover** | **0.298** | ❌ |
| **tempmax** | **0.298** | ❌ |
| **winddir** | **0.296** | ❌ |
| solarenergy | 0.272 | ✅ |
| humidity | 0.271 | ✅ |
| **feelslike** | **0.261** | ❌ |
| windgust | 0.258 | ✅ |
| **uvindex** | **0.232** | ❌ |
| visibility | 0.192 | ✅ |
| **precipprob** | **0.184** | ❌ |
| precip | 0.155 | ✅ |

Proposal Bab 4.6 hanya memilih 6 fitur dengan justifikasi literatur. Lima fitur cuaca dengan |r| ≥ 0.18 (cloudcover, tempmax, winddir, feelslike, uvindex, precipprob) yang juga punya basis fisika atmosfer **tidak ikut** karena bukan bagian dari seleksi proposal.

### 1.2 24% target adalah hasil imputasi, bukan observasi nyata
Polusi mentah berformat jam-an. Dari 8.776 kombinasi (hari × stasiun), **2.094 (24%) tidak punya observasi jam-an sama sekali**. Setelah aggregate ke daily mean → NaN → diisi oleh `ffill`/`bfill`/median (proses preprocessing v1 Bab 3.3). Akibatnya:
- Model "belajar" dari data palsu yang dibuat oleh imputasi.
- Imputasi pada dasarnya menyalin nilai tetangga → easy untuk diprediksi → menaikkan R² secara artifisial.

### 1.3 Sensor error PM2.5 = 0
Empat baris di stasiun kebun jeruk awal 2023 punya PM2.5 = 0. Dalam lingkungan urban Jakarta, secara fisik tidak masuk akal. Sensor mati, harusnya `NaN`, bukan 0.

### 1.4 Perbaikan di dataset v2

| Aspek | v1 (proposal) | v2 (perbaikan, OOS) |
|---|---|---|
| Jumlah baris | 7.679 | 5.609 |
| Fitur cuaca | 6 (sesuai proposal) | 16 (semua yang relevan) |
| Encoding winddir | n/a | sin/cos (variabel circular) |
| Baris dari imputasi | ~24% | 0% (semua observasi nyata, n_obs ≥ 12) |
| Sensor error PM2.5 = 0 | 4 baris dibiarkan | Dibuang |
| Statistik intraday | n/a | `pm25_std`, `pm25_max` per hari |

**⚠️ Status akademik:** dataset v2 **DI LUAR scope proposal** karena menambahkan fitur cuaca baru yang tidak tercantum di Bab 4.6 dengan sitasi literaturnya. v2 ada sebagai *eksperimen dokumentasi pengetahuan*, BUKAN untuk laporan utama. Laporan tetap pakai v1 + RF Regression sesuai komitmen proposal.

---

## 2. Matriks Perbandingan 3 Model × 2 Dataset

Semua model dipilih berdasarkan **`val_RMSE` terkecil**, evaluasi di test sekali pada akhir. Time split 70/15/15 berbasis tanggal unik.

| Model | Dataset | Best Config | val_RMSE | test_RMSE | test_R² | test_MAE | test_MAPE |
|---|---|---|---|---|---|---|---|
| **RF** | **v1 (6 cuaca)** | Random Forest Grid Search | 11.33 | 12.29 | **0.673** | 8.78 | 17.80 |
| LSTM | v1 (6 cuaca) | LSTM A_LSTM64 L30 | 11.61 | 13.68 | 0.602 | 9.95 | 19.71 |
| XGBoost | v1 (6 cuaca) | XGBoost Grid Search | 8.66 | 12.40 | 0.668 | 8.89 | 18.70 |
| RF | v2 (16 cuaca, OOS) | RF Grid Search (v2) | 8.85 | 14.30 | 0.639 | 10.42 | 25.47 |
| LSTM | v2 (16 cuaca, OOS) | LSTM A_LSTM64 L14 (v2) | 14.33 | 16.03 | 0.550 | 11.83 | 26.73 |
| XGBoost | v2 (16 cuaca, OOS) | XGBoost v3 Grid Search | 8.41 | 14.10 | 0.656 | 10.03 | 25.61 |

📄 File CSV: [matriks_perbandingan_3model_x_2dataset.csv](matriks_perbandingan_3model_x_2dataset.csv) | Tabel lengkap 27 model: [perbandingan_lengkap_v3.csv](perbandingan_lengkap_v3.csv)

---

## 3. Interpretasi (Jujur)

### 3.1 Di v1 (proposal scope)
- **RF Grid Search adalah pemenang `test_R²` (0.673)**, sesuai metode utama proposal. Selisih dengan XGBoost (0.668) hanya ~0.5%, jadi RF tetap kompetitif.
- XGBoost menang `val_RMSE` (8.66 vs RF 11.33), tapi selisihnya menyempit di test set → RF tidak overfit, XGBoost agak overfit ke validasi.
- LSTM kalah pada semua metrik regresi (R² 0.602 vs RF 0.673), wajar karena dataset relatif kecil (7.679 baris, time horizon hanya 3 tahun) dan signal weather harian → PM2.5 lebih cocok untuk model tree-based.

### 3.2 Di v2 (out-of-scope)
- Pada v2, **XGBoost (0.656) tipis menang** atas RF (0.639) dan LSTM (0.550).
- Tapi **R² test SEMUA model di v2 lebih rendah dari di v1**. Apakah artinya v2 lebih buruk?
- **Tidak.** Test set di v1 dan v2 tidak sama: v2 drop 24% baris hasil imputasi + sensor error. Test v1 punya rows yang "mudah ditebak" karena diimputasi dari tetangga → R² terlihat lebih tinggi. Test v2 hanya berisi observasi nyata → lebih sulit.

### 3.3 Pola yang konsisten lintas model
- **Gap besar val_RMSE → test_RMSE di v2** (RF: 8.85 → 14.30, XGBoost: 8.41 → 14.10) mengindikasikan **distribution shift** antara periode validasi (Sep-Nov 2024) dan periode test (Nov 2024-Des 2024). Period test punya kejutan PM2.5 yang weather + history tidak bisa jelaskan.
- LSTM v2 jauh lebih lemah daripada LSTM v1 (R² 0.550 vs 0.602) — kemungkinan karena training set v2 lebih kecil (3.834 sequences vs ~5.300 di v1), LSTM butuh banyak data.

### 3.4 Kesimpulan Utama
**R² test mentok di ~0.65-0.68 untuk semua model di semua dataset.** Ini bukan kebetulan — ini batas teoretis kemampuan memprediksi daily mean PM2.5 dari cuaca + history saja. Mengganti model (RF → LSTM → XGBoost) hanya menggerakkan R² ±0.05. Yang akan break through batas ini bukan tuning model, tapi:
1. **Data per-jam** (bukan daily aggregate)
2. **Fitur emisi** (kebakaran lahan NASA FIRMS, traffic data)
3. **Fitur dinamika atmosfer** (PBL height dari ERA5, AOD satelit)

---

## 4. Rekomendasi untuk Laporan Akademik

**Gunakan dataset v1 + Random Forest Regression sebagai hasil utama** (sesuai proposal):
- Model: RF Grid Search
- Hasil: val_RMSE 11.33, test_RMSE 12.29, **test_R² 0.673**, test_MAE 8.78
- Lokasi: [02_Modelling.ipynb](../02_Modelling.ipynb), [model_pm25_best_final.pkl](model_pm25_best_final.pkl)

**Pemodelan tambahan untuk diskusi** (in-scope dengan dataset v1):
- LSTM sebagai pembanding deep learning → test_R² 0.602
- XGBoost sebagai pembanding gradient boosting → test_R² 0.668
- Diskusi: ketiga model konvergen di sekitar R² 0.60-0.67, mengonfirmasi bahwa keterbatasan bukan pada model.

**Eksperimen di luar scope proposal** (dataset v2, untuk diskusi limitasi/future work):
- Hanya disinggung di bagian "Limitasi & Saran Penelitian Lanjutan".
- Tunjukkan bahwa preprocessing yang lebih ketat (drop 24% imputasi) ternyata **menurunkan R² test** ke ~0.64-0.66, karena ke-mudah-an v1 sebagian datang dari data hasil imputasi.
- Saran future work: penelitian lanjutan dengan data jam-an + fitur eksternal (FIRMS, ERA5) diperlukan untuk lompat ke R² > 0.75.

---

## 5. File yang Relevan

### Dalam scope proposal (untuk laporan):
- [dataset_final_model.csv](../../data/final%20for%20modelling/dataset_final_model.csv) - dataset v1
- [02_Modelling.ipynb](../02_Modelling.ipynb) - RF (model utama)
- [02_Modelling_LSTM.ipynb](../02_Modelling_LSTM.ipynb) - LSTM (pembanding)
- [02_Modelling_XGBoost.ipynb](../02_Modelling_XGBoost.ipynb) - XGBoost (pembanding)
- [model_pm25_best_final.pkl](model_pm25_best_final.pkl) + [model_pm25_metadata_final.json](model_pm25_metadata_final.json)

### Di luar scope (dokumentasi tambahan):
- [dataset_final_model_v2.csv](../../data/final%20for%20modelling/dataset_final_model_v2.csv) - dataset v2
- [01_EDA_Preprocess_v2.ipynb](../01_EDA_Preprocess_v2.ipynb) - preprocessing v2
- [model_xgboost_perbaikan_v3.pkl](model_xgboost_perbaikan_v3.pkl) + [metadata_xgboost_v3.json](metadata_xgboost_v3.json)
- [perbandingan_lengkap_v3.csv](perbandingan_lengkap_v3.csv) - tabel lengkap 27 model
- [matriks_perbandingan_3model_x_2dataset.csv](matriks_perbandingan_3model_x_2dataset.csv) - matriks 3×2
