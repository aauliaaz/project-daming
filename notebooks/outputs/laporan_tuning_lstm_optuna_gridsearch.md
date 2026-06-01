# Perbandingan Tuning LSTM: Optuna TPE vs Grid Search

- Lookback dikunci pada `7` hari, mengikuti pemenang eksperimen manual sebelumnya.
- Pemilihan model memakai `val_RMSE` terkecil. Test set hanya dilaporkan setelah pemenang tiap metode dipilih.
- Grid Search mencoba seluruh kombinasi diskret. Optuna TPE memilih konfigurasi berikutnya secara adaptif.

## Hasil Utama

| metode | n_evaluasi | ruang_pencarian | waktu_total_detik | rata2_waktu_per_trial | best_units | best_dropout | best_learning_rate | best_epoch | best_val_MAE | best_val_RMSE | best_val_R2 | best_test_MAE | best_test_RMSE | best_test_R2 | best_test_MAPE | recall_TidakSehat_test | jumlah_prediksi_PM25_gt_100 | keterangan |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Optuna TPE | 16 | units={32,64,96,128}; dropout=[0.1,0.4]; learning_rate=[3e-4,2e-3] log | 165.8 | 10.4 | 128 | 0.38969 | 0.00139049 | 10 | 9.0844 | 12.0085 | 0.5735 | 9.4201 | 13.1996 | 0.6237 | 18.5104 | 0.137 | 29 | <== PEMENANG by val_RMSE |
| Grid Search | 12 | units={32,64,128}; dropout={0.1,0.3}; learning_rate={1e-3,5e-4} | 113.8 | 9.5 | 32 | 0.3 | 0.001 | 27 | 9.2372 | 12.1239 | 0.5653 | 9.7491 | 13.5011 | 0.6063 | 19.4214 | 0.1644 | 36 |  |

Pemenang tuning formal: **Optuna TPE** dengan validation RMSE **12.0085**.

## Dibanding Eksperimen Manual

| sumber | model | val_RMSE | test_MAE | test_RMSE | test_R2 | recall_TidakSehat_test | keterangan |
|---|---|---|---|---|---|---|---|
| Optuna TPE | LSTM units=128, dropout=0.38969, lr=0.00139049 | 12.0085 | 9.4201 | 13.1996 | 0.6237 | 0.137 | <== TERBAIK by val_RMSE |
| Grid Search | LSTM units=32, dropout=0.3, lr=0.001 | 12.1239 | 9.7491 | 13.5011 | 0.6063 | 0.1644 |  |
| Eksperimen manual sebelumnya | LSTM Perbaikan B_LSTM128_Dropout0.3_Dense64 (L7) | 12.1889 | 9.3741 | 13.1544 | 0.6263 | 0.0822 |  |

## Final Retrain

Konfigurasi pemenang dilatih ulang pada gabungan train + validation. Test tidak dipakai untuk training.

- Test MAE: `8.9954`
- Test RMSE: `12.7177`
- Test R2: `0.6507`
- Test MAPE: `17.9112%`

## Perbedaan Metode

| Aspek | Grid Search | Optuna TPE |
|---|---|---|
| Strategi | Mencoba semua kombinasi pada grid | Sampling adaptif, makin fokus pada area menjanjikan |
| Ruang nilai | Diskret dan mudah diaudit | Dapat memakai rentang kontinu |
| Biaya komputasi | Cepat membesar saat parameter bertambah | Dibatasi langsung oleh jumlah trial |
| Kelebihan proyek ini | Mudah dijelaskan saat presentasi | Lebih fleksibel untuk learning rate dan dropout |
