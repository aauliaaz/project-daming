# Perbandingan Tuning LSTM: Optuna TPE vs Grid Search

- Lookback dikunci pada `7` hari, mengikuti pemenang eksperimen manual sebelumnya.
- Pemilihan model memakai `val_RMSE` terkecil. Test set hanya dilaporkan setelah pemenang tiap metode dipilih.
- Grid Search mencoba seluruh kombinasi diskret. Optuna TPE memilih konfigurasi berikutnya secara adaptif.

## Hasil Utama

| metode | n_evaluasi | ruang_pencarian | waktu_total_detik | rata2_waktu_per_trial | best_units | best_dropout | best_learning_rate | best_epoch | best_val_MAE | best_val_RMSE | best_val_R2 | best_test_MAE | best_test_RMSE | best_test_R2 | best_test_MAPE | recall_TidakSehat_test | jumlah_prediksi_PM25_gt_100 | keterangan |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Optuna TPE | 16 | units={32,64,96,128}; dropout=[0.1,0.4]; learning_rate=[3e-4,2e-3] log | 183.7 | 11.5 | 32 | 0.317454 | 0.00112594 | 22 | 9.0046 | 12.1812 | 0.5303 | 10.7002 | 15.214 | 0.4889 | 20.2128 | 0.2368 | 54 | <== PEMENANG by val_RMSE |
| Grid Search | 12 | units={32,64,128}; dropout={0.1,0.3}; learning_rate={1e-3,5e-4} | 138.1 | 11.5 | 32 | 0.3 | 0.001 | 22 | 9.059 | 12.2503 | 0.525 | 10.7567 | 15.2504 | 0.4865 | 20.2648 | 0.2105 | 54 |  |

Pemenang tuning : **Optuna TPE** dengan validation RMSE **12.1812**.

## Dibanding Eksperimen Manual

| sumber | model | val_RMSE | test_MAE | test_RMSE | test_R2 | recall_TidakSehat_test | keterangan |
|---|---|---|---|---|---|---|---|
| Optuna TPE | LSTM units=32, dropout=0.317454, lr=0.00112594 | 12.1812 | 10.7002 | 15.214 | 0.4889 | 0.2368 | <== TERBAIK by val_RMSE |
| Grid Search | LSTM units=32, dropout=0.3, lr=0.001 | 12.2503 | 10.7567 | 15.2504 | 0.4865 | 0.2105 |  |
| Eksperimen manual sebelumnya | LSTM Perbaikan B_LSTM128_Dropout0.3_Dense64 (L14) | 12.6521 | 11.078 | 15.6667 | 0.4604 | 0.0921 |  |

## Final Retrain

Konfigurasi pemenang dilatih ulang pada gabungan train + validation. Test tidak dipakai untuk training.

- Test MAE: `10.1945`
- Test RMSE: `14.5596`
- Test R2: `0.5319`
- Test MAPE: `19.5818%`
