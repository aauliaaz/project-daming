"""
Generate tabel perbandingan + visualisasi 3 model (RF, LSTM, XGBoost) x 2 dataset (v1, v2).

Sumber data: notebooks/outputs/perbandingan_lengkap_v3.csv (hasil dari semua eksperimen modelling).
Output    : notebooks/outputs/perbandingan_3model_x_2dataset.{csv,png}
            notebooks/outputs/perbandingan_delta_v1_v2.csv

Jalankan dari root project:
    python src/perbandingan_model.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DIR_OUT = ROOT / 'notebooks' / 'outputs'
PATH_SUMBER = DIR_OUT / 'perbandingan_lengkap_v3.csv'


def ambil_terbaik_per_kategori(df, filter_jenis):
    """Ambil baris dengan val_RMSE terkecil dari kategori model tertentu."""
    sub = df[df['Jenis_Model'].apply(lambda x: filter_jenis(x) if isinstance(x, str) else False)]
    if len(sub) == 0:
        return None
    return sub.sort_values('val_RMSE').iloc[0]


def bangun_tabel_perbandingan(df_lengkap):
    """Bangun matriks 3 model x 2 dataset dari tabel lengkap semua eksperimen."""
    konfigurasi = [
        ('Random Forest', 'v1 (6 cuaca, proposal)',
         lambda x: 'Random Forest (eksperimen' in x or 'Random Forest (Perbaikan)' in x),
        ('LSTM', 'v1 (6 cuaca, proposal)',
         lambda x: x == 'LSTM (Perbaikan)' or 'LSTM (eksperimen' in x),
        ('XGBoost', 'v1 (6 cuaca, proposal)',
         lambda x: x == 'XGBoost (Perbaikan v2)'),
        ('Random Forest', 'v2 (16 cuaca, bersih)',
         lambda x: 'Random Forest (v2' in x),
        ('LSTM', 'v2 (16 cuaca, bersih)',
         lambda x: 'LSTM (v2' in x),
        ('XGBoost', 'v2 (16 cuaca, bersih)',
         lambda x: 'XGBoost (v3' in x),
    ]

    rows = []
    for model, dataset, filter_jenis in konfigurasi:
        terbaik = ambil_terbaik_per_kategori(df_lengkap, filter_jenis)
        if terbaik is None:
            rows.append({'Model': model, 'Dataset': dataset, 'Best_Config': '-',
                         'val_RMSE': None, 'test_RMSE': None, 'test_R2': None,
                         'test_MAE': None, 'test_MAPE': None})
            continue
        rows.append({
            'Model': model, 'Dataset': dataset,
            'Best_Config': terbaik['Model'],
            'val_RMSE':  round(float(terbaik['val_RMSE']), 4),
            'test_RMSE': round(float(terbaik['RMSE']),     4),
            'test_R2':   round(float(terbaik['R2']),       4),
            'test_MAE':  round(float(terbaik['MAE']),      4),
            'test_MAPE': round(float(terbaik['MAPE']),     4),
        })
    return pd.DataFrame(rows)


def bangun_tabel_delta(df_matrix):
    """Hitung delta v1 -> v2 per model."""
    urutan_model = ['Random Forest', 'LSTM', 'XGBoost']
    rows = []
    for m in urutan_model:
        v1 = df_matrix[(df_matrix['Model'] == m) & (df_matrix['Dataset'].str.startswith('v1'))]
        v2 = df_matrix[(df_matrix['Model'] == m) & (df_matrix['Dataset'].str.startswith('v2'))]
        if len(v1) == 0 or len(v2) == 0:
            continue
        v1, v2 = v1.iloc[0], v2.iloc[0]
        rows.append({
            'Model': m,
            'delta_test_R2':   round(v2['test_R2']   - v1['test_R2'],   3),
            'delta_test_RMSE': round(v2['test_RMSE'] - v1['test_RMSE'], 2),
            'delta_test_MAE':  round(v2['test_MAE']  - v1['test_MAE'],  2),
        })
    return pd.DataFrame(rows)


def buat_visualisasi(df_matrix, path_png):
    """Bar chart 2x2: test_R2, test_RMSE, test_MAE, val_RMSE membandingkan v1 vs v2 per model."""
    urutan_model = ['Random Forest', 'LSTM', 'XGBoost']
    metrik = [
        ('test_R2',   'Test R² (semakin tinggi semakin baik)', 'R²', (0.4, 0.75)),
        ('test_RMSE', 'Test RMSE (semakin rendah semakin baik)',    'RMSE',   None),
        ('test_MAE',  'Test MAE (semakin rendah semakin baik)',     'MAE',    None),
        ('val_RMSE',  'Validation RMSE (kriteria pemilihan model)', 'val_RMSE', None),
    ]
    warna_v1, warna_v2 = '#1f77b4', '#ff7f0e'
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for idx, (kol, judul, ylabel, ylim) in enumerate(metrik):
        ax = axes[idx // 2, idx % 2]
        x = np.arange(len(urutan_model))
        lebar = 0.35
        val_v1 = [df_matrix[(df_matrix['Model'] == m) & (df_matrix['Dataset'].str.startswith('v1'))][kol].iloc[0]
                  for m in urutan_model]
        val_v2 = [df_matrix[(df_matrix['Model'] == m) & (df_matrix['Dataset'].str.startswith('v2'))][kol].iloc[0]
                  for m in urutan_model]
        bar1 = ax.bar(x - lebar/2, val_v1, lebar, label='v1 (6 cuaca, proposal)', color=warna_v1)
        bar2 = ax.bar(x + lebar/2, val_v2, lebar, label='v2 (16 cuaca, bersih)',  color=warna_v2)
        for b, v in zip(bar1, val_v1):
            ax.text(b.get_x() + b.get_width()/2, b.get_height(), f'{v:.3f}',
                    ha='center', va='bottom', fontsize=9)
        for b, v in zip(bar2, val_v2):
            ax.text(b.get_x() + b.get_width()/2, b.get_height(), f'{v:.3f}',
                    ha='center', va='bottom', fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(urutan_model)
        ax.set_title(judul, fontsize=11, fontweight='bold')
        ax.set_ylabel(ylabel)
        ax.legend(loc='best', fontsize=9)
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        if ylim:
            ax.set_ylim(ylim)

    plt.suptitle('Perbandingan 3 Model × 2 Dataset (v1 proposal vs v2 OOS)',
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(path_png, dpi=130, bbox_inches='tight')
    plt.close()


def main():
    if not PATH_SUMBER.exists():
        raise FileNotFoundError(f'Sumber tidak ditemukan: {PATH_SUMBER}\n'
                                'Jalankan modelling RF/LSTM/XGBoost terlebih dahulu.')

    df_lengkap = pd.read_csv(PATH_SUMBER)
    df_matrix  = bangun_tabel_perbandingan(df_lengkap)
    df_delta   = bangun_tabel_delta(df_matrix)

    path_csv_matrix = DIR_OUT / 'perbandingan_3model_x_2dataset.csv'
    path_csv_delta  = DIR_OUT / 'perbandingan_delta_v1_v2.csv'
    path_png_matrix = DIR_OUT / 'perbandingan_3model_x_2dataset.png'

    df_matrix.to_csv(path_csv_matrix, index=False)
    df_delta.to_csv(path_csv_delta, index=False)
    buat_visualisasi(df_matrix, path_png_matrix)

    print('\n=== TABEL PERBANDINGAN 3 MODEL x 2 DATASET ===')
    print(df_matrix.to_string(index=False))
    print('\n=== DELTA v1 -> v2 (per model) ===')
    print(df_delta.to_string(index=False))
    print()
    print(f'CSV  : {path_csv_matrix}')
    print(f'CSV  : {path_csv_delta}')
    print(f'PNG  : {path_png_matrix}')


if __name__ == '__main__':
    main()
