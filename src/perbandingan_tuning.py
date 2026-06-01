"""
Gabungkan hasil tuning dari 3 model (Random Forest, LSTM, XGBoost) ke satu tabel + visualisasi.

Sumber data (dari notebook 02_Modelling*.ipynb):
- RF      : notebooks/outputs/hasil_model_regresi_optuna_gridsearch.csv
- LSTM    : notebooks/outputs/hasil_eksperimen_lstm.csv + evaluasi_lstm_perbaikan.csv
            + perbandingan_tuning_lstm_optuna_gridsearch.csv
- XGBoost : notebooks/outputs/hasil_model_xgboost_v2.csv

Output:
- notebooks/outputs/perbandingan_tuning_per_model.csv
- notebooks/outputs/perbandingan_tuning_per_model.png

Jalankan dari root:
    python src/perbandingan_tuning.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DIR_OUT = ROOT / 'notebooks' / 'outputs'


def baca_rf():
    """Baca tuning RF: Baseline, Manual, Optuna, Grid Search (+ varian)."""
    p = DIR_OUT / 'hasil_model_regresi_optuna_gridsearch.csv'
    if not p.exists(): return pd.DataFrame()
    df = pd.read_csv(p)
    # Sisipkan kolom Model dan Tuning_Method
    df['Model'] = 'Random Forest'
    df['Tuning_Method'] = df['model'].apply(_klasifikasi_rf)
    return df[['Model', 'Tuning_Method', 'model', 'val_RMSE', 'val_R2',
               'test_RMSE', 'test_R2', 'test_MAE', 'test_MAPE']].rename(
        columns={'model': 'Konfigurasi'})


def _klasifikasi_rf(nama):
    n = nama.lower()
    if 'baseline' in n: return 'Baseline'
    if 'awal' in n: return 'Default'
    if 'manual' in n: return 'Manual Tuning'
    if 'grid' in n: return 'Grid Search'
    if 'optuna' in n: return 'Optuna'
    return 'Lainnya'


def baca_xgboost():
    """Baca tuning XGBoost: Baseline, Optuna, Grid Search."""
    p = DIR_OUT / 'hasil_model_xgboost_v2.csv'
    if not p.exists(): return pd.DataFrame()
    df = pd.read_csv(p)
    df['Model'] = 'XGBoost'
    df['Tuning_Method'] = df['Model'].iloc[0]  # placeholder
    df['Tuning_Method'] = df['Model'].combine_first(df['Model'])
    df['Tuning_Method'] = df['Model'].apply(lambda x: x)  # reset
    df['Tuning_Method'] = [_klasifikasi_rf(nm) for nm in df['Model']]
    # Ulang dengan logika benar
    df['Tuning_Method'] = [_klasifikasi_rf(nm) for nm in df.get('Model', df['Tuning_Method'])]
    # Workaround: pakai kolom asli 'Model' dari hasil_model_xgboost_v2.csv (huruf kapital di CSV)
    p2 = pd.read_csv(p)
    df = pd.DataFrame({
        'Model': 'XGBoost',
        'Tuning_Method': [_klasifikasi_rf(nm) for nm in p2['Model']],
        'Konfigurasi': p2['Model'],
        'val_RMSE': p2['val_RMSE'], 'val_R2': p2['val_R2'],
        'test_RMSE': p2['test_RMSE'], 'test_R2': p2['test_R2'],
        'test_MAE': p2['test_MAE'], 'test_MAPE': p2['test_MAPE'],
    })
    return df


def baca_lstm():
    """Baca eksperimen LSTM (eksperimen awal + perbaikan + tuning formal)."""
    rows = []
    p1 = DIR_OUT / 'hasil_eksperimen_lstm.csv'
    if p1.exists():
        df1 = pd.read_csv(p1)
        for _, r in df1.iterrows():
            rows.append({
                'Model': 'LSTM', 'Tuning_Method': 'Arsitektur Search',
                'Konfigurasi': r['model'],
                'val_RMSE': r['val_RMSE'], 'val_R2': r.get('val_R2'),
                'test_RMSE': r['test_RMSE'], 'test_R2': r['test_R2'],
                'test_MAE': r['test_MAE'], 'test_MAPE': r['test_MAPE'],
            })
    p2 = DIR_OUT / 'evaluasi_lstm_perbaikan.csv'
    if p2.exists():
        df2 = pd.read_csv(p2)
        for _, r in df2.iterrows():
            method = 'Lookback Search'
            if '+ Weighted' in r['model']: method = 'Weighted Training'
            rows.append({
                'Model': 'LSTM', 'Tuning_Method': method,
                'Konfigurasi': r['model'],
                'val_RMSE': r['val_RMSE'], 'val_R2': None,
                'test_RMSE': r['RMSE'], 'test_R2': r['R2'],
                'test_MAE': r['MAE'], 'test_MAPE': r['MAPE'],
            })
    p3 = DIR_OUT / 'perbandingan_tuning_lstm_optuna_gridsearch.csv'
    if p3.exists():
        df3 = pd.read_csv(p3)
        for _, r in df3.iterrows():
            rows.append({
                'Model': 'LSTM', 'Tuning_Method': r['metode'],
                'Konfigurasi': (
                    f"LSTM units={int(r['best_units'])}, dropout={r['best_dropout']}, "
                    f"lr={r['best_learning_rate']}"
                ),
                'val_RMSE': r['best_val_RMSE'], 'val_R2': r['best_val_R2'],
                'test_RMSE': r['best_test_RMSE'], 'test_R2': r['best_test_R2'],
                'test_MAE': r['best_test_MAE'], 'test_MAPE': r['best_test_MAPE'],
            })
    return pd.DataFrame(rows)


def buat_visualisasi(df, path_png):
    """Bar chart val_RMSE per (Model, Tuning_Method) - sorted dalam tiap model."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    warna = {'Baseline': '#888888', 'Default': '#888888',
             'Manual Tuning': '#1f77b4', 'Optuna': '#ff7f0e',
             'Grid Search': '#2ca02c',
             'Arsitektur Search': '#d62728', 'Lookback Search': '#9467bd',
             'Weighted Training': '#8c564b', 'Lainnya': '#7f7f7f'}

    for ax, model in zip(axes, ['Random Forest', 'XGBoost', 'LSTM']):
        sub = df[df['Model'] == model].sort_values('val_RMSE').reset_index(drop=True)
        if len(sub) == 0:
            ax.set_title(f'{model} (tidak ada data)'); continue
        colors = [warna.get(m, '#7f7f7f') for m in sub['Tuning_Method']]
        bars = ax.barh(sub['Konfigurasi'][::-1], sub['val_RMSE'][::-1], color=colors[::-1])
        for b, v in zip(bars, sub['val_RMSE'][::-1]):
            ax.text(b.get_width(), b.get_y() + b.get_height()/2, f' {v:.2f}',
                    va='center', fontsize=8)
        ax.set_xlabel('val_RMSE (semakin rendah semakin baik)')
        ax.set_title(f'{model}: Perbandingan Tuning', fontweight='bold')
        ax.grid(axis='x', linestyle='--', alpha=0.5)
        # Legend per axis
        methods_in_axis = sub['Tuning_Method'].unique()
        for m in methods_in_axis:
            ax.barh([], [], color=warna.get(m, '#7f7f7f'), label=m)
        ax.legend(loc='lower right', fontsize=8)

    plt.suptitle('Perbandingan Metode Tuning per Model (sorted by val_RMSE)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(); plt.savefig(path_png, dpi=130, bbox_inches='tight')
    plt.close()


def main():
    df_rf = baca_rf()
    df_xgb = baca_xgboost()
    df_lstm = baca_lstm()

    df_all = pd.concat([df_rf, df_xgb, df_lstm], ignore_index=True, sort=False)
    # Round
    for c in ['val_RMSE', 'val_R2', 'test_RMSE', 'test_R2', 'test_MAE', 'test_MAPE']:
        if c in df_all.columns:
            df_all[c] = pd.to_numeric(df_all[c], errors='coerce').round(4)

    df_all = df_all.sort_values(['Model', 'val_RMSE']).reset_index(drop=True)

    # Tandai TERBAIK per model
    df_all['TERBAIK_per_model'] = ''
    for m in df_all['Model'].unique():
        idx_best = df_all[df_all['Model'] == m]['val_RMSE'].idxmin()
        if pd.notna(idx_best):
            df_all.loc[idx_best, 'TERBAIK_per_model'] = 'TERBAIK'

    path_csv = DIR_OUT / 'perbandingan_tuning_per_model.csv'
    path_png = DIR_OUT / 'perbandingan_tuning_per_model.png'
    df_all.to_csv(path_csv, index=False)
    buat_visualisasi(df_all, path_png)

    # Print ringkas
    print('\n=== PERBANDINGAN TUNING PER MODEL ===')
    print(df_all[['Model', 'Tuning_Method', 'Konfigurasi', 'val_RMSE', 'test_RMSE', 'test_R2', 'TERBAIK_per_model']].to_string(index=False))

    print(f'\nCSV : {path_csv}')
    print(f'PNG : {path_png}')

    print('\n=== TERBAIK per MODEL (by val_RMSE) ===')
    terbaik = df_all[df_all['TERBAIK_per_model'] == 'TERBAIK']
    print(terbaik[['Model', 'Tuning_Method', 'Konfigurasi', 'val_RMSE', 'test_RMSE', 'test_R2']].to_string(index=False))


if __name__ == '__main__':
    main()
