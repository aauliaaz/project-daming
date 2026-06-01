"""
Per-station RF: train 1 model RF terpisah untuk tiap stasiun.
Bandingkan dengan unified RF (1 model untuk semua stasiun).

Tetap dalam scope proposal (6 fitur cuaca + RF Regression). Hanya cara training-nya yang dipecah per stasiun.
"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import PredefinedSplit, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

SEED = 42; np.random.seed(SEED)
ROOT = Path('.')
DIR_OUT = ROOT / 'notebooks' / 'outputs'
PATH_DATA = ROOT / 'data' / 'final for modelling' / 'dataset_final_model.csv'
TARGET = 'ISPU PM2.5'

# ====== Load + FE (sama dengan section RF Perbaikan) ======
df = pd.read_csv(PATH_DATA)
df['tanggal'] = pd.to_datetime(df['tanggal'])
kolom_stasiun = [c for c in df.columns if c.startswith('station_')]
df['station'] = df[kolom_stasiun].idxmax(axis=1).str.replace('station_', '', regex=False)
df['bulan_sin'] = np.sin(2*np.pi*df['bulan']/12); df['bulan_cos'] = np.cos(2*np.pi*df['bulan']/12)
df['hari_minggu_sin'] = np.sin(2*np.pi*df['hari_minggu']/7); df['hari_minggu_cos'] = np.cos(2*np.pi*df['hari_minggu']/7)
def musim(b):
    if b in (11,12,1,2,3): return 'Hujan'
    if b == 4: return 'Transisi'
    return 'Kemarau'
df['musim'] = df['bulan'].map(musim)
for lag in [1,3,7]: df[f'pm25_lag_{lag}'] = df.groupby('station')[TARGET].shift(lag)
prev = df.groupby('station')[TARGET].shift(1)
def rs(s,w,f): return s.groupby(df['station']).rolling(w).agg(f).reset_index(level=0, drop=True)
df['pm25_rolling_mean_3'] = rs(prev,3,'mean'); df['pm25_rolling_mean_7'] = rs(prev,7,'mean')
df['pm25_rolling_max_7']  = rs(prev,7,'max');  df['pm25_rolling_std_7']  = rs(prev,7,'std')

fitur_lag = ['pm25_lag_1','pm25_lag_3','pm25_lag_7','pm25_rolling_mean_3','pm25_rolling_mean_7','pm25_rolling_max_7','pm25_rolling_std_7']
df = df.dropna(subset=fitur_lag + [TARGET]).reset_index(drop=True)

# Fitur per-station: TIDAK include station one-hot (karena 1 model per stasiun)
NUM_PER_ST = (['temp','humidity','visibility','windgust','solarenergy','precip']
              + fitur_lag + ['bulan_sin','bulan_cos','hari_minggu_sin','hari_minggu_cos'])
FITUR_PER_ST = NUM_PER_ST + ['musim']

# Fitur unified: include station one-hot
NUM_UNIFIED = NUM_PER_ST + kolom_stasiun
FITUR_UNIFIED = NUM_UNIFIED + ['musim']

# Time split berdasarkan tanggal global (konsisten dgn modelling sebelumnya)
ud = np.array(sorted(df['tanggal'].unique())); n = len(ud)
b1 = pd.Timestamp(ud[int(n*0.70)]); b2 = pd.Timestamp(ud[int(n*0.85)])

def buat_pipe(fitur_num, fitur_kat, params=None):
    p = dict(n_estimators=600, max_depth=15, min_samples_leaf=2, min_samples_split=5,
             random_state=SEED, n_jobs=-1)
    if params: p.update(params)
    pre = ColumnTransformer([
        ('num', SimpleImputer(strategy='median'), fitur_num),
        ('cat', Pipeline([('imp',SimpleImputer(strategy='most_frequent')),
                          ('oh',OneHotEncoder(handle_unknown='ignore'))]), fitur_kat),
    ])
    return Pipeline([('pre',pre), ('model', RandomForestRegressor(**p))])

def stat(y, p):
    return {'R2': float(r2_score(y,p)), 'RMSE': float(np.sqrt(mean_squared_error(y,p))),
            'MAE': float(mean_absolute_error(y,p))}

# ====== Unified baseline (referensi) ======
print('========== UNIFIED RF (referensi) ==========')
m_tr = df['tanggal'] < b1
m_va = (df['tanggal'] >= b1) & (df['tanggal'] < b2)
m_te = df['tanggal'] >= b2
X_tr_u, X_va_u, X_te_u = df.loc[m_tr, FITUR_UNIFIED], df.loc[m_va, FITUR_UNIFIED], df.loc[m_te, FITUR_UNIFIED]
y_tr_u, y_va_u, y_te_u = df.loc[m_tr, TARGET], df.loc[m_va, TARGET], df.loc[m_te, TARGET]

# Grid Search untuk unified
pen = np.r_[np.full(len(X_tr_u),-1), np.zeros(len(X_va_u), dtype=int)]
ps = PredefinedSplit(test_fold=pen)
Xg, yg = pd.concat([X_tr_u, X_va_u]).reset_index(drop=True), pd.concat([y_tr_u, y_va_u]).reset_index(drop=True)
grid_unified = {
    'model__n_estimators': [400, 600],
    'model__max_depth': [None, 15, 20],
    'model__min_samples_leaf': [1, 2],
    'model__min_samples_split': [2, 5],
}
gs_u = GridSearchCV(buat_pipe(NUM_UNIFIED, ['musim']), grid_unified, cv=ps,
                    scoring='neg_root_mean_squared_error', n_jobs=-1, refit=True)
gs_u.fit(Xg, yg)
params_unified = {k.replace('model__',''):v for k,v in gs_u.best_params_.items()}
model_unified = gs_u.best_estimator_
s_va_u = stat(y_va_u, model_unified.predict(X_va_u))
s_te_u = stat(y_te_u, model_unified.predict(X_te_u))
print(f'  Best params  : {params_unified}')
print(f'  Val  RMSE {s_va_u["RMSE"]:.4f} | R2 {s_va_u["R2"]:.4f}')
print(f'  Test RMSE {s_te_u["RMSE"]:.4f} | R2 {s_te_u["R2"]:.4f}')

# ====== Per-station RF ======
print('\n========== PER-STATION RF ==========')
stasiun_list = sorted(df['station'].unique())
hasil_per_st = []
prediksi_per_st_all = []

for st in stasiun_list:
    df_st = df[df['station'] == st].copy().reset_index(drop=True)
    m_tr_s = df_st['tanggal'] < b1
    m_va_s = (df_st['tanggal'] >= b1) & (df_st['tanggal'] < b2)
    m_te_s = df_st['tanggal'] >= b2
    X_tr_s, X_va_s, X_te_s = df_st.loc[m_tr_s, FITUR_PER_ST], df_st.loc[m_va_s, FITUR_PER_ST], df_st.loc[m_te_s, FITUR_PER_ST]
    y_tr_s, y_va_s, y_te_s = df_st.loc[m_tr_s, TARGET], df_st.loc[m_va_s, TARGET], df_st.loc[m_te_s, TARGET]

    # Grid Search per stasiun (lebih kecil grid karena data lebih sedikit)
    pen_s = np.r_[np.full(len(X_tr_s),-1), np.zeros(len(X_va_s), dtype=int)]
    ps_s = PredefinedSplit(test_fold=pen_s)
    Xg_s = pd.concat([X_tr_s, X_va_s]).reset_index(drop=True)
    yg_s = pd.concat([y_tr_s, y_va_s]).reset_index(drop=True)
    grid_st = {
        'model__n_estimators': [300, 600],
        'model__max_depth': [None, 10, 20],
        'model__min_samples_leaf': [1, 2],
    }
    gs_s = GridSearchCV(buat_pipe(NUM_PER_ST, ['musim']), grid_st, cv=ps_s,
                        scoring='neg_root_mean_squared_error', n_jobs=-1, refit=True)
    gs_s.fit(Xg_s, yg_s)
    params_s = {k.replace('model__',''):v for k,v in gs_s.best_params_.items()}
    model_s = gs_s.best_estimator_

    s_va = stat(y_va_s, model_s.predict(X_va_s))
    s_te = stat(y_te_s, model_s.predict(X_te_s))
    prediksi_per_st_all.append(pd.DataFrame({
        'station': st, 'tanggal': df_st.loc[m_te_s, 'tanggal'].values,
        'actual': y_te_s.values, 'predicted': model_s.predict(X_te_s),
    }))
    hasil_per_st.append({
        'station': st,
        'n_train': int(m_tr_s.sum()), 'n_val': int(m_va_s.sum()), 'n_test': int(m_te_s.sum()),
        'best_params': str(params_s),
        'val_RMSE': round(s_va['RMSE'], 4), 'val_R2': round(s_va['R2'], 4),
        'test_RMSE': round(s_te['RMSE'], 4), 'test_R2': round(s_te['R2'], 4), 'test_MAE': round(s_te['MAE'], 4),
    })
    print(f'  {st:<18s} val_RMSE {s_va["RMSE"]:6.3f} | test_RMSE {s_te["RMSE"]:6.3f} R2 {s_te["R2"]:.4f}')

df_per_st = pd.DataFrame(hasil_per_st)
df_per_st.to_csv(DIR_OUT / 'evaluasi_per_station_rf.csv', index=False)

# ====== Overall aggregate dari per-station predictions ======
df_pred_all = pd.concat(prediksi_per_st_all, ignore_index=True)
df_pred_all.to_csv(DIR_OUT / 'prediksi_per_station_rf.csv', index=False)
y_all = df_pred_all['actual'].values
p_all = df_pred_all['predicted'].values
s_per_overall = stat(y_all, p_all)

# ====== Unified per-station (sebagai benchmark) ======
unified_pred_per_st = []
for st in stasiun_list:
    df_st = df[df['station'] == st].copy()
    m_te_s = df_st['tanggal'] >= b2
    X_st_te = df_st.loc[m_te_s, FITUR_UNIFIED]
    y_st_te = df_st.loc[m_te_s, TARGET]
    p_st_te = model_unified.predict(X_st_te)
    unified_pred_per_st.append({
        'station': st,
        'unified_test_R2': round(float(r2_score(y_st_te, p_st_te)), 4),
        'unified_test_RMSE': round(float(np.sqrt(mean_squared_error(y_st_te, p_st_te))), 4),
        'unified_test_MAE': round(float(mean_absolute_error(y_st_te, p_st_te)), 4),
    })
df_unified_per_st = pd.DataFrame(unified_pred_per_st)

# ====== Comparison ======
df_compare = df_per_st[['station', 'test_R2', 'test_RMSE', 'test_MAE']].merge(
    df_unified_per_st, on='station')
df_compare['delta_R2'] = (df_compare['test_R2'] - df_compare['unified_test_R2']).round(4)
df_compare['delta_RMSE'] = (df_compare['test_RMSE'] - df_compare['unified_test_RMSE']).round(4)
df_compare.to_csv(DIR_OUT / 'perbandingan_per_station_vs_unified.csv', index=False)

print('\n========== HASIL KOMBINASI ==========')
print(f'Unified RF (Grid Search) - test_R2  : {s_te_u["R2"]:.4f} | test_RMSE: {s_te_u["RMSE"]:.4f} | test_MAE: {s_te_u["MAE"]:.4f}')
print(f'Per-station RF (concat)  - test_R2  : {s_per_overall["R2"]:.4f} | test_RMSE: {s_per_overall["RMSE"]:.4f} | test_MAE: {s_per_overall["MAE"]:.4f}')
print(f'Selisih R^2 (per-st - unified)      : {s_per_overall["R2"] - s_te_u["R2"]:+.4f}')

print('\n=== PER STASIUN: per-station vs unified ===')
print(df_compare.to_string(index=False))

# ====== Save model + metadata ======
joblib.dump(model_unified, DIR_OUT / 'model_rf_unified_perstation_compare.pkl')

import json
meta = {
    'eksperimen': 'Per-station RF vs Unified RF (in-scope proposal)',
    'dataset': 'dataset_final_model.csv (v1)',
    'unified': {
        'val_RMSE': round(s_va_u['RMSE'], 4),
        'test_RMSE': round(s_te_u['RMSE'], 4),
        'test_R2': round(s_te_u['R2'], 4),
        'test_MAE': round(s_te_u['MAE'], 4),
        'params': params_unified,
    },
    'per_station_overall': {
        'test_RMSE': round(s_per_overall['RMSE'], 4),
        'test_R2': round(s_per_overall['R2'], 4),
        'test_MAE': round(s_per_overall['MAE'], 4),
    },
    'delta_test_R2': round(s_per_overall['R2'] - s_te_u['R2'], 4),
}
with open(DIR_OUT / 'metadata_per_station_rf.json', 'w', encoding='utf-8') as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)

# ====== Visualisasi ======
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar chart per-station vs unified R^2
x = np.arange(len(df_compare))
w = 0.35
axes[0].bar(x - w/2, df_compare['unified_test_R2'], w, label='Unified RF', color='#1f77b4')
axes[0].bar(x + w/2, df_compare['test_R2'],         w, label='Per-station RF', color='#ff7f0e')
axes[0].set_xticks(x); axes[0].set_xticklabels(df_compare['station'], rotation=30, ha='right')
axes[0].set_ylabel('Test R^2')
axes[0].set_title('Test R^2: Unified vs Per-Station RF')
axes[0].axhline(s_te_u['R2'], color='blue',   ls='--', lw=1, label=f'Overall unified {s_te_u["R2"]:.3f}')
axes[0].axhline(s_per_overall['R2'], color='orange', ls='--', lw=1, label=f'Overall per-st {s_per_overall["R2"]:.3f}')
axes[0].legend(loc='lower left', fontsize=9)
axes[0].grid(axis='y', linestyle='--', alpha=0.5)
for i, v in enumerate(df_compare['unified_test_R2']):
    axes[0].text(i - w/2, v, f'{v:.2f}', ha='center', va='bottom', fontsize=8)
for i, v in enumerate(df_compare['test_R2']):
    axes[0].text(i + w/2, v, f'{v:.2f}', ha='center', va='bottom', fontsize=8)

# Delta R^2 per station
axes[1].bar(x, df_compare['delta_R2'], color=['green' if v > 0 else 'red' for v in df_compare['delta_R2']])
axes[1].set_xticks(x); axes[1].set_xticklabels(df_compare['station'], rotation=30, ha='right')
axes[1].set_ylabel('Delta Test R^2 (per-station - unified)')
axes[1].set_title('Delta R^2: positif = per-station lebih baik')
axes[1].axhline(0, color='black', lw=1)
axes[1].grid(axis='y', linestyle='--', alpha=0.5)
for i, v in enumerate(df_compare['delta_R2']):
    axes[1].text(i, v, f'{v:+.3f}', ha='center', va='bottom' if v > 0 else 'top', fontsize=9)

plt.suptitle('Per-Station RF vs Unified RF (dalam scope proposal)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(DIR_OUT / 'perbandingan_per_station_vs_unified.png', dpi=130, bbox_inches='tight')
plt.close()

print()
print('=== FILES ===')
for nm in ['evaluasi_per_station_rf.csv', 'prediksi_per_station_rf.csv',
           'perbandingan_per_station_vs_unified.csv', 'perbandingan_per_station_vs_unified.png',
           'model_rf_unified_perstation_compare.pkl', 'metadata_per_station_rf.json']:
    p = DIR_OUT / nm
    print(f'  [{"OK" if p.exists() else "MISSING"}] {nm}')
