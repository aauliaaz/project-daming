"""
4-Split (train_core + validation + train_full + test) untuk 3 model x 2 dataset.

Date boundaries (sesuai setup awal user):
  train_core : 2022-01-01 -> 2024-01-15
  validation : 2024-01-16 -> 2024-05-26
  train_full : 2022-01-01 -> 2024-05-26 (train_core + val)
  test       : 2024-05-27 -> 2025-01-01

Workflow per model:
  1. Tune hyperparameter pakai train_core, evaluasi di val
  2. Pilih best by val_RMSE
  3. Retrain pada train_full
  4. Evaluasi sekali di test

Models: RF (Grid Search), XGBoost (Optuna + Grid), LSTM (Arch A/B/C @ L14)
Datasets: v1 (6 cuaca, proposal) + v2 (16 cuaca, OOS bersih)
"""
import os, json, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from pathlib import Path
import joblib

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import PredefinedSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import xgboost as xgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

RANDOM = 42
np.random.seed(RANDOM); tf.random.set_seed(RANDOM); keras.utils.set_random_seed(RANDOM)

ROOT = Path('.')
DIR_OUT = ROOT / 'notebooks' / 'outputs'
DIR_OUT.mkdir(exist_ok=True, parents=True)
TARGET = 'ISPU PM2.5'

# Date boundaries 4-split
BATAS_VAL_MULAI  = pd.Timestamp('2024-01-16')
BATAS_VAL_AKHIR  = pd.Timestamp('2024-05-26')
BATAS_TEST_MULAI = pd.Timestamp('2024-05-27')

# ============================================================
# Helper: load + FE per dataset
# ============================================================
def load_dan_fe(versi):
    """versi = 'v1' atau 'v2'. Returns df dengan semua fitur siap split."""
    if versi == 'v1':
        path = ROOT / 'data' / 'final for modelling' / 'dataset_final_model.csv'
        kolom_cuaca = ['temp', 'humidity', 'visibility', 'windgust', 'solarenergy', 'precip']
    else:
        path = ROOT / 'data' / 'final for modelling' / 'dataset_final_model_v2.csv'
        kolom_cuaca = ['temp', 'humidity', 'visibility', 'windgust', 'solarenergy', 'precip',
                       'cloudcover', 'tempmax', 'tempmin', 'feelslike', 'uvindex',
                       'precipprob', 'sealevelpressure', 'dew', 'winddir_sin', 'winddir_cos']

    df = pd.read_csv(path)
    df['tanggal'] = pd.to_datetime(df['tanggal'])
    kolom_stasiun = [c for c in df.columns if c.startswith('station_')]
    df['station'] = df[kolom_stasiun].idxmax(axis=1).str.replace('station_', '', regex=False)

    # Cyclical time
    if 'bulan' not in df.columns: df['bulan'] = df['tanggal'].dt.month
    if 'hari_minggu' not in df.columns: df['hari_minggu'] = df['tanggal'].dt.dayofweek
    df['bulan_sin'] = np.sin(2*np.pi*df['bulan']/12); df['bulan_cos'] = np.cos(2*np.pi*df['bulan']/12)
    df['hari_minggu_sin'] = np.sin(2*np.pi*df['hari_minggu']/7); df['hari_minggu_cos'] = np.cos(2*np.pi*df['hari_minggu']/7)

    # Musim
    def musim(b):
        if b in (11,12,1,2,3): return 'Hujan'
        if b == 4: return 'Transisi'
        return 'Kemarau'
    df['musim'] = df['bulan'].map(musim)

    # Lag & rolling per station
    for lag in [1,3,7]: df[f'pm25_lag_{lag}'] = df.groupby('station')[TARGET].shift(lag)
    prev = df.groupby('station')[TARGET].shift(1)
    def rs(s,w,f): return s.groupby(df['station']).rolling(w).agg(f).reset_index(level=0, drop=True)
    df['pm25_rolling_mean_3'] = rs(prev,3,'mean'); df['pm25_rolling_mean_7'] = rs(prev,7,'mean')
    df['pm25_rolling_max_7']  = rs(prev,7,'max');  df['pm25_rolling_std_7']  = rs(prev,7,'std')

    fitur_lag = ['pm25_lag_1','pm25_lag_3','pm25_lag_7','pm25_rolling_mean_3',
                 'pm25_rolling_mean_7','pm25_rolling_max_7','pm25_rolling_std_7']
    kolom_cuaca_ada = [c for c in kolom_cuaca if c in df.columns]
    df = df.dropna(subset=fitur_lag + [TARGET]).reset_index(drop=True)

    return df, kolom_cuaca_ada, fitur_lag, kolom_stasiun


def split_4(df):
    """Bangun 4 mask: train_core, val, train_full, test."""
    m_core = df['tanggal'] < BATAS_VAL_MULAI
    m_val  = (df['tanggal'] >= BATAS_VAL_MULAI) & (df['tanggal'] <= BATAS_VAL_AKHIR)
    m_full = df['tanggal'] <= BATAS_VAL_AKHIR
    m_test = df['tanggal'] >= BATAS_TEST_MULAI
    return m_core, m_val, m_full, m_test


def ev(y, yp):
    y = np.asarray(y); yp = np.asarray(yp)
    return {'MAE': float(mean_absolute_error(y, yp)),
            'RMSE': float(np.sqrt(mean_squared_error(y, yp))),
            'R2': float(r2_score(y, yp))}


# ============================================================
# Model: Random Forest 4-split
# ============================================================
def run_rf(df, kolom_cuaca, fitur_lag, kolom_stasiun, label):
    print(f'\n--- RF 4-split [{label}] ---')
    NUM = kolom_cuaca + fitur_lag + ['bulan_sin','bulan_cos','hari_minggu_sin','hari_minggu_cos'] + kolom_stasiun
    FITUR = NUM + ['musim']

    m_core, m_val, m_full, m_test = split_4(df)
    X_core, X_val_, X_full, X_test = df.loc[m_core, FITUR], df.loc[m_val, FITUR], df.loc[m_full, FITUR], df.loc[m_test, FITUR]
    y_core, y_val_, y_full, y_test = df.loc[m_core, TARGET], df.loc[m_val, TARGET], df.loc[m_full, TARGET], df.loc[m_test, TARGET]
    print(f'  Train_core {len(X_core)} | Val {len(X_val_)} | Train_full {len(X_full)} | Test {len(X_test)}')

    def buat_pipe(params=None):
        p = dict(random_state=RANDOM, n_jobs=-1)
        if params: p.update(params)
        pre = ColumnTransformer([
            ('num', SimpleImputer(strategy='median'), NUM),
            ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                              ('oh', OneHotEncoder(handle_unknown='ignore'))]), ['musim']),
        ])
        return Pipeline([('pre', pre), ('model', RandomForestRegressor(**p))])

    # === Tune: Grid Search di train_core + val ===
    pen = np.r_[np.full(len(X_core), -1), np.zeros(len(X_val_), dtype=int)]
    ps = PredefinedSplit(test_fold=pen)
    Xg = pd.concat([X_core, X_val_]).reset_index(drop=True)
    yg = pd.concat([y_core, y_val_]).reset_index(drop=True)
    grid = {
        'model__n_estimators': [400, 600],
        'model__max_depth': [None, 15, 20],
        'model__min_samples_leaf': [1, 2],
        'model__min_samples_split': [2, 5],
    }
    t0 = time.time()
    gs = GridSearchCV(buat_pipe(), grid, cv=ps, scoring='neg_root_mean_squared_error', n_jobs=-1, refit=False)
    gs.fit(Xg, yg)
    params_best = {k.replace('model__',''): v for k,v in gs.best_params_.items()}
    val_rmse = -gs.best_score_
    print(f'  Tune (Grid): val_RMSE {val_rmse:.4f} ({time.time()-t0:.1f}s)')
    print(f'  Best params: {params_best}')

    # === Retrain pada Train_full ===
    model_final = buat_pipe(params_best)
    model_final.fit(X_full, y_full)
    pred_test = model_final.predict(X_test)
    test_metrics = ev(y_test, pred_test)
    print(f'  Test: RMSE {test_metrics["RMSE"]:.4f} | R2 {test_metrics["R2"]:.4f} | MAE {test_metrics["MAE"]:.4f}')
    return {'model': 'Random Forest', 'dataset': label, 'tuning': 'Grid Search',
            'val_RMSE': round(val_rmse, 4),
            'test_RMSE': round(test_metrics['RMSE'], 4),
            'test_R2': round(test_metrics['R2'], 4),
            'test_MAE': round(test_metrics['MAE'], 4),
            'params': params_best,
            'n_train_core': int(m_core.sum()), 'n_val': int(m_val.sum()),
            'n_train_full': int(m_full.sum()), 'n_test': int(m_test.sum())}


# ============================================================
# Model: XGBoost 4-split
# ============================================================
def run_xgb(df, kolom_cuaca, fitur_lag, kolom_stasiun, label):
    print(f'\n--- XGBoost 4-split [{label}] ---')
    NUM = kolom_cuaca + fitur_lag + ['bulan_sin','bulan_cos','hari_minggu_sin','hari_minggu_cos'] + kolom_stasiun
    FITUR = NUM + ['musim']

    m_core, m_val, m_full, m_test = split_4(df)
    X_core, X_val_, X_full, X_test = df.loc[m_core, FITUR], df.loc[m_val, FITUR], df.loc[m_full, FITUR], df.loc[m_test, FITUR]
    y_core, y_val_, y_full, y_test = df.loc[m_core, TARGET], df.loc[m_val, TARGET], df.loc[m_full, TARGET], df.loc[m_test, TARGET]
    print(f'  Train_core {len(X_core)} | Val {len(X_val_)} | Train_full {len(X_full)} | Test {len(X_test)}')

    def buat_pipe(params=None):
        p = dict(n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                 min_child_weight=1, reg_alpha=0.0, reg_lambda=1.0,
                 objective='reg:squarederror', random_state=RANDOM, tree_method='hist', n_jobs=-1, verbosity=0)
        if params: p.update(params)
        pre = ColumnTransformer([
            ('num', SimpleImputer(strategy='median'), NUM),
            ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                              ('oh', OneHotEncoder(handle_unknown='ignore'))]), ['musim']),
        ])
        return Pipeline([('pre', pre), ('model', xgb.XGBRegressor(**p))])

    # === Tune: Optuna (30 trials) ===
    def obj(trial):
        p = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 1200, step=100),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 2.0),
        }
        pp = buat_pipe(p); pp.fit(X_core, y_core)
        return float(np.sqrt(mean_squared_error(y_val_, pp.predict(X_val_))))

    t0 = time.time()
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=RANDOM))
    study.optimize(obj, n_trials=30, show_progress_bar=False)
    params_optuna = study.best_params
    val_rmse_optuna = study.best_value
    print(f'  Optuna 30 trials: val_RMSE {val_rmse_optuna:.4f} ({time.time()-t0:.1f}s)')

    # === Tune: Grid Search ===
    pen = np.r_[np.full(len(X_core), -1), np.zeros(len(X_val_), dtype=int)]
    ps = PredefinedSplit(test_fold=pen)
    Xg = pd.concat([X_core, X_val_]).reset_index(drop=True)
    yg = pd.concat([y_core, y_val_]).reset_index(drop=True)
    grid = {
        'model__n_estimators': [300, 600, 1000],
        'model__max_depth': [4, 6, 8],
        'model__learning_rate': [0.03, 0.05, 0.1],
        'model__subsample': [0.7, 0.9],
        'model__colsample_bytree': [0.7, 0.9],
    }
    t0 = time.time()
    gs = GridSearchCV(buat_pipe(), grid, cv=ps, scoring='neg_root_mean_squared_error', n_jobs=-1, refit=False)
    gs.fit(Xg, yg)
    params_grid = {k.replace('model__',''): v for k,v in gs.best_params_.items()}
    val_rmse_grid = -gs.best_score_
    print(f'  Grid Search: val_RMSE {val_rmse_grid:.4f} ({time.time()-t0:.1f}s)')

    # === Pilih best by val_RMSE ===
    if val_rmse_grid < val_rmse_optuna:
        params_best = params_grid; val_rmse = val_rmse_grid; tuning = 'Grid Search'
    else:
        params_best = params_optuna; val_rmse = val_rmse_optuna; tuning = 'Optuna'

    print(f'  -> Best tuning: {tuning} (val_RMSE {val_rmse:.4f})')

    # === Retrain pada Train_full ===
    model_final = buat_pipe(params_best)
    model_final.fit(X_full, y_full)
    pred_test = model_final.predict(X_test)
    test_metrics = ev(y_test, pred_test)
    print(f'  Test: RMSE {test_metrics["RMSE"]:.4f} | R2 {test_metrics["R2"]:.4f} | MAE {test_metrics["MAE"]:.4f}')
    return {'model': 'XGBoost', 'dataset': label, 'tuning': tuning,
            'val_RMSE': round(val_rmse, 4),
            'test_RMSE': round(test_metrics['RMSE'], 4),
            'test_R2': round(test_metrics['R2'], 4),
            'test_MAE': round(test_metrics['MAE'], 4),
            'params': params_best,
            'n_train_core': int(m_core.sum()), 'n_val': int(m_val.sum()),
            'n_train_full': int(m_full.sum()), 'n_test': int(m_test.sum())}


# ============================================================
# Model: LSTM 4-split
# ============================================================
def arsitektur_A(L, F):
    m = keras.Sequential([layers.Input(shape=(L,F)), layers.LSTM(64), layers.Dropout(0.2),
                          layers.Dense(32, activation='relu'), layers.Dense(1)])
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss='mse', metrics=['mae'])
    return m
def arsitektur_B(L, F):
    m = keras.Sequential([layers.Input(shape=(L,F)), layers.LSTM(128), layers.Dropout(0.3),
                          layers.Dense(64, activation='relu'), layers.Dense(1)])
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss='mse', metrics=['mae'])
    return m
def arsitektur_C(L, F):
    m = keras.Sequential([layers.Input(shape=(L,F)), layers.LSTM(64, return_sequences=True),
                          layers.Dropout(0.2), layers.LSTM(32), layers.Dropout(0.2),
                          layers.Dense(32, activation='relu'), layers.Dense(1)])
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss='mse', metrics=['mae'])
    return m


def run_lstm(df, kolom_cuaca, fitur_lag, kolom_stasiun, label, lookback=14):
    print(f'\n--- LSTM 4-split [{label}] ---')

    # One-hot musim manual (LSTM gak pakai ColumnTransformer)
    musim_dummies = pd.get_dummies(df['musim'], prefix='musim').astype(float)
    for col in ['musim_Hujan','musim_Kemarau','musim_Transisi']:
        if col not in musim_dummies.columns: musim_dummies[col] = 0.0
    for c in ['musim_Hujan','musim_Kemarau','musim_Transisi']:
        df[c] = musim_dummies[c].values
    musim_cols = ['musim_Hujan','musim_Kemarau','musim_Transisi']

    # Impute median (LSTM perlu data complete)
    for c in kolom_cuaca:
        df[c] = df[c].fillna(df[c].median())

    FITUR = ([TARGET] + kolom_cuaca + ['bulan_sin','bulan_cos','hari_minggu_sin','hari_minggu_cos']
             + musim_cols + kolom_stasiun)
    F = len(FITUR)

    # Build sequences
    Xs, ys, tgs, sts = [], [], [], []
    for st, sub in df.groupby('station'):
        sub = sub.sort_values('tanggal').reset_index(drop=True)
        if len(sub) <= lookback: continue
        arr = sub[FITUR].values.astype(np.float32)
        y = sub[TARGET].values.astype(np.float32)
        tgl = sub['tanggal'].values
        for i in range(lookback, len(sub)):
            Xs.append(arr[i-lookback:i]); ys.append(y[i]); tgs.append(tgl[i]); sts.append(st)
    X = np.stack(Xs); y = np.array(ys, dtype=np.float32); tgl = pd.to_datetime(np.array(tgs))

    m_core = tgl < BATAS_VAL_MULAI
    m_val  = (tgl >= BATAS_VAL_MULAI) & (tgl <= BATAS_VAL_AKHIR)
    m_full = tgl <= BATAS_VAL_AKHIR
    m_test = tgl >= BATAS_TEST_MULAI
    print(f'  Train_core {m_core.sum()} | Val {m_val.sum()} | Train_full {m_full.sum()} | Test {m_test.sum()}')

    # Scale: fit pada train_core saja saat tuning
    sx_tuning = StandardScaler().fit(X[m_core].reshape(-1, F))
    def trans(a, sx): return sx.transform(a.reshape(-1, F)).reshape(a.shape).astype(np.float32)
    X_core_s = trans(X[m_core], sx_tuning); X_val_s = trans(X[m_val], sx_tuning)
    y_core_a = y[m_core]; y_val_a = y[m_val]

    # === Tune: 3 arsitektur @ lookback 14 ===
    hasil_arch = []
    for nm, fac in [('A_LSTM64_Dropout0.2_Dense32', arsitektur_A),
                    ('B_LSTM128_Dropout0.3_Dense64', arsitektur_B),
                    ('C_LSTM64+LSTM32_Dropout0.2_Dense32', arsitektur_C)]:
        keras.utils.set_random_seed(RANDOM)
        model = fac(lookback, F)
        cb = [keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True, monitor='val_loss'),
              keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-5, monitor='val_loss')]
        t0 = time.time()
        h = model.fit(X_core_s, y_core_a, validation_data=(X_val_s, y_val_a),
                      epochs=80, batch_size=128, verbose=0, callbacks=cb)
        val_rmse = float(np.sqrt(mean_squared_error(y_val_a, model.predict(X_val_s, verbose=0).flatten())))
        print(f'    Arch {nm:40s} val_RMSE {val_rmse:.4f} ({time.time()-t0:.1f}s)')
        hasil_arch.append((nm, fac, val_rmse))

    nm_best, fac_best, val_rmse = min(hasil_arch, key=lambda x: x[2])
    print(f'  -> Best arch: {nm_best} (val_RMSE {val_rmse:.4f})')

    # === Retrain pada Train_full (re-scale dengan train_full) ===
    sx_final = StandardScaler().fit(X[m_full].reshape(-1, F))
    X_full_s = trans(X[m_full], sx_final); X_test_s = trans(X[m_test], sx_final)
    y_full_a = y[m_full]; y_test_a = y[m_test]

    keras.utils.set_random_seed(RANDOM)
    model_final = fac_best(lookback, F)
    cb_final = [keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-5, monitor='loss')]
    # Pakai average epochs dari tuning (proxy untuk berapa epochs ideal)
    epochs_retrain = 40
    model_final.fit(X_full_s, y_full_a, epochs=epochs_retrain, batch_size=128, verbose=0, callbacks=cb_final)
    pred_test = model_final.predict(X_test_s, verbose=0).flatten()
    test_metrics = ev(y_test_a, pred_test)
    print(f'  Test (retrain on train_full, {epochs_retrain} epochs): RMSE {test_metrics["RMSE"]:.4f} | R2 {test_metrics["R2"]:.4f}')

    return {'model': 'LSTM', 'dataset': label, 'tuning': f'Arch Search @ L{lookback}',
            'val_RMSE': round(val_rmse, 4),
            'test_RMSE': round(test_metrics['RMSE'], 4),
            'test_R2': round(test_metrics['R2'], 4),
            'test_MAE': round(test_metrics['MAE'], 4),
            'params': {'arch': nm_best, 'lookback': lookback, 'epochs_retrain': epochs_retrain},
            'n_train_core': int(m_core.sum()), 'n_val': int(m_val.sum()),
            'n_train_full': int(m_full.sum()), 'n_test': int(m_test.sum())}


# ============================================================
# Main: jalankan semua 6 kombinasi
# ============================================================
def main():
    semua_hasil = []

    for versi in ['v1', 'v2']:
        print(f'\n{"="*70}')
        print(f'  DATASET {versi.upper()}')
        print(f'{"="*70}')
        df, kolom_cuaca, fitur_lag, kolom_stasiun = load_dan_fe(versi)
        label = f"{versi} ({len(kolom_cuaca)} cuaca)"
        print(f'  Loaded: shape {df.shape}, {len(kolom_cuaca)} fitur cuaca')

        # RF
        try:
            hasil_rf = run_rf(df.copy(), kolom_cuaca, fitur_lag, kolom_stasiun, label)
            semua_hasil.append(hasil_rf)
        except Exception as e:
            print(f'  RF ERROR: {e}')

        # XGBoost
        try:
            hasil_xgb = run_xgb(df.copy(), kolom_cuaca, fitur_lag, kolom_stasiun, label)
            semua_hasil.append(hasil_xgb)
        except Exception as e:
            print(f'  XGB ERROR: {e}')

        # LSTM
        try:
            hasil_lstm = run_lstm(df.copy(), kolom_cuaca, fitur_lag, kolom_stasiun, label)
            semua_hasil.append(hasil_lstm)
        except Exception as e:
            print(f'  LSTM ERROR: {e}')

    # ============================================================
    # Save hasil
    # ============================================================
    df_hasil = pd.DataFrame(semua_hasil)
    df_hasil['params_str'] = df_hasil['params'].apply(lambda x: json.dumps(x))
    df_hasil_save = df_hasil.drop(columns=['params'])
    df_hasil_save.to_csv(DIR_OUT / 'perbandingan_4split_3model_2dataset.csv', index=False)

    # Save metadata per (model, dataset)
    for h in semua_hasil:
        key = f"{h['model'].lower().replace(' ','_')}_{h['dataset'].split()[0]}"
        with open(DIR_OUT / f'4split_{key}.json', 'w', encoding='utf-8') as f:
            json.dump(h, f, indent=2, ensure_ascii=False)

    # ============================================================
    # Visualisasi: bar chart 3 model x 2 dataset
    # ============================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    metrik_plot = [('test_R2', 'Test R² (tinggi = baik)', (0.4, 0.75)),
                   ('test_RMSE', 'Test RMSE (rendah = baik)', None)]
    model_order = ['Random Forest', 'XGBoost', 'LSTM']

    for ax, (kol, judul, ylim) in zip(axes, metrik_plot):
        x = np.arange(len(model_order))
        w = 0.35
        val_v1 = [df_hasil[(df_hasil['model']==m) & (df_hasil['dataset'].str.startswith('v1'))][kol].iloc[0]
                  if len(df_hasil[(df_hasil['model']==m) & (df_hasil['dataset'].str.startswith('v1'))]) > 0 else 0
                  for m in model_order]
        val_v2 = [df_hasil[(df_hasil['model']==m) & (df_hasil['dataset'].str.startswith('v2'))][kol].iloc[0]
                  if len(df_hasil[(df_hasil['model']==m) & (df_hasil['dataset'].str.startswith('v2'))]) > 0 else 0
                  for m in model_order]
        b1 = ax.bar(x-w/2, val_v1, w, label='v1 (6 cuaca, proposal)', color='#1f77b4')
        b2 = ax.bar(x+w/2, val_v2, w, label='v2 (16 cuaca, bersih)', color='#ff7f0e')
        for b, v in zip(b1, val_v1):
            ax.text(b.get_x()+b.get_width()/2, b.get_height(), f'{v:.3f}', ha='center', va='bottom', fontsize=9)
        for b, v in zip(b2, val_v2):
            ax.text(b.get_x()+b.get_width()/2, b.get_height(), f'{v:.3f}', ha='center', va='bottom', fontsize=9)
        ax.set_xticks(x); ax.set_xticklabels(model_order)
        ax.set_title(judul, fontweight='bold')
        ax.legend(loc='best'); ax.grid(axis='y', linestyle='--', alpha=0.5)
        if ylim: ax.set_ylim(ylim)

    plt.suptitle('4-Split (train_core + val + train_full + test) - 3 Model x 2 Dataset',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(); plt.savefig(DIR_OUT / 'perbandingan_4split_3model_2dataset.png', dpi=130, bbox_inches='tight')
    plt.close()

    print('\n\n' + '='*70)
    print('  HASIL AKHIR 4-SPLIT')
    print('='*70)
    print(df_hasil_save[['model', 'dataset', 'tuning', 'val_RMSE', 'test_RMSE', 'test_R2', 'test_MAE']].to_string(index=False))

    print(f'\nFile output:')
    print(f'  {DIR_OUT / "perbandingan_4split_3model_2dataset.csv"}')
    print(f'  {DIR_OUT / "perbandingan_4split_3model_2dataset.png"}')
    for h in semua_hasil:
        key = f"{h['model'].lower().replace(' ','_')}_{h['dataset'].split()[0]}"
        print(f'  {DIR_OUT / f"4split_{key}.json"}')


if __name__ == '__main__':
    main()
