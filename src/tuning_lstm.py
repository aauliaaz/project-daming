
## python src/tuning_lstm.py

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import time
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import joblib
import matplotlib
import numpy as np
import optuna
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "final for modelling" / "dataset_final_model.csv"
DIR_OUTPUT = ROOT / "notebooks" / "outputs"

TARGET = "ISPU PM2.5"
KOLOM_CUACA = ["temp", "humidity", "visibility", "windgust", "solarenergy", "precip"]
RANDOM = 42


@dataclass
class DataBundle:
    fitur: list[str]
    lookback: int
    tanggal: pd.DatetimeIndex
    stasiun: np.ndarray
    X: np.ndarray
    y: np.ndarray
    m_train: np.ndarray
    m_val: np.ndarray
    m_test: np.ndarray
    scaler_X: StandardScaler
    scaler_y: StandardScaler
    X_train_s: np.ndarray
    X_val_s: np.ndarray
    X_test_s: np.ndarray
    y_train_s: np.ndarray
    y_val_s: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lookback", type=int, default=7)
    parser.add_argument("--optuna-trials", type=int, default=16)
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument(
        "--grid-limit",
        type=int,
        default=None,
        help="Batasi jumlah kombinasi Grid Search untuk smoke test.",
    )
    parser.add_argument(
        "--skip-final",
        action="store_true",
        help="Lewati retrain train+validation; berguna untuk smoke test.",
    )
    return parser.parse_args()


def set_seed() -> None:
    random.seed(RANDOM)
    np.random.seed(RANDOM)
    tf.keras.utils.set_random_seed(RANDOM)


def mape_aman(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1.0) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.abs(y_true) > epsilon
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def metrik_regresi(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "R2": float(r2_score(y_true, y_pred)),
        "MAPE": mape_aman(y_true, y_pred),
    }


def kategori_ispu(value: float) -> str:
    if value <= 50:
        return "Baik"
    if value <= 100:
        return "Sedang"
    if value < 200:
        return "Tidak Sehat"
    if value < 300:
        return "Sangat Tidak Sehat"
    return "Berbahaya"


def recall_tidak_sehat(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    aktual = pd.Series(np.asarray(y_true)).map(kategori_ispu)
    prediksi = pd.Series(np.asarray(y_pred)).map(kategori_ispu)
    return float(
        recall_score(
            aktual,
            prediksi,
            labels=["Tidak Sehat"],
            average="macro",
            zero_division=0,
        )
    )


def evaluasi_pm25_tinggi(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = y_true > 100
    if not mask.any():
        return {
            "jumlah_aktual_PM25_gt_100": 0,
            "jumlah_prediksi_PM25_gt_100": int((y_pred > 100).sum()),
            "MAE_PM25_gt_100": float("nan"),
            "RMSE_PM25_gt_100": float("nan"),
        }
    return {
        "jumlah_aktual_PM25_gt_100": int(mask.sum()),
        "jumlah_prediksi_PM25_gt_100": int((y_pred > 100).sum()),
        "MAE_PM25_gt_100": float(mean_absolute_error(y_true[mask], y_pred[mask])),
        "RMSE_PM25_gt_100": float(mean_squared_error(y_true[mask], y_pred[mask]) ** 0.5),
    }


def label_musim(bulan: int) -> str:
    if bulan in (11, 12, 1, 2, 3):
        return "Hujan"
    if bulan == 4:
        return "Transisi"
    return "Kemarau"


def bangun_sequences(
    df: pd.DataFrame, fitur: list[str], lookback: int
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, np.ndarray]:
    X_list: list[np.ndarray] = []
    y_list: list[float] = []
    tanggal_list: list[np.datetime64] = []
    stasiun_list: list[str] = []
    for stasiun, sub in df.groupby("station"):
        sub = sub.sort_values("tanggal").reset_index(drop=True)
        arr = sub[fitur].values.astype(np.float32)
        y = sub[TARGET].values.astype(np.float32)
        tanggal = sub["tanggal"].values
        for i in range(lookback, len(sub)):
            X_list.append(arr[i - lookback : i])
            y_list.append(y[i])
            tanggal_list.append(tanggal[i])
            stasiun_list.append(stasiun)
    return (
        np.stack(X_list),
        np.asarray(y_list, dtype=np.float32),
        pd.to_datetime(np.asarray(tanggal_list)),
        np.asarray(stasiun_list),
    )


def siapkan_data(lookback: int) -> DataBundle:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH, parse_dates=["tanggal"])
    station_cols = [c for c in df.columns if c.startswith("station_")]
    if not station_cols:
        raise ValueError("Kolom one-hot station_ tidak ditemukan.")

    df["station"] = df[station_cols].idxmax(axis=1).str.replace("station_", "", regex=False)
    tanpa_cuaca = df[KOLOM_CUACA].isna().all(axis=1)
    df = df.loc[~tanpa_cuaca].copy()
    df = df.sort_values(["station", "tanggal"]).reset_index(drop=True)

    df["bulan_sin"] = np.sin(2 * np.pi * df["bulan"] / 12)
    df["bulan_cos"] = np.cos(2 * np.pi * df["bulan"] / 12)
    df["hari_minggu_sin"] = np.sin(2 * np.pi * df["hari_minggu"] / 7)
    df["hari_minggu_cos"] = np.cos(2 * np.pi * df["hari_minggu"] / 7)
    df["musim"] = df["bulan"].map(label_musim)

    musim = pd.get_dummies(df["musim"], prefix="musim").astype(float)
    musim_cols = ["musim_Hujan", "musim_Kemarau", "musim_Transisi"]
    for col in musim_cols:
        df[col] = musim[col] if col in musim else 0.0

    fitur = (
        [TARGET]
        + KOLOM_CUACA
        + ["bulan_sin", "bulan_cos", "hari_minggu_sin", "hari_minggu_cos"]
        + musim_cols
        + station_cols
    )
    for col in fitur:
        if df[col].isna().any():
            df[col] = df.groupby("station")[col].transform(lambda s: s.fillna(s.median()))
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    X, y, tanggal, stasiun = bangun_sequences(df, fitur, lookback)
    tanggal_unik = np.array(sorted(np.unique(tanggal)))
    batas_train = pd.Timestamp(tanggal_unik[int(len(tanggal_unik) * 0.70)])
    batas_val = pd.Timestamp(tanggal_unik[int(len(tanggal_unik) * 0.85)])
    m_train = np.asarray(tanggal < batas_train)
    m_val = np.asarray((tanggal >= batas_train) & (tanggal < batas_val))
    m_test = np.asarray(tanggal >= batas_val)

    n_fitur = X.shape[2]
    scaler_X = StandardScaler().fit(X[m_train].reshape(-1, n_fitur))
    scaler_y = StandardScaler().fit(y[m_train].reshape(-1, 1))

    def trans_X(values: np.ndarray) -> np.ndarray:
        return scaler_X.transform(values.reshape(-1, n_fitur)).reshape(values.shape).astype(np.float32)

    def trans_y(values: np.ndarray) -> np.ndarray:
        return scaler_y.transform(values.reshape(-1, 1)).ravel().astype(np.float32)

    print("Dataset tuning LSTM siap")
    print(f"  Data source     : {DATA_PATH}")
    print(f"  Baris dibuang   : {int(tanpa_cuaca.sum())} (tanpa seluruh data cuaca)")
    print(f"  Lookback        : {lookback} hari")
    print(f"  Fitur/timestep  : {len(fitur)}")
    print(f"  Train sequence  : {int(m_train.sum())}")
    print(f"  Validation      : {int(m_val.sum())}")
    print(f"  Test            : {int(m_test.sum())}")

    return DataBundle(
        fitur=fitur,
        lookback=lookback,
        tanggal=tanggal,
        stasiun=stasiun,
        X=X,
        y=y,
        m_train=m_train,
        m_val=m_val,
        m_test=m_test,
        scaler_X=scaler_X,
        scaler_y=scaler_y,
        X_train_s=trans_X(X[m_train]),
        X_val_s=trans_X(X[m_val]),
        X_test_s=trans_X(X[m_test]),
        y_train_s=trans_y(y[m_train]),
        y_val_s=trans_y(y[m_val]),
    )


def bangun_model(params: dict[str, Any], lookback: int, n_fitur: int) -> keras.Model:
    units = int(params["units"])
    model = keras.Sequential(
        [
            layers.Input(shape=(lookback, n_fitur)),
            layers.LSTM(units),
            layers.Dropout(float(params["dropout"])),
            layers.Dense(max(16, units // 2), activation="relu"),
            layers.Dense(1),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=float(params["learning_rate"])),
        loss="mse",
        metrics=["mae"],
    )
    return model


def inverse_y(scaler: StandardScaler, values: np.ndarray) -> np.ndarray:
    return scaler.inverse_transform(np.asarray(values).reshape(-1, 1)).ravel()


def latih_kandidat(
    data: DataBundle,
    params: dict[str, Any],
    max_epochs: int,
    patience: int,
    evaluasi_test: bool = False,
) -> dict[str, Any]:
    keras.backend.clear_session()
    gc.collect()
    set_seed()
    model = bangun_model(params, data.lookback, data.X.shape[2])
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
            verbose=0,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            patience=max(2, patience // 2),
            factor=0.5,
            min_lr=1e-5,
            verbose=0,
        ),
    ]
    mulai = time.time()
    history = model.fit(
        data.X_train_s,
        data.y_train_s,
        validation_data=(data.X_val_s, data.y_val_s),
        epochs=max_epochs,
        batch_size=64,
        callbacks=callbacks,
        verbose=0,
    )
    waktu = time.time() - mulai
    pred_val = inverse_y(data.scaler_y, model.predict(data.X_val_s, verbose=0))
    hasil: dict[str, Any] = {
        **params,
        "epoch_terbaik": int(np.argmin(history.history["val_loss"])) + 1,
        "jumlah_epoch": len(history.history["loss"]),
        "waktu_detik": round(waktu, 1),
        "val": metrik_regresi(data.y[data.m_val], pred_val),
    }
    if evaluasi_test:
        pred_test = inverse_y(data.scaler_y, model.predict(data.X_test_s, verbose=0))
        hasil["test"] = metrik_regresi(data.y[data.m_test], pred_test)
        hasil["recall_TidakSehat_test"] = recall_tidak_sehat(data.y[data.m_test], pred_test)
        hasil["evaluasi_pm25_tinggi"] = evaluasi_pm25_tinggi(data.y[data.m_test], pred_test)
        hasil["pred_test"] = pred_test
        hasil["model"] = model
        hasil["history"] = history.history
    return hasil


def row_tuning(hasil: dict[str, Any], nomor: int) -> dict[str, Any]:
    return {
        "trial": nomor,
        "units": int(hasil["units"]),
        "dropout": round(float(hasil["dropout"]), 6),
        "learning_rate": round(float(hasil["learning_rate"]), 8),
        "val_MAE": round(hasil["val"]["MAE"], 4),
        "val_RMSE": round(hasil["val"]["RMSE"], 4),
        "val_R2": round(hasil["val"]["R2"], 4),
        "val_MAPE": round(hasil["val"]["MAPE"], 4),
        "epoch_terbaik": hasil["epoch_terbaik"],
        "jumlah_epoch": hasil["jumlah_epoch"],
        "waktu_detik": hasil["waktu_detik"],
    }


def jalankan_grid(
    data: DataBundle, max_epochs: int, patience: int, grid_limit: int | None
) -> tuple[pd.DataFrame, float]:
    ruang = {
        "units": [32, 64, 128],
        "dropout": [0.1, 0.3],
        "learning_rate": [1e-3, 5e-4],
    }
    kombinasi = list(product(ruang["units"], ruang["dropout"], ruang["learning_rate"]))
    if grid_limit is not None:
        kombinasi = kombinasi[:grid_limit]
    print(f"\nGrid Search: {len(kombinasi)} kombinasi diskret")

    mulai = time.time()
    rows: list[dict[str, Any]] = []
    for nomor, (units, dropout, learning_rate) in enumerate(kombinasi, 1):
        params = {"units": units, "dropout": dropout, "learning_rate": learning_rate}
        hasil = latih_kandidat(data, params, max_epochs, patience)
        rows.append(row_tuning(hasil, nomor))
        print(
            f"  [{nomor:02d}/{len(kombinasi):02d}] units={units:3d} dropout={dropout:.1f} "
            f"lr={learning_rate:.0e} -> val_RMSE={hasil['val']['RMSE']:.4f} "
            f"ep={hasil['epoch_terbaik']:02d} waktu={hasil['waktu_detik']:.1f}s"
        )
    waktu = time.time() - mulai
    df = pd.DataFrame(rows).sort_values("val_RMSE").reset_index(drop=True)
    df["keterangan"] = np.where(df.index == 0, "<== TERBAIK Grid Search", "")
    df.to_csv(DIR_OUTPUT / "hasil_tuning_lstm_gridsearch.csv", index=False)
    return df, waktu


def jalankan_optuna(
    data: DataBundle, max_epochs: int, patience: int, n_trials: int
) -> tuple[pd.DataFrame, float]:
    print(f"\nOptuna TPE: {n_trials} trial adaptif")
    rows: list[dict[str, Any]] = []

    def objective(trial: optuna.Trial) -> float:
        params = {
            "units": trial.suggest_categorical("units", [32, 64, 96, 128]),
            "dropout": trial.suggest_float("dropout", 0.1, 0.4),
            "learning_rate": trial.suggest_float("learning_rate", 3e-4, 2e-3, log=True),
        }
        hasil = latih_kandidat(data, params, max_epochs, patience)
        rows.append(row_tuning(hasil, trial.number))
        print(
            f"  [{trial.number + 1:02d}/{n_trials:02d}] units={params['units']:3d} "
            f"dropout={params['dropout']:.3f} lr={params['learning_rate']:.6f} "
            f"-> val_RMSE={hasil['val']['RMSE']:.4f} "
            f"ep={hasil['epoch_terbaik']:02d} waktu={hasil['waktu_detik']:.1f}s"
        )
        return float(hasil["val"]["RMSE"])

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    studi = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM),
        study_name="lstm_pm25_optuna_tpe",
    )
    mulai = time.time()
    studi.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    waktu = time.time() - mulai

    df = pd.DataFrame(rows).sort_values("val_RMSE").reset_index(drop=True)
    df["keterangan"] = np.where(df.index == 0, "<== TERBAIK Optuna TPE", "")
    df.to_csv(DIR_OUTPUT / "hasil_tuning_lstm_optuna.csv", index=False)
    return df, waktu


def params_dari_row(row: pd.Series) -> dict[str, Any]:
    return {
        "units": int(row["units"]),
        "dropout": float(row["dropout"]),
        "learning_rate": float(row["learning_rate"]),
    }


def ringkas_pemenang_metode(
    metode: str,
    df_tuning: pd.DataFrame,
    waktu_total: float,
    best_row: pd.Series,
    hasil_best: dict[str, Any],
    ruang: str,
) -> dict[str, Any]:
    high = hasil_best["evaluasi_pm25_tinggi"]
    return {
        "metode": metode,
        "n_evaluasi": len(df_tuning),
        "ruang_pencarian": ruang,
        "waktu_total_detik": round(waktu_total, 1),
        "rata2_waktu_per_trial": round(waktu_total / len(df_tuning), 1),
        "best_units": int(hasil_best["units"]),
        "best_dropout": round(float(hasil_best["dropout"]), 6),
        "best_learning_rate": round(float(hasil_best["learning_rate"]), 8),

        "best_epoch": int(best_row["epoch_terbaik"]),
        "best_val_MAE": round(float(best_row["val_MAE"]), 4),
        "best_val_RMSE": round(float(best_row["val_RMSE"]), 4),
        "best_val_R2": round(float(best_row["val_R2"]), 4),
        "best_test_MAE": round(hasil_best["test"]["MAE"], 4),
        "best_test_RMSE": round(hasil_best["test"]["RMSE"], 4),
        "best_test_R2": round(hasil_best["test"]["R2"], 4),
        "best_test_MAPE": round(hasil_best["test"]["MAPE"], 4),
        "recall_TidakSehat_test": round(hasil_best["recall_TidakSehat_test"], 4),
        "jumlah_prediksi_PM25_gt_100": high["jumlah_prediksi_PM25_gt_100"],
    }


def buat_plot_perbandingan(
    df_grid: pd.DataFrame, df_optuna: pd.DataFrame, df_compare: pd.DataFrame
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    warna = ["#4C72B0", "#DD8452"]
    label = df_compare["metode"].tolist()

    val = df_compare["best_val_RMSE"].tolist()
    axes[0, 0].bar(label, val, color=warna)
    axes[0, 0].set_title("Best Validation RMSE")
    axes[0, 0].set_ylabel("RMSE")

    test = df_compare["best_test_RMSE"].tolist()
    axes[0, 1].bar(label, test, color=warna)
    axes[0, 1].set_title("Test RMSE dari Pemenang Validation")
    axes[0, 1].set_ylabel("RMSE")

    waktu = df_compare["waktu_total_detik"].tolist()
    axes[1, 0].bar(label, waktu, color=warna)
    axes[1, 0].set_title("Total Waktu Tuning")
    axes[1, 0].set_ylabel("detik")

    grid_run = df_grid.sort_values("trial")["val_RMSE"].to_numpy()
    optuna_run = df_optuna.sort_values("trial")["val_RMSE"].to_numpy()
    axes[1, 1].plot(
        np.arange(1, len(grid_run) + 1),
        np.minimum.accumulate(grid_run),
        marker="o",
        label="Grid Search",
    )
    axes[1, 1].plot(
        np.arange(1, len(optuna_run) + 1),
        np.minimum.accumulate(optuna_run),
        marker="o",
        label="Optuna TPE",
    )
    axes[1, 1].set_title("Konvergensi Best val_RMSE")
    axes[1, 1].set_xlabel("trial ke-")
    axes[1, 1].set_ylabel("best val_RMSE sejauh ini")
    axes[1, 1].legend()

    for ax in axes.flat:
        ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(DIR_OUTPUT / "perbandingan_tuning_lstm.png", dpi=150, bbox_inches="tight")
    plt.close()


def retrain_final(
    data: DataBundle, params: dict[str, Any], epoch_terbaik: int
) -> tuple[keras.Model, StandardScaler, StandardScaler, np.ndarray, dict[str, float], float]:
    keras.backend.clear_session()
    gc.collect()
    set_seed()

    X_train_val = np.concatenate([data.X[data.m_train], data.X[data.m_val]])
    y_train_val = np.concatenate([data.y[data.m_train], data.y[data.m_val]])
    n_fitur = X_train_val.shape[2]
    scaler_X = StandardScaler().fit(X_train_val.reshape(-1, n_fitur))
    scaler_y = StandardScaler().fit(y_train_val.reshape(-1, 1))

    def trans_X(values: np.ndarray) -> np.ndarray:
        return scaler_X.transform(values.reshape(-1, n_fitur)).reshape(values.shape).astype(np.float32)

    X_train_val_s = trans_X(X_train_val)
    X_test_s = trans_X(data.X[data.m_test])
    y_train_val_s = scaler_y.transform(y_train_val.reshape(-1, 1)).ravel().astype(np.float32)

    model = bangun_model(params, data.lookback, n_fitur)
    mulai = time.time()
    model.fit(
        X_train_val_s,
        y_train_val_s,
        epochs=max(1, epoch_terbaik),
        batch_size=64,
        verbose=0,
    )
    waktu = time.time() - mulai
    pred_test = inverse_y(scaler_y, model.predict(X_test_s, verbose=0))
    return model, scaler_X, scaler_y, pred_test, metrik_regresi(data.y[data.m_test], pred_test), waktu


def buat_prediksi_csv(data: DataBundle, pred_test: np.ndarray) -> None:
    df = pd.DataFrame(
        {
            "tanggal": data.tanggal[data.m_test],
            "station": data.stasiun[data.m_test],
            TARGET: data.y[data.m_test],
            "prediksi_pm25": pred_test,
        }
    )
    df["residual"] = df[TARGET] - df["prediksi_pm25"]
    df["abs_error"] = df["residual"].abs()
    df["kategori_aktual"] = df[TARGET].map(kategori_ispu)
    df["kategori_prediksi"] = df["prediksi_pm25"].map(kategori_ispu)
    df.to_csv(DIR_OUTPUT / "prediksi_vs_aktual_lstm_tuning.csv", index=False)


def buat_perbandingan_manual(df_compare: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    path_manual = DIR_OUTPUT / "evaluasi_lstm_perbaikan.csv"
    if path_manual.exists():
        manual = pd.read_csv(path_manual).sort_values("val_RMSE").iloc[0]
        rows.append(
            {
                "sumber": "Eksperimen manual sebelumnya",
                "model": manual["model"],
                "val_RMSE": manual["val_RMSE"],
                "test_MAE": manual["MAE"],
                "test_RMSE": manual["RMSE"],
                "test_R2": manual["R2"],
                "recall_TidakSehat_test": manual["recall_Tidak_Sehat"],
            }
        )
    for _, row in df_compare.iterrows():
        rows.append(
            {
                "sumber": row["metode"],
                "model": (
                    f"LSTM units={int(row['best_units'])}, dropout={row['best_dropout']}, "
                    f"lr={row['best_learning_rate']}"
                ),
                "val_RMSE": row["best_val_RMSE"],
                "test_MAE": row["best_test_MAE"],
                "test_RMSE": row["best_test_RMSE"],
                "test_R2": row["best_test_R2"],
                "recall_TidakSehat_test": row["recall_TidakSehat_test"],
            }
        )
    df = pd.DataFrame(rows).sort_values("val_RMSE").reset_index(drop=True)
    df["keterangan"] = np.where(df.index == 0, "<== TERBAIK by val_RMSE", "")
    df.to_csv(DIR_OUTPUT / "perbandingan_lstm_manual_vs_tuning.csv", index=False)
    return df


def dataframe_ke_markdown(df: pd.DataFrame) -> str:
    """Buat tabel Markdown sederhana tanpa dependency opsional pandas.tabulate."""
    kolom = [str(c) for c in df.columns]
    baris = []
    for values in df.itertuples(index=False, name=None):
        baris.append([str(v).replace("|", "\\|") for v in values])
    lines = [
        "| " + " | ".join(kolom) + " |",
        "|" + "|".join("---" for _ in kolom) + "|",
    ]
    lines.extend("| " + " | ".join(values) + " |" for values in baris)
    return "\n".join(lines)


def buat_laporan(
    args: argparse.Namespace,
    df_compare: pd.DataFrame,
    df_manual: pd.DataFrame,
    final_metrics: dict[str, float] | None,
) -> None:
    best = df_compare.sort_values("best_val_RMSE").iloc[0]
    lines = [
        "# Perbandingan Tuning LSTM: Optuna TPE vs Grid Search",
        "",
        f"- Lookback dikunci pada `{args.lookback}` hari, mengikuti pemenang eksperimen manual sebelumnya.",
        "- Pemilihan model memakai `val_RMSE` terkecil. Test set hanya dilaporkan setelah pemenang tiap metode dipilih.",
        "- Grid Search mencoba seluruh kombinasi diskret. Optuna TPE memilih konfigurasi berikutnya secara adaptif.",
        "",
        "## Hasil Utama",
        "",
        dataframe_ke_markdown(df_compare),
        "",
        f"Pemenang tuning : **{best['metode']}** dengan validation RMSE **{best['best_val_RMSE']:.4f}**.",
        "",
        "## Dibanding Eksperimen Manual",
        "",
        dataframe_ke_markdown(df_manual),
        "",
    ]
    if final_metrics is not None:
        lines.extend(
            [
                "## Final Retrain",
                "",
                "Konfigurasi pemenang dilatih ulang pada gabungan train + validation. Test tidak dipakai untuk training.",
                "",
                f"- Test MAE: `{final_metrics['MAE']:.4f}`",
                f"- Test RMSE: `{final_metrics['RMSE']:.4f}`",
                f"- Test R2: `{final_metrics['R2']:.4f}`",
                f"- Test MAPE: `{final_metrics['MAPE']:.4f}%`",
                "",
            ]
        )

    (DIR_OUTPUT / "laporan_tuning_lstm_optuna_gridsearch.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    DIR_OUTPUT.mkdir(parents=True, exist_ok=True)
    set_seed()
    data = siapkan_data(args.lookback)

    df_grid, waktu_grid = jalankan_grid(data, args.max_epochs, args.patience, args.grid_limit)
    df_optuna, waktu_optuna = jalankan_optuna(
        data, args.max_epochs, args.patience, args.optuna_trials
    )

    params_grid = params_dari_row(df_grid.iloc[0])
    params_optuna = params_dari_row(df_optuna.iloc[0])
    print("\nEvaluasi test hanya untuk pemenang validation tiap metode")
    best_grid = latih_kandidat(data, params_grid, args.max_epochs, args.patience, evaluasi_test=True)
    best_optuna = latih_kandidat(data, params_optuna, args.max_epochs, args.patience, evaluasi_test=True)

    df_compare = pd.DataFrame(
        [
            ringkas_pemenang_metode(
                "Grid Search",
                df_grid,
                waktu_grid,
                df_grid.iloc[0],
                best_grid,
                "units={32,64,128}; dropout={0.1,0.3}; learning_rate={1e-3,5e-4}",
            ),
            ringkas_pemenang_metode(
                "Optuna TPE",
                df_optuna,
                waktu_optuna,
                df_optuna.iloc[0],
                best_optuna,
                "units={32,64,96,128}; dropout=[0.1,0.4]; learning_rate=[3e-4,2e-3] log",
            ),
        ]
    )
    df_compare = df_compare.sort_values("best_val_RMSE").reset_index(drop=True)
    df_compare["keterangan"] = np.where(df_compare.index == 0, "<== PEMENANG by val_RMSE", "")
    df_compare.to_csv(DIR_OUTPUT / "perbandingan_tuning_lstm_optuna_gridsearch.csv", index=False)
    buat_plot_perbandingan(df_grid, df_optuna, df_compare)
    df_manual = buat_perbandingan_manual(df_compare)

    best_method = str(df_compare.iloc[0]["metode"])
    best_search = best_grid if best_method == "Grid Search" else best_optuna
    best_search_row = df_grid.iloc[0] if best_method == "Grid Search" else df_optuna.iloc[0]
    best_params = {
        "units": int(best_search["units"]),
        "dropout": float(best_search["dropout"]),
        "learning_rate": float(best_search["learning_rate"]),
    }

    final_metrics: dict[str, float] | None = None
    if not args.skip_final:
        print(f"\nRetrain final train+validation memakai pemenang: {best_method}")
        model, scaler_X, scaler_y, pred_final, final_metrics, waktu_final = retrain_final(
            data, best_params, int(best_search_row["epoch_terbaik"])
        )
        recall_final = recall_tidak_sehat(data.y[data.m_test], pred_final)
        high_final = evaluasi_pm25_tinggi(data.y[data.m_test], pred_final)
        model.save(DIR_OUTPUT / "model_lstm_terbaik_tuning.keras")
        joblib.dump(
            {
                "scaler_X": scaler_X,
                "scaler_y": scaler_y,
                "fitur": data.fitur,
                "lookback": data.lookback,
            },
            DIR_OUTPUT / "scaler_lstm_terbaik_tuning.pkl",
        )
        buat_prediksi_csv(data, pred_final)
        metadata = {
            "algoritma": "LSTM (Keras)",
            "metode_pemenang": best_method,
            "best_config": best_params,
            "lookback": data.lookback,
            "n_features": len(data.fitur),
            "features_per_timestep": data.fitur,
            "skema_split": "time-based 70/15/15 pada tanggal target sequence",
            "pemilihan_model": "best validation RMSE; test tidak dipakai untuk memilih konfigurasi",
            "final_model_dilatih_pada": "train + validation; test tidak dipakai untuk training",
            "metrik_search_validation": {
                "MAE": float(best_search_row["val_MAE"]),
                "RMSE": float(best_search_row["val_RMSE"]),
                "R2": float(best_search_row["val_R2"]),
                "MAPE": float(best_search_row["val_MAPE"]),
            },
            "metrik_search_test": best_search["test"],
            "metrik_final_test_after_retrain": final_metrics,
            "recall_TidakSehat_final_test": recall_final,
            "evaluasi_pm25_tinggi_final_test": high_final,
            "waktu_grid_search_detik": waktu_grid,
            "waktu_optuna_detik": waktu_optuna,
            "waktu_retrain_final_detik": waktu_final,
            "n_evaluasi_grid": len(df_grid),
            "n_evaluasi_optuna": len(df_optuna),
        }
        (DIR_OUTPUT / "metadata_lstm_terbaik_tuning.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    buat_laporan(args, df_compare, df_manual, final_metrics)
    print("\n=== PERBANDINGAN OPTUNA VS GRID SEARCH ===")
    print(
        df_compare[
            [
                "metode",
                "n_evaluasi",
                "waktu_total_detik",
                "best_units",
                "best_dropout",
                "best_learning_rate",
                "best_val_RMSE",
                "best_test_RMSE",
                "best_test_R2",
                "recall_TidakSehat_test",
                "keterangan",
            ]
        ].to_string(index=False)
    )
    print("\nOutput tersimpan di:", DIR_OUTPUT)


if __name__ == "__main__":
    main()
