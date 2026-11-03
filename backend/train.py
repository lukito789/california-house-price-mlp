"""Latih MLP regresi. Jalankan:  python train.py"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers, regularizers

from preprocess import TARGET, TARGET_SCALE, fit_prep, transform

BASE = Path(__file__).parent
DATA = BASE.parent / "data" / "housing.csv"
ART = BASE / "artifacts"
ART.mkdir(exist_ok=True)

L2 = 1e-4
DROPOUT = 0.2
SEED = 42


def build_model(n_features: int) -> keras.Model:
    reg = regularizers.l2(L2)
    model = keras.Sequential([
        layers.Input(shape=(n_features,)),
        layers.Dense(128, activation="relu", kernel_regularizer=reg),
        layers.Dropout(DROPOUT),
        layers.Dense(64, activation="relu", kernel_regularizer=reg),
        layers.Dropout(DROPOUT),
        layers.Dense(32, activation="relu", kernel_regularizer=reg),
        layers.Dense(1, activation="linear"),  # output linear (regresi)
    ])
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    return model


def main(epochs: int = 200):
    if not DATA.exists():
        raise SystemExit(f"Dataset tidak ditemukan: {DATA}\n"
                         "Download housing.csv dari Kaggle dan taruh di folder data/.")
    df = pd.read_csv(DATA).dropna(subset=[TARGET, "ocean_proximity"])
    df = df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    n = len(df)
    n_tr, n_va = int(n * 0.70), int(n * 0.15)
    tr, va, te = df[:n_tr], df[n_tr:n_tr + n_va], df[n_tr + n_va:]

    prep = fit_prep(tr)
    Xtr, Xva, Xte = (transform(d, prep) for d in (tr, va, te))
    ytr, yva, yte = (d[TARGET].values.astype("float32") / TARGET_SCALE for d in (tr, va, te))

    keras.utils.set_random_seed(SEED)
    model = build_model(Xtr.shape[1])
    model.summary()
    hist = model.fit(
        Xtr, ytr, validation_data=(Xva, yva), epochs=epochs, batch_size=64, verbose=2,
        callbacks=[keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True),
                   keras.callbacks.ReduceLROnPlateau(patience=8, factor=0.5, min_lr=1e-5)],
    )

    pred = model.predict(Xte, verbose=0).ravel()
    mae = float(np.mean(np.abs(pred - yte)) * TARGET_SCALE)
    rmse = float(np.sqrt(np.mean((pred - yte) ** 2)) * TARGET_SCALE)
    r2 = float(1 - np.sum((yte - pred) ** 2) / np.sum((yte - yte.mean()) ** 2))

    model.save(ART / "model.keras")
    (ART / "prep.json").write_text(json.dumps(prep))
    (ART / "metrics.json").write_text(json.dumps({
        "test_mae_usd": mae, "test_rmse_usd": rmse, "test_r2": r2,
        "n_train": len(tr), "n_val": len(va), "n_test": len(te),
        "l2": L2, "dropout": DROPOUT,
        "history": {k: [float(x) for x in v] for k, v in hist.history.items()
                    if k in ("loss", "val_loss", "mae", "val_mae")},
    }))
    print(f"\nTEST  MAE=${mae:,.0f}  RMSE=${rmse:,.0f}  R2={r2:.3f}\nModel tersimpan di {ART}")


if __name__ == "__main__":
    main()
