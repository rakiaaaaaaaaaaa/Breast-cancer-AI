import argparse, os, json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
from tensorflow import keras

from .utils import load_wdbc_csv_or_sklearn, preprocess
from .models import build_autoencoder, build_cnn_classifier

def plot_history(hist, outpath):
    plt.figure()
    plt.plot(hist.history.get("loss", []), label="train_loss")
    plt.plot(hist.history.get("val_loss", []), label="val_loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training History")
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()

def main(args):
    data_path = args.data_path if args.data_path else "data/data.csv"
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_wdbc_csv_or_sklearn(data_path)
    X, y = preprocess(df)

    # Split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # Reshape for 1D CNN
    X_train_c = X_train_s[..., None]
    X_val_c = X_val_s[..., None]
    X_test_c = X_test_s[..., None]

    # Class weights
    classes = np.unique(y_train)
    class_weights = dict(zip(
        classes,
        compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    ))

    # Autoencoder pretraining
    ae, enc = build_autoencoder(X_train_c.shape[1], latent_dim=args.latent_dim)
    ae_hist = ae.fit(
        X_train_c, X_train_c,
        validation_data=(X_val_c, X_val_c),
        epochs=args.ae_epochs, batch_size=args.batch_size, verbose=2
    )
    ae.save(outdir / "autoencoder.keras")
    plot_history(ae_hist, outdir / "autoencoder_training.png")

    # Transfer learning: frozen encoder
    clf = build_cnn_classifier(X_train_c.shape[1], encoder=enc, freeze_encoder=True)
    callbacks = [keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True, monitor="val_auc", mode="max")]
    hist = clf.fit(
        X_train_c, y_train, validation_data=(X_val_c, y_val),
        epochs=args.epochs, batch_size=args.batch_size, verbose=2,
        class_weight=class_weights, callbacks=callbacks
    )
    clf.save(outdir / "cnn_frozen.keras")
    plot_history(hist, outdir / "cnn_frozen_training.png")

    # Fine-tune: unfreeze encoder layers
    for layer in clf.layers:
        layer.trainable = True
    clf.compile(optimizer=keras.optimizers.Adam(1e-4),
                loss="binary_crossentropy",
                metrics=[keras.metrics.AUC(name="auc"), "accuracy"])
    hist_ft = clf.fit(
        X_train_c, y_train, validation_data=(X_val_c, y_val),
        epochs=max(10, args.epochs//2), batch_size=args.batch_size, verbose=2,
        class_weight=class_weights, callbacks=callbacks
    )
    clf.save(outdir / "cnn_finetuned.keras")
    plot_history(hist_ft, outdir / "cnn_finetuned_training.png")

    # Evaluate
    y_proba = clf.predict(X_test_c).ravel()
    y_pred = (y_proba >= 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    roc = roc_auc_score(y_test, y_proba)
    prc = average_precision_score(y_test, y_proba)

    # Save metrics
    metrics = {"roc_auc": float(roc), "avg_precision": float(prc),
               "confusion_matrix": cm.tolist(), "classification_report": report}
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # Plot ROC & PR
    from sklearn.metrics import roc_curve, precision_recall_curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    prec, rec, _ = precision_recall_curve(y_test, y_proba)

    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0,1],[0,1], linestyle="--")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.title("ROC Curve")
    plt.savefig(outdir / "roc_curve.png", bbox_inches="tight"); plt.close()

    plt.figure()
    plt.plot(rec, prec)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall Curve")
    plt.savefig(outdir / "pr_curve.png", bbox_inches="tight"); plt.close()

    # Save scaler
    import joblib
    joblib.dump(scaler, outdir / "scaler.joblib")

    print("Done. Metrics:", metrics)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, default=None, help="Path to Kaggle WDBC CSV (data.csv). If missing, falls back to sklearn dataset.")
    p.add_argument("--outdir", type=str, default="outputs", help="Directory to save models and plots.")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--ae_epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--latent_dim", type=int, default=16)
    args = p.parse_args()
    main(args)
