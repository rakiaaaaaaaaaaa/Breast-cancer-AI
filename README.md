# Breast Cancer Classification using CNN with Transfer Learning (Autoencoder)

This project builds a **1D Convolutional Neural Network (CNN)** classifier on the **Wisconsin Diagnostic Breast Cancer (WDBC)** tabular dataset, and applies **transfer learning via an unsupervised autoencoder**:
- First, an autoencoder learns a compact representation of the 30 input features (unsupervised).
- Then, the **encoder** is transferred into a supervised CNN classifier (frozen → fine-tuned).
- We compare results with strong tabular baselines (Logistic Regression, Random Forest, Gradient Boosting).

> Why CNN on tabular?  
> The 30 engineered features (means, standard errors, worst) have local relationships (e.g., radii/perimeter/area groups). A 1D CNN can capture local interactions in the ordered feature vector. The autoencoder pretraining helps the CNN start from a structure-aware initialization—**a form of transfer learning** appropriate for tabular data (where ImageNet-style TL isn't applicable).

---

## Dataset

- Expected file: `data/data.csv` in Kaggle's *breast-cancer-wisconsin-data* format (a.k.a. WDBC), columns include:
  - `id`, `diagnosis` (M/B), 30 numeric features like `radius_mean`, `texture_mean`, ..., and a trailing `Unnamed: 32` (NaN).
- If `data/data.csv` is **not found**, the notebook will **fall back** to `sklearn.datasets.load_breast_cancer()` and auto-construct the same schema.

---

## Quickstart

1. Create and activate a Python 3.10+ environment.
2. Install deps:

```bash
pip install -r requirements.txt
```

3. Launch the notebook:

```bash
jupyter lab notebooks/breast_cancer_cnn_transfer.ipynb
```

4. Optional: Use the script workflow:

```bash
python -m src.train --data_path data/data.csv --epochs 50 --batch_size 32
```

Artifacts (trained models, plots, metrics) are saved under `outputs/`.

---

## What you get

- **EDA & preprocessing** (drop ID & NaNs, encode `diagnosis`, scale features).
- **Train/Val/Test split** with stratification and **class weights**.
- **Baselines:** Logistic Regression, Random Forest, Gradient Boosting.
- **Autoencoder pretraining** → transfer the encoder to a **1D CNN classifier**.
- **Training curves**, **confusion matrix**, **ROC/PR curves**, **classification report**.
- **Model saving** (`.keras`/`.h5`) + **inference snippet**.
- **Reproducibility:** fixed seeds, version printouts.

---

## Requirements

See `requirements.txt`. If TensorFlow is heavy for your environment, you can switch to CPU-only install.
