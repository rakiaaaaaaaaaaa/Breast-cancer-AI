# Breast Cancer Classification using CNN with Transfer Learning (Autoencoder)

This project builds a **1D Convolutional Neural Network (CNN)** classifier on the **Wisconsin Diagnostic Breast Cancer (WDBC)** tabular dataset, and applies **transfer learning via an unsupervised autoencoder**:
- First, an autoencoder learns a compact representation of the 30 input features (unsupervised).
- Then, the **encoder** is transferred into a supervised CNN classifier (frozen → fine-tuned).
- We compare results with strong tabular baselines (Logistic Regression, Random Forest, Gradient Boosting).

> Why CNN on tabular?  
> The 30 engineered features (means, standard errors, worst) have local relationships (e.g., radii/perimeter/area groups). A 1D CNN can capture local interactions in the ordered feature vector. The autoencoder pretraining helps the CNN start from a structure-aware initialization—**a form of transfer learning** appropriate for tabular data (where ImageNet-style TL isn't applicable).

---

## Dataset


---
