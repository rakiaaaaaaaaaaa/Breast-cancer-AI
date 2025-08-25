from typing import Tuple, Dict, Any, Optional
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_autoencoder(n_features: int, latent_dim: int = 16) -> Tuple[keras.Model, keras.Model]:
    inp = keras.Input(shape=(n_features, 1))
    x = layers.Conv1D(32, 3, padding="same", activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(64, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling1D()(x)
    latent = layers.Dense(latent_dim, activation="relu", name="latent")(x)

    # Decoder
    x = layers.Dense((n_features//2)*32, activation="relu")(latent)
    x = layers.Reshape((n_features//2, 32))(x)
    x = layers.UpSampling1D(2)(x)
    x = layers.Conv1D(32, 3, padding="same", activation="relu")(x)
    out = layers.Conv1D(1, 3, padding="same")(x)

    autoencoder = keras.Model(inp, out, name="autoencoder")
    encoder = keras.Model(inp, latent, name="encoder")
    autoencoder.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return autoencoder, encoder

def build_cnn_classifier(n_features: int, encoder: Optional[keras.Model] = None, freeze_encoder: bool = True) -> keras.Model:
    inp = keras.Input(shape=(n_features, 1))
    if encoder is not None:
        # Rebuild encoder layers onto inp
        x = inp
        for layer in encoder.layers[1:]:  # skip InputLayer
            layer._name = layer.name + "_tl"
            x = layer(x)
        # Small head
        x = layers.Dense(32, activation="relu")(x)
    else:
        x = layers.Conv1D(32, 3, padding="same", activation="relu")(inp)
        x = layers.BatchNormalization()(x)
        x = layers.Conv1D(64, 3, padding="same", activation="relu")(x)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(64, activation="relu")(x)

    x = layers.Dropout(0.2)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inp, out, name="cnn_classifier")

    if encoder is not None and freeze_encoder:
        for layer in model.layers:
            if layer.name.endswith("_tl"):
                layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc"), "accuracy"]
    )
    return model
