"""Autoencoder para compresion antes del pipeline cuantico"""

import tensorflow as tf


def build_autoencoder(input_dim: int, latent_dim: int, learning_rate: float = 1e-3):
    encoder = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(latent_dim, activation="linear", name="latent_space"),
        ],
        name=f"encoder_{latent_dim}",
    )

    decoder = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(latent_dim,)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(input_dim, activation="sigmoid"),
        ],
        name=f"decoder_{latent_dim}",
    )

    autoencoder_input = tf.keras.Input(shape=(input_dim,), name="autoencoder_input")
    encoded = encoder(autoencoder_input)
    reconstructed = decoder(encoded)
    autoencoder = tf.keras.Model(autoencoder_input, reconstructed, name=f"autoencoder_{latent_dim}")
    autoencoder.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
    )
    return autoencoder, encoder
