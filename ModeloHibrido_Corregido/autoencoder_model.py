"""Autoencoder convolucional con reduccion final por resize.

La reduccion se hace al maximo con capas Conv + MaxPooling (28 -> 32 por padding,
y luego 32 -> 16 -> 8 -> 4) y solo el ultimo paso, de 4x4 a la rejilla objetivo
(target_hw), se hace con una capa Resizing (interpolacion). El latente resultante
es target_hw[0] x target_hw[1] x 1, cuyo numero de celdas = nº de qubits.

El encoder es identico al del modelo clasico; la unica diferencia entre ambos
modelos es el bloque clasificador.
"""

import tensorflow as tf


def build_autoencoder(input_shape, target_hw, learning_rate: float = 1e-3):
    th, tw = target_hw

    encoder = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.ZeroPadding2D(2),  # 28 -> 32
            tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D(pool_size=2, padding="same"),  # 16
            tf.keras.layers.Conv2D(16, 3, activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D(pool_size=2, padding="same"),  # 8
            tf.keras.layers.Conv2D(1, 3, activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D(pool_size=2, padding="same"),  # 4x4x1
            tf.keras.layers.Lambda(lambda x: tf.image.resize(x, (th, tw))),
            tf.keras.layers.Flatten(name="latent_space"),  # mapa (H,W,1) -> vector (H*W,) = n_qubits
        ],
        name=f"encoder_{th}x{tw}",
    )

    decoder = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(th * tw,)),
            tf.keras.layers.Reshape((th, tw, 1)),  # vector (H*W,) -> mapa (H,W,1)
            tf.keras.layers.Lambda(lambda x: tf.image.resize(x, (4, 4))),
            tf.keras.layers.Conv2D(16, 3, activation="relu", padding="same"),
            tf.keras.layers.UpSampling2D(size=2),  # 8
            tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
            tf.keras.layers.UpSampling2D(size=2),  # 16
            tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
            tf.keras.layers.UpSampling2D(size=2),  # 32
            tf.keras.layers.Cropping2D(2),  # 28
            tf.keras.layers.Conv2D(1, 3, activation="sigmoid", padding="same"),
        ],
        name=f"decoder_{th}x{tw}",
    )

    autoencoder_input = tf.keras.Input(shape=input_shape, name="autoencoder_input")
    reconstructed = decoder(encoder(autoencoder_input))
    autoencoder = tf.keras.Model(autoencoder_input, reconstructed, name=f"autoencoder_{th}x{tw}")
    autoencoder.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
    )
    return autoencoder, encoder
