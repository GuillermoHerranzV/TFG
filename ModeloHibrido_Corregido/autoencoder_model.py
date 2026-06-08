"""Autoencoder convolucional con reduccion final por resize y flatten para la entrada al circuito"""

import tensorflow as tf


def build_autoencoder(input_shape, target_hw, learning_rate: float = 1e-3):
    th, tw = target_hw

    encoder = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),
            # Padding para que la reduccion quede en numeros pares y al reconstruir no haya problemas de redondeo
            tf.keras.layers.ZeroPadding2D(2),  # 28 -> 32
            tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D(pool_size=2, padding="same"),  # 16
            tf.keras.layers.Conv2D(16, 3, activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D(pool_size=2, padding="same"),  # 8
            tf.keras.layers.Conv2D(1, 3, activation="relu", padding="same"),
            tf.keras.layers.MaxPooling2D(pool_size=2, padding="same"),  # 4x4x1
            # Reduccion final para alcanzar el numero de qubits deseado
            tf.keras.layers.Lambda(lambda x: tf.image.resize(x, (th, tw))),
            # Flatten para la entrada al circuito
            tf.keras.layers.Flatten(name="latent_space"),
        ],
        name=f"encoder_{th}x{tw}",
    )

    decoder = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(th * tw,)),
            tf.keras.layers.Reshape((th, tw, 1)),
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
