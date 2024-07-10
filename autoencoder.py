"""
File for Autoencoder class. Creates simple autoencoder
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class Autoencoder:
    def __init__(self, input_dim=50, code_dim=3, architecture=(32,), regularization=None, masking=False):
        # Set regularization
        if regularization == "l2" or regularization == "L2":
            regularizer = tf.keras.regularizers.l2(0.005)
        elif regularization == "l1" or regularization == "L1":
            regularizer = tf.keras.regularizers.l1(0.005)
        else:
            regularizer = None

        # Define the input layer
        self._input_layer = tf.keras.layers.Input(shape=(input_dim,))

        # Define the encoder layers
        prev = self._input_layer
        cur = None
        for num in architecture:
            cur = tf.keras.layers.Dense(num, activation=tf.keras.layers.LeakyReLU(alpha=0.01), kernel_regularizer=regularizer)(prev)
            prev = cur
        # encoded = tf.keras.layers.Dense(8, activation='relu')(self._input_layer)
        self._encoding_layer = tf.keras.layers.Dense(code_dim, activation=tf.keras.layers.LeakyReLU(alpha=0.01), kernel_regularizer=regularizer)(cur)

        # Define the decoder layer
        prev = self._encoding_layer
        for i in range(len(architecture)-1, -1, -1):
            cur = tf.keras.layers.Dense(architecture[i], activation=tf.keras.layers.LeakyReLU(alpha=0.01), kernel_regularizer=regularizer)(prev)
            prev = cur
        # decoded = tf.keras.layers.Dense(8, activation='relu')(self._encoding_layer)
        decoded = tf.keras.layers.Dense(input_dim, activation='relu', kernel_regularizer=regularizer)(cur)

        # Now, time to compile model. Apply data masking if requested
        self._model = tf.keras.models.Model(self._input_layer, decoded)
        self._masking = masking
        if self._masking:
            self._model.compile(optimizer='adam', loss=MaskedMSE())
        else:
            self._model.compile(optimizer='adam', loss='mean_squared_error')

        # Save encoding layer of untrained model (for diagnostic purposes only)
        self._encoder = tf.keras.models.Model(self._input_layer, self._encoding_layer)

    def load_weights(self, file_path):
        self._model.load_weights(file_path)

    def train_model(self, x_train, batch_size=256, epochs=20, plot_valid_loss=False):
        x_train, x_valid = self._train_validate_split(x_train)
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            verbose=1,
            restore_best_weights=True)
        checkpoint_path = "models/model_checkpoint_{epoch:02d}.h5"  # Filepath with epoch number placeholder
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_weights_only=True,  # Set to True to save only model weights
            verbose=1)
        history = self._model.fit(
            x_train, x_train,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            validation_data=(x_valid, x_valid),
            callbacks=[early_stopping])
        self._encoder = tf.keras.models.Model(self._input_layer, self._encoding_layer)

        if plot_valid_loss:
            # Plot validation loss versus epoch
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Validation Loss')
            plt.title('Validation Loss vs. Epoch')
            plt.legend()
            plt.show()

    def print_weights(self):
        # Print weights and biases of each layer
        # Print weights and biases of each layer, skipping the input layer
        for i, layer in enumerate(self._model.layers[1:], start=1):
            weights, biases = layer.get_weights()
            print(f"Layer {i}: {layer.name}")
            print(f"Weights:\n{weights}")
            print(f"Biases:\n{biases}")
            print("-" * 30)

    def _train_validate_split(self, x_train):
        # Get the length of the original array
        num_data = len(x_train)

        # Generate indices for the 10% sample (randomly chosen without replacement)
        sample_indices = np.random.choice(num_data, size=int(num_data * 0.2), replace=False)

        # Create the 20% sample array
        x_valid = x_train[sample_indices]

        # Create the 80% array (excluding elements from the 10% sample)
        remainder_indices = np.setdiff1d(np.arange(num_data), sample_indices)
        new_x_train = x_train[remainder_indices]

        return new_x_train, x_valid

    def reconstruction_error(self, x_test):
        """
        Accepts 2d array of feature vectors as input, gets reconstruction error of each data point.
        Masks out padded features values, i.e. elements where original data is zero is ignored.
        :param x_test:
        :return:
        """
        reconstructions = self._model.predict(x_test)
        errors = np.sum((reconstructions - x_test) ** 2 * (~(x_test == 0)), axis=1)

        return errors

    def get_encodings(self, x_test):
        encodings = self._encoder.predict(x_test)
        return encodings


class MaskedMSE(tf.keras.losses.Loss):
    """
    Modified MSE loss function with padded feature values masked out.
    Use this for variable-input-length autoencoder that uses masking
    """
    def __init__(self, name="masked_mean_square_error"):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        # Assuming y_true and y_pred have the same shape
        mask = tf.cast(tf.not_equal(y_true, 0), dtype=tf.float32)  # Mask for non-zero values
        mse = tf.reduce_sum(tf.square(y_true - y_pred) * mask) / tf.maximum(tf.reduce_sum(mask), 1.0)
        return mse
