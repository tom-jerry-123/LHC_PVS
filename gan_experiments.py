

import numpy as np
import keras
from autoencoder import Autoencoder
from diagnostic import load_selected_reconstructions_wrapper
from helpers import shuffle_data
from experiments import data_loading_wrapper
from gan_network import GanAutoencoder


def discriminator_experiment():
    """
    Post-hoc train a discriminator to distinguish between real data and reconstruction (PU only)
    Autoencoder reconstructions do not look like real vertex data at all so I expect very high accuracy from this
    Results: 0.9997 accuracy on PU, 0.975 on HS, i.e. autoencoder reconstructions looking nothing like real data
    :return:
    """
    model_path = "models/final_pt_model.keras"
    model = Autoencoder(input_dim=50, code_dim=3, architecture=(17,), regularization="L2")
    model.load_model(model_path)
    X_data, recs, is_tt, y_data = load_selected_reconstructions_wrapper(model)

    pu_orgs = X_data[y_data == 0]
    pu_recs = recs[y_data == 0]
    D_y = np.concatenate((np.zeros(len(pu_orgs)), np.ones(len(pu_recs))))
    D_X = np.vstack((pu_orgs, pu_recs))
    D_X, D_y = shuffle_data(D_X, D_y)
    split_idx = int(len(D_y) * 0.6)
    D_X_train = D_X[:split_idx]
    D_X_test = D_X[split_idx:]
    D_y_train = D_y[:split_idx]
    D_y_test = D_y[split_idx:]

    # create discriminator
    discriminator = keras.Sequential([
        keras.layers.Dense(32, activation='relu', input_shape=(50,), kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dense(1, activation='sigmoid', kernel_regularizer=keras.regularizers.l2(0.001))
    ])
    discriminator.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    discriminator.fit(
        D_X_train,
        D_y_train,
        epochs=10,
        batch_size=64,
        validation_split=0.2  # Use 20% of training data for validation
    )
    discriminator.evaluate(D_X_test, D_y_test)
    print(np.sum(D_y_test))


def gan_test():
    ttbar_train, ttbar_X, ttbar_y, ttbar_pt2, ttbar_reco_zs, ttbar_event_nums, ttbar_hs_zs = data_loading_wrapper(
        "data_batches/ttbar_small_500e.csv",
        "data_batches/ttbar_hs_truth_z.csv", 300)

    model = GanAutoencoder(input_dim=50, latent_dim=3)
    model.train(ttbar_train, epochs=10, batch_size=256)


if __name__ == "__main__":
    gan_test()