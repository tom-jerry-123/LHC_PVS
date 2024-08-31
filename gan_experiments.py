

import numpy as np
import keras

import gan_network
from autoencoder import Autoencoder
from diagnostic import load_selected_reconstructions_wrapper
from helpers import shuffle_data
from experiments import data_loading_wrapper, evaluate_results
from gan_network import GanNetwork


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

    print("\n\n*** Evaluating Critic Network ***\n")
    discriminator.evaluate(D_X_test, D_y_test)
    print(np.sum(D_y_test))


def quick_gan_test():
    ttbar_train, ttbar_X, ttbar_y, ttbar_pt2, ttbar_reco_zs, ttbar_event_nums, ttbar_hs_zs = data_loading_wrapper(
        "data_batches/ttbar_small_500e.csv",
        "data_batches/ttbar_hs_truth_z.csv", 300)

    model = GanNetwork(input_dim=50, latent_dim=5)
    model.compile(keras.optimizers.Adam(1e-4), keras.optimizers.Adam(1e-4))
    model.fit(ttbar_train, ttbar_train, epochs=10, batch_size=256)

    rec_errs = model.get_reconstruction_errors(ttbar_X)
    print(rec_errs)


def run_gan_experiment():
    # *** Loading Data ***
    ttbar_train, ttbar_X, ttbar_y, ttbar_pt2, ttbar_reco_zs, ttbar_event_nums, ttbar_hs_zs = data_loading_wrapper(
        "data_batches/ttbar_big_7500e.csv",
        "data_batches/ttbar_hs_truth_z.csv", 3500)
    vbf_train, vbf_X, vbf_y, vbf_pt2, vbf_reco_zs, vbf_event_nums, vbf_hs_zs = data_loading_wrapper(
        "data_batches/vbf_big_7500e.csv", "data_batches/vbf_hs_truth_z.csv", 3500)
    X_train = np.vstack((ttbar_train, vbf_train))

    # *** Training / saving / loading model, then doing inference ***
    # model = GanNetwork(input_dim=50, latent_dim=5)
    # model.compile(d_optimizer=keras.optimizers.Adam(1e-4), a_optimizer=keras.optimizers.Adam(1e-4))
    # model.fit(X_train, X_train, epochs=20, batch_size=256)
    # model.save_model("models/gan_full_5d_code")
    model = GanNetwork.load_model("models/gan_full_5d_code")
    ttbar_scores = model.get_discriminator_predictions(ttbar_X)
    vbf_scores = model.get_discriminator_predictions(vbf_X)

    # *** Evaluating Model ***
    evaluate_results(ttbar_pt2, ttbar_scores, ttbar_y, ttbar_reco_zs, ttbar_hs_zs, model_name="Critic",
                     dataset_name="TTBAR")
    evaluate_results(vbf_pt2, vbf_scores, vbf_y, vbf_reco_zs, vbf_hs_zs, model_name="Critic", dataset_name="VBF")


if __name__ == "__main__":
    # run_gan_experiment()
    discriminator_experiment()