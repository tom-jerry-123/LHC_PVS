"""
Main file for running models and experiments
"""

import matplotlib.pyplot as plt
import numpy as np

from autoencoder import Autoencoder
from data_loading import *
from plotting import *
from helpers import *
import file_paths
from vertex_density_calc import density_from_z_coord


"""
Variables to use
"""
FOLDER_10 = "10_track_batches"
PT_FILE_10 = "10_track_pt_batch_"
FOLDER_25 = "25_track_batches"
PT_FILE_25 = "ttbar_pt_batch_"
FOLDER_50 = "50_track_batches"
PT_FILE_50 = "50_track_batch_"

PT2_FOLDER = "sum_pt2_batches"
SUM_PT_FILE = "ttbar_sum_pt2_batch_"
TRAINING_BATCH_RANGE = (0, 2)
TESTING_BATCH_RANGE = (2, 3)


def run_rec_err_model(autoencoder, folder, file):
    """
    Trains an autoencoder model and tests it using data from folder and files
    :param autoencoder: autoencoder neural net to use
    :param folder: folder of data file. if it exists
    :param file: file prefix (assuming files are numbered at the end to indicate batch number)
    :return:
    """
    # Load training, testing data
    ttbar_train, ttbar_x, ttbar_y = load_train_test(folder + "/" + file, TRAINING_BATCH_RANGE, TESTING_BATCH_RANGE)
    ttbar_pt2, ttbar_y = load_data(PT2_FOLDER + "/" + SUM_PT_FILE, TESTING_BATCH_RANGE)

    last = 1
    test_thresholds = [last + i * 0.2 for i in range(0, 500)]

    last = 5
    pt_thresholds = [last + i for i in range(200)]

    model = autoencoder
    model.train_model(ttbar_train, epochs=30, plot_valid_loss=False)

    # Load weights of saved model
    # model.load_weights("models/final_model.h5")

    # Compute reconstruction errors
    rec_err = model.reconstruction_error(ttbar_x)

    # # Compute encodings, and plot them
    # encodings = model.get_encodings(ttbar_x)
    # mask = ttbar_y == 1
    # hs_encodings = encodings[mask]
    # pu_encodings = encodings[~mask]
    # plot_encodings([pu_encodings, hs_encodings], ["PU", "HS"])

    # Get classifications
    ttbar_encoder_yhat = get_classification(rec_err, ttbar_y)
    ttbar_base_yhat = get_classification(ttbar_pt2, ttbar_y)

    # Print Recall Values
    ttbar_encoder_recall = np.sum((ttbar_encoder_yhat == 1) & (ttbar_y == 1)) / np.sum((ttbar_y == 1))
    ttbar_base_recall = np.sum((ttbar_base_yhat == 1) & (ttbar_y == 1)) / np.sum((ttbar_y == 1))

    # Print Recall Scores
    print("\n*** Printing Recall Scores ***")
    print(f"{'':10} {'ENCODER':10} {'SUM-PT2':10}")
    # print(f"{'TTBAR':10} {ttbar_encoder_recall:<10.4f} {ttbar_base_recall:<10.4f}")
    print(f"{'TTBAR':10} {ttbar_encoder_recall:<10.4f} N/A")

    # Separate pu, vbf_hs, ttbar_hs data for roc curves
    # Do this for errors first
    ttbar_mask = ttbar_y == 1
    pu_err = rec_err[~ttbar_mask]
    ttbar_hs_err = rec_err[ttbar_mask]
    # # Now, doing this for sum-pt2 values
    # pu_pt2 = ttbar_pt2[~ttbar_mask]
    # ttbar_hs_pt2 = ttbar_pt2[ttbar_mask]

    # Plot log errors
    # Random sample indices for pu vertices (we don't want to plot all of them, or else it'll be too cluttered)
    rand_pu_idxs = np.random.choice(len(pu_err), int(len(pu_err) * 0.05), replace=False)
    plot_log_reco_err(pu_err, ttbar_hs_err)
    # plot err vs pt2
    # plot_err_vs_pt2(err_lst=[pu_err[rand_pu_idxs], ttbar_hs_err],
    #                 pt_lst=[pu_pt2[rand_pu_idxs], ttbar_hs_pt2], labels=['PU', 'TTBAR HS', 'VBF HS'])

    # # Collect Mistakes
    # collect_mistakes(vbf_pt2, vbf_encoder_yhat, vbf_y, "vbf_correction_list.csv")
    # collect_mistakes(ttbar_pt2, ttbar_encoder_yhat, ttbar_y, "ttbar_correction_list.csv")
    #
    # Get tp / fp rates for model
    # ttbar_tp, ttbar_fp = get_fp_tp(test_thresholds, pu_err, ttbar_hs_err)
    # Get tp / fp rates for baseline
    # ttbar_baseline_tp, ttbar_baseline_fp = get_fp_tp(pt_thresholds, pu_pt2, ttbar_hs_pt2)

    # headers = ['tpr', 'fpr']
    # np.savetxt("fpr_tpr/tpr_fpr_ttbar_decoder.csv", np.vstack((headers, np.column_stack((ttbar_tp, ttbar_fp)))),
    #            fmt='%s', delimiter=',')
    # np.savetxt("fpr_tpr/tpr_fpr_ttbar_sum-pt2.csv",
    #            np.vstack((headers, np.column_stack((ttbar_baseline_tp, ttbar_baseline_fp)))), fmt='%s', delimiter=',')

    # Plot the ROC curves
    # plot_roc_curve(tp_lst=[ttbar_baseline_tp, ttbar_tp],
    #                fp_lst=[ttbar_baseline_fp, ttbar_fp],
    #                labels=["Sum pt2", "Decoder"], title="TTBAR ROC Curves")


def run_density_eff_test(autoencoder, file_path):
    """
    For our efficiency vs density plot, we assume bin size of 10 when doing density sorting
    :param model:
    :param file_path:
    :return:
    """
    BIN_SIZE = 10
    BIN_RANGE = (-120, 120)
    ERR_THRESHOLD = 80

    # Load training, testing data
    ttbar_train, ttbar_x, ttbar_y = load_train_test(file_path, TRAINING_BATCH_RANGE, TESTING_BATCH_RANGE)
    # Load data for pt2
    ttbar_pt2, _trash = load_data(PT2_FOLDER + "/" + SUM_PT_FILE, TESTING_BATCH_RANGE)
    # Load z-coordinate data for recovertex
    ttbar_reco_z, _trash = load_data("50_track_batches/50_track_z_coord_batch_", TESTING_BATCH_RANGE)
    # Load z-coordinate data for correct hard-scatter vertex
    hs_truth_z = load_truth_hs_z(file_paths.ROOT_PATH, 1000, 1500)

    if len(hs_truth_z) == np.sum(ttbar_y == 1):
        print("Passed sanity check: extracted one z-coordinate for every hs vertex")
    else:
        raise RuntimeError("Error: hs_truth_z not the same length as number of hs vertices!")

    # Get classifications of sum-pt2
    ttbar_base_yhat = get_classification(ttbar_pt2, ttbar_y)

    # Get histogram of densities. First, densities of selected vertices
    base_selected_zs = ttbar_reco_z[ttbar_base_yhat.astype(bool)]
    base_densities = density_from_z_coord(base_selected_zs)  # densities from selected vertex
    base_selected_freq, base_bin_edges = np.histogram(base_densities, bins=20)
    plot_histogram(base_densities, num_bins=20, title="Density Histograms for Selected Vertices Using Sum-pt2", x_label="Density")
    # Now, densities of correctly selected vertices, i.e. true positives
    base_tp_zs = base_selected_zs[abs(base_selected_zs - hs_truth_z) < 1]
    base_tp_densities = density_from_z_coord(base_tp_zs)
    base_tp_freq, base_bin_edges = np.histogram(base_tp_densities, bins=base_bin_edges)
    plot_histogram(base_tp_densities, num_bins=20, title="Density Histogram for Successful Selection, Sum-pt2", x_label="Density")

    # Calculate efficiencies, then plot
    efficiencies = base_tp_freq / base_selected_freq
    midpoints = (base_bin_edges[:-1] + base_bin_edges[1:]) / 2
    line_plot(midpoints, efficiencies, title="Efficiency vs. Density, Sum-pt2", xlabel="Vertex Density",
              ylabel="Efficiency")

    # Now, do the density hists for the autoencoder.
    model = autoencoder
    model.train_model(ttbar_train, epochs=30, plot_valid_loss=False)
    ttbar_enc_yhat = get_classification(model.reconstruction_error(ttbar_x), ttbar_y)
    # Get hist. of densities
    enc_selected_zs = ttbar_reco_z[ttbar_enc_yhat.astype(bool)]
    enc_densities = density_from_z_coord(enc_selected_zs)
    enc_selected_freq, enc_bin_edges = np.histogram(enc_densities, bins=20)
    plot_histogram(enc_densities, num_bins=20, title="Density Histograms for Selected Vertices Using Autoencoder",
                   x_label="Density")
    # Now, densities of correctly selected vertices, i.e. true positives
    enc_tp_zs = enc_selected_zs[abs(enc_selected_zs - hs_truth_z) < 1]
    enc_tp_densities = density_from_z_coord(enc_tp_zs)
    enc_tp_freq, enc_bin_edges = np.histogram(enc_tp_densities, bins=enc_bin_edges)
    plot_histogram(enc_tp_densities, num_bins=enc_bin_edges, title="Density Histogram for Successful Selection, Autoencoder",
                   x_label="Density")

    enc_efficiencies = enc_tp_freq / enc_selected_freq
    midpoints = (enc_bin_edges[:-1] + enc_bin_edges[1:]) / 2
    line_plot(midpoints, enc_efficiencies, title="Efficiency vs. Density, Autoencoder", xlabel="Vertex Density",
              ylabel="Efficiency")


def quick_model_diagnostic(autoencoder: Autoencoder, file_path):
    # Load training, testing data
    ttbar_train, ttbar_x, ttbar_y = load_train_test(file_path, TRAINING_BATCH_RANGE, TESTING_BATCH_RANGE)
    # Remove z-coordinates
    ttbar_train = ttbar_train[:, :-1]
    ttbar_x = ttbar_x[:, :-1]

    model = autoencoder
    model.train_model(ttbar_train, epochs=10, plot_valid_loss=True)

    # print model weights
    model.print_weights()

    errors = model.reconstruction_error(ttbar_x)
    y_mask = ttbar_y == 1
    hs_errs = errors[y_mask]
    pu_errs = errors[~y_mask]

    plot_reco_err(pu_errs, hs_errs)
    plot_log_reco_err(pu_errs, hs_errs)


if __name__ == "__main__":
    run_density_eff_test(Autoencoder(input_dim=50, code_dim=3, architecture=(8, 5), regularization="L2"), FOLDER_50 + "/" + PT_FILE_50)
