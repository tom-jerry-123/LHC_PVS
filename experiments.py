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
FEAT_FOLDER = "50_track_batches_old"
TTBAR_FEAT_FILE = "50_track_ttbar_pt_"
TTBAR_Z_FILE = "50_track_ttbar_z_"
VBF_PT_FILE = "50_track_vbf_pt_"
VBF_Z_FILE = "50_track_vbf_z_"

PT2_FOLDER = "sum_pt2_batches"
TTBAR_SUM_PT2 = "ttbar_sum_pt2_"
VBF_SUM_PT2 = "vbf_sum_pt2_"


TRAINING_BATCH_RANGE = (0, 8)
TESTING_BATCH_RANGE = (8, 15)
BATCH_SIZE = 500


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
    pu_err = pu_err[rand_pu_idxs]
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


def run_density_eff_test(autoencoder):
    """
    For our efficiency vs density plot, we assume bin size of 10 when doing density sorting
    :param model:
    :param file_path:
    :return:
    """

    # Load training, testing data
    ttbar_train, ttbar_x, ttbar_y = load_train_test(FEAT_FOLDER + "/" + TTBAR_FEAT_FILE, TRAINING_BATCH_RANGE, TESTING_BATCH_RANGE)
    # vbf_train, vbf_x, vbf_y = load_train_test(FEAT_FOLDER + "/" + VBF_PT_FILE, TRAINING_BATCH_RANGE, TESTING_BATCH_RANGE)
    # x_train = np.vstack((ttbar_train, vbf_train))

    # sum_log_pt = np.sum(np.log(ttbar_x + 0.01) * (ttbar_x > 0.1), axis=1)

    # Load data for pt2
    ttbar_pt2, _trash = load_data(PT2_FOLDER + "/" + TTBAR_SUM_PT2, TESTING_BATCH_RANGE)
    # vbf_pt2, _trash = load_data(PT2_FOLDER + "/" + VBF_SUM_PT2, TESTING_BATCH_RANGE)

    # Load z-coordinate data for recovertex
    ttbar_reco_z, _trash = load_data(FEAT_FOLDER + "/" + TTBAR_Z_FILE, TESTING_BATCH_RANGE)
    # vbf_reco_z, _trash = load_data(FEAT_FOLDER + "/" + VBF_Z_FILE, TESTING_BATCH_RANGE)
    # Load z-coordinate data for correct hard-scatter vertex
    ttbar_hs_truth_z = load_truth_hs_z(file_paths.ROOT_PATH, TESTING_BATCH_RANGE[0] * BATCH_SIZE, TESTING_BATCH_RANGE[1] * BATCH_SIZE)
    # vbf_hs_truth_z = load_truth_hs_z(file_paths.VBF_ROOT_PATH, TESTING_BATCH_RANGE[0] * BATCH_SIZE, TESTING_BATCH_RANGE[1] * BATCH_SIZE)

    if len(ttbar_hs_truth_z) == np.sum(ttbar_y == 1):
        print("Passed sanity check: extracted one z-coordinate for every TTBAR hs vertex")
    else:
        raise RuntimeError("Error: hs_truth_z not the same length as number of hs vertices for TTBAR!")

    # Train the autoencoder
    model = autoencoder
    model.train_model(ttbar_train, epochs=10, plot_valid_loss=True)
    ttbar_err = model.reconstruction_error(ttbar_x)
    # vbf_err = model.reconstruction_error(vbf_x)

    # Get efficiencies for TTBAR
    pt2_ttbar_eff, pt2_ttbar_midpts, pt2_ttbar_std = get_efficiencies_vs_density(ttbar_pt2, ttbar_y, ttbar_reco_z, ttbar_hs_truth_z, num_bins=10, plot_hist=False, algo_name="Sum-pt2")
    enc_ttbar_eff, enc_ttbar_midpts, enc_ttbar_std = get_efficiencies_vs_density(ttbar_err, ttbar_y, ttbar_reco_z, ttbar_hs_truth_z, plot_hist=False, algo_name="Autoencoder")

    # Get the efficiencies for VBF
    # pt2_vbf_eff, pt2_vbf_midpts, pt2_vbf_std = get_efficiencies_vs_density(vbf_pt2, vbf_y, vbf_reco_z, vbf_hs_truth_z, algo_name="Sum-pt2")
    # enc_vbf_eff, enc_vbf_midpts = get_efficiencies_vs_density(vbf_err, vbf_y, vbf_reco_z, vbf_hs_truth_z, plot_hist=False, algo_name="Autoencoder")

    # Plot the curves
    line_plot([pt2_ttbar_midpts, enc_ttbar_midpts], [pt2_ttbar_eff, enc_ttbar_eff], [pt2_ttbar_std, enc_ttbar_std], ['Sum-pt2', 'Sum-log-pt'], title="Efficiency vs. Density for TTBAR", xlabel="Vertex Density",
              ylabel="Efficiency")
    # line_plot([pt2_vbf_midpts], [pt2_vbf_eff], [pt2_vbf_std], labels=['Sum-pt2'], title="Efficiency vs. Density for VBF", xlabel="Vertex Density",
    #           ylabel="Efficiency", ylim=(0.7, 1.0))

    # Print Average Efficiency Scores
    print("\n*** Printing Recall Scores ***")
    print(f"{'':10} {'SUM-LOG-PT':10} {'SUM-PT2':10}")
    print(f"{'TTBAR':10} {np.mean(enc_ttbar_eff):<10.4f} {np.mean(pt2_ttbar_eff):<10.4f}")

    # separate pu and hs errors for plotting
    ttbar_mask = ttbar_y == 1
    pu_err = ttbar_err[~ttbar_mask]
    ttbar_hs_err = ttbar_err[ttbar_mask]
    pu_pt2 = ttbar_pt2[~ttbar_mask]
    hs_pt2 = ttbar_pt2[ttbar_mask]

    # Plot log errors
    # Random sample indices for pu vertices (we don't want to plot all of them, or else it'll be too cluttered)
    rand_pu_idxs = np.random.choice(len(pu_err), 5000, replace=False)
    pu_err = pu_err[rand_pu_idxs]
    pu_pt2 = pu_pt2[rand_pu_idxs]

    plot_log_reco_err(pu_err, ttbar_hs_err)
    plot_err_vs_pt2([pu_err, ttbar_hs_err], [pu_pt2, hs_pt2], ["PU", "HS"])


def quick_model_diagnostic(autoencoder: Autoencoder):
    # Light-weight testing for autoencoder
    # Load training, testing data
    ttbar_train, ttbar_x, ttbar_y = load_train_test("50_track_batches_old/50_track_ttbar_pt_", TRAINING_BATCH_RANGE, TESTING_BATCH_RANGE)


    model = autoencoder
    # model.train_model(ttbar_train, epochs=5, plot_valid_loss=True)
    model.save_weights("pt_massive_model.weights.h5")

    # print model weights
    # model.print_weights()

    errors = model.reconstruction_error(ttbar_x)
    y_mask = ttbar_y == 1
    hs_errs = errors[y_mask]
    pu_errs = errors[~y_mask]

    plot_log_reco_err(pu_errs, hs_errs)


"""
Section break
Code after this comment is all for the experiment
"""


def ttbar_data_loading_wrapper():
    """
    Wrapper function for loading ttbar, to keep things in run_experiment() (de facto main function) clean
    :return:
    """
    ttbar_data, ttbar_y = load_csv("data_batches/ttbar_small_500e.csv")
    # Note: after here, ttbar y is only for testing data
    ttbar_train_set, ttbar_test_set, ttbar_y = train_test_split(ttbar_data, ttbar_y, split_e_num=300,
                                                                remove_training_hs=True)
    # split into features
    pt_idxs = np.arange(0, 100, 2)
    ttbar_train = ttbar_train_set[:, pt_idxs]
    ttbar_X = ttbar_test_set[:, pt_idxs]
    ttbar_reco_zs = ttbar_test_set[:, 100]
    # Load truth zs
    ttbar_hs_zs = load_truth_hs_z(file_paths.ROOT_PATH, 300, 500)
    # for now, we'll calculate pt2
    ttbar_pt2 = np.sum(ttbar_X ** 2, axis=1)
    # Do sanity checks
    sanity_checks(ttbar_y, ttbar_hs_zs)

    return ttbar_train, ttbar_X, ttbar_y, ttbar_pt2, ttbar_reco_zs, ttbar_hs_zs


def sanity_checks(y, hs_zs):
    assert np.sum(y == 1) == len(hs_zs)

def evaluate_results(pt2, errs, y, reco_zs, hs_zs, model_name="model", dataset_name="Data"):
    # Get efficiencies vs density
    pt2_eff, pt2_midpts, pt2_std = get_efficiencies_vs_density(pt2, y, reco_zs, hs_zs,
                                                               num_bins=10, plot_hist=False, algo_name="Sum-pt2")
    model_eff, model_midpts, model_std = get_efficiencies_vs_density(errs, y, reco_zs, hs_zs,
                                                                     num_bins=10, plot_hist=False, algo_name=model_name)

    # Plot the efficiencies
    line_plot([pt2_midpts, model_midpts], [pt2_eff, model_eff], [pt2_std, model_std],
              ['Sum-pt2', model_name], title="Efficiency vs. Density for "+dataset_name, xlabel="Vertex Density",
              ylabel="Efficiency")

    # Print Average Efficiency Scores
    print("\n*** Printing Recall Scores ***")
    print(f"{'':10} {model_name:10} {'SUM-PT2':10}")
    print(f"{dataset_name:10} {np.mean(model_eff):<10.4f} {np.mean(pt2_eff):<10.4f}")

    # Now, plot the log reco errors and err vs pt2 plots
    # separate pu and hs errors for plotting
    hs_mask = y == 1
    pu_err = errs[~hs_mask]
    hs_err = errs[hs_mask]
    pu_pt2 = pt2[~hs_mask]
    hs_pt2 = pt2[hs_mask]

    # Plot log errors
    # Random sample indices for pu vertices (we don't want to plot all of them, or else it'll be too cluttered)
    rand_pu_idxs = np.random.choice(len(pu_err), 10000, replace=False)
    pu_err = pu_err[rand_pu_idxs]
    pu_pt2 = pu_pt2[rand_pu_idxs]

    plot_log_reco_err(pu_err, hs_err)
    plot_err_vs_pt2([pu_err, hs_err], [pu_pt2, hs_pt2], ["PU", "HS"])


def run_experiment():
    """
    Main function for running the experiments
    :return:
    """
    # *** Loading Data ***
    ttbar_train, ttbar_X, ttbar_y, ttbar_pt2, ttbar_reco_zs, ttbar_hs_zs = ttbar_data_loading_wrapper()
    # vbf_train, vbf_X, vbf_y, vbf_pt2, vbf_reco_zs, vbf_hs_zs = vbf_data_loading_wrapper()

    # *** Training / saving / loading model, then doing inference ***
    model = Autoencoder(input_dim=50, code_dim=3, architecture=(17,), regularization="L2")
    model.train_model(ttbar_train, epochs=10, plot_valid_loss=False)
    # model.save("models/final_pt_model.keras")
    ttbar_errs = model.reconstruction_error(ttbar_X)

    # *** Evaluating Model ***
    evaluate_results(ttbar_pt2, ttbar_errs, ttbar_y, ttbar_reco_zs, ttbar_hs_zs, model_name="Autoencoder", dataset_name="TTBAR")


if __name__ == "__main__":
    """
    Architectures:
    pt, 1 by dR: input dim 100, hidden (58, 7)
    pt: input dim 50, hidden (17,)
    """
    run_experiment()
