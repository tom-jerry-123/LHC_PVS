"""
Main file for running models and experiments
"""
import matplotlib.pyplot as plt
import numpy as np

from autoencoder import Autoencoder
from data_loading import *
from plotting import *
from helpers import *


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
    # ttbar_pt2, ttbar_y = load_data(PT2_FOLDER + "/" + SUM_PT_FILE, TESTING_BATCH_RANGE)
    # Remove z-coordinates
    ttbar_train = ttbar_train[:, :-1]
    ttbar_x = ttbar_x[:, :-1]

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
    # ttbar_base_yhat = get_classification(ttbar_pt2, ttbar_y)

    # Print Recall Values
    ttbar_encoder_recall = np.sum((ttbar_encoder_yhat == 1) & (ttbar_y == 1)) / np.sum((ttbar_y == 1))
    # ttbar_base_recall = np.sum((ttbar_base_yhat == 1) & (ttbar_y == 1)) / np.sum((ttbar_y == 1))

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
    # NEED TO REDO THE SUM-PT2 BATCHES
    # ttbar_pt2,  = load_data(PT2_FOLDER + "/" + SUM_PT_FILE, TESTING_BATCH_RANGE)

    # Remove the z-coordinate column from training data; remove and store z-coordinates in testing data
    ttbar_train = ttbar_train[:, :-1]
    ttbar_z_coord = ttbar_x[:, -1]
    ttbar_x = ttbar_x[:, :-1]

    # Do density bin sorting. First get indices of the events
    split_idxs = list(np.where(ttbar_y)[0])
    split_idxs.append(len(ttbar_y))
    split_idxs = np.array(split_idxs)
    sorted_test_data = {i: [] for i in range(1, 11)}
    # Now, loop through each event to do the sorting
    for i in range(1, len(split_idxs)):
        start = split_idxs[i-1]
        end = split_idxs[i]
        event_z = ttbar_z_coord[start:end]
        # Assuming bin size of 10
        edges = np.arange(BIN_RANGE[0], BIN_RANGE[1] + BIN_SIZE, BIN_SIZE)
        counts, edges = np.histogram(event_z, bins=edges)
        # Add value of 1 as dummy value: all values outside edges range set to 1
        counts = np.concatenate((counts, np.array([1])))
        # get which bin each z-coordinate is int
        bin_idxs = np.digitize(event_z, edges) - 1
        bin_idxs[bin_idxs < 0] = len(counts) - 1
        # compute densities based on the bins
        density_bracket = counts[bin_idxs]
        density_bracket[density_bracket > 10] = 10
        event_test = np.column_stack((ttbar_x[start:end], ttbar_y[start:end]))
        for num in range(1, 11):
            vertex_idxs = np.where(density_bracket == num)[0]
            bracket_test_data = event_test[vertex_idxs]
            sorted_test_data[num].append(bracket_test_data)
    print("Finished sorting. Stacking each bin into one array.")
    for num in range(1, 11):
        sorted_test_data[num] = np.vstack(sorted_test_data[num])

    # Train Autoencoder
    model = autoencoder
    model.train_model(ttbar_train, epochs=10, plot_valid_loss=True)

    # Test the model
    efficiencies = {i: 0.0 for i in range(1, 11)}
    for num in range(1, 11):
        cur_x = sorted_test_data[num][:, :-1]
        cur_y = sorted_test_data[num][:, -1]
        errs = model.reconstruction_error(cur_x)
        predictions = errs > ERR_THRESHOLD
        efficiency = np.sum(predictions & (cur_y == 1)) / np.sum(cur_y == 1)
        print(f"Efficiency for bracket {num}: {efficiency}")
        efficiencies[num] = efficiency

    # Plot efficiencies
    point_pairs = list(efficiencies.items())
    densities = [point_pairs[i][0]/10 for i in range(10)]
    recall_rates = [point_pairs[i][1] for i in range(10)]
    plt.plot(densities, recall_rates)
    plt.xlabel("Vertex Density")
    plt.ylabel("Efficiency")
    plt.title("HS Vertex Selection Efficiency vs Pileup Density")
    plt.show()


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
    # # 10 tracks, compact neural net
    # run_rec_err_model(Autoencoder(input_dim=10, code_dim=3, architecture=(8, 6), regularization="l2"), FOLDER_10, PT_FILE_10)
    # # 25 tracks, compact neural net
    # run_rec_err_model(Autoencoder(input_dim=25, code_dim=3, architecture=(8, 5)), FOLDER_25, PT_FILE_25)
    # 50 tracks, compact neural net
    regularizations = [None, "L1", "L2"]
    architectures = [(8, 5)]
    for reg in regularizations:
        for arch in architectures:
            print("*****")
            print(f"Regularization: {reg}, Architecture: {arch}")
            print("*****")
            run_rec_err_model(Autoencoder(input_dim=50, code_dim=3, architecture=arch, regularization=reg), FOLDER_50, PT_FILE_50)
            print()
    # quick_model_diagnostic(Autoencoder(input_dim=50, code_dim=3, architecture=(8, 5), regularization='l2'), FOLDER_50 + "/" + PT_FILE_50)
    # run_density_eff_test(Autoencoder(input_dim=50, code_dim=3, architecture=(8, 5)), FOLDER_50 + "/" + PT_FILE_50)
