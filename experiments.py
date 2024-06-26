"""
Main file for running models and experiments
"""

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
PT_FILE_50 = "50_track_pt_batch_"

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
    ttbar_train, ttbar_x, ttbar_y = load_train_test(folder, file, TRAINING_BATCH_RANGE, TESTING_BATCH_RANGE)
    ttbar_pt2, ttbar_y = load_data(PT2_FOLDER, SUM_PT_FILE, TESTING_BATCH_RANGE)

    last = 1
    test_thresholds = [last + i * 0.2 for i in range(0, 500)]

    last = 5
    pt_thresholds = [last + i for i in range(200)]

    model = autoencoder
    model.train_model(ttbar_train, epochs=10, plot_valid_loss=True)

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
    print(f"{'TTBAR':10} {ttbar_encoder_recall:<10.4f} {ttbar_base_recall:<10.4f}")

    # Separate pu, vbf_hs, ttbar_hs data for roc curves
    # Do this for errors first
    ttbar_mask = ttbar_y == 1
    pu_err = rec_err[~ttbar_mask]
    ttbar_hs_err = rec_err[ttbar_mask]
    # Now, doing this for sum-pt2 values
    pu_pt2 = ttbar_pt2[~ttbar_mask]
    ttbar_hs_pt2 = ttbar_pt2[ttbar_mask]

    # Plot log errors
    # Random sample indices for pu vertices (we don't want to plot all of them, or else it'll be too cluttered)
    rand_pu_idxs = np.random.choice(len(pu_err), int(len(pu_err) * 0.05), replace=False)
    plot_log_reco_err(pu_err, ttbar_hs_err)
    # plot err vs pt2
    plot_err_vs_pt2(err_lst=[pu_err[rand_pu_idxs], ttbar_hs_err],
                    pt_lst=[pu_pt2[rand_pu_idxs], ttbar_hs_pt2], labels=['PU', 'TTBAR HS', 'VBF HS'])

    # # Collect Mistakes
    # collect_mistakes(vbf_pt2, vbf_encoder_yhat, vbf_y, "vbf_correction_list.csv")
    # collect_mistakes(ttbar_pt2, ttbar_encoder_yhat, ttbar_y, "ttbar_correction_list.csv")
    #
    # Get tp / fp rates for model
    ttbar_tp, ttbar_fp = get_fp_tp(test_thresholds, pu_err, ttbar_hs_err)
    # Get tp / fp rates for baseline
    ttbar_baseline_tp, ttbar_baseline_fp = get_fp_tp(pt_thresholds, pu_pt2, ttbar_hs_pt2)

    # headers = ['tpr', 'fpr']
    # np.savetxt("fpr_tpr/tpr_fpr_ttbar_decoder.csv", np.vstack((headers, np.column_stack((ttbar_tp, ttbar_fp)))),
    #            fmt='%s', delimiter=',')
    # np.savetxt("fpr_tpr/tpr_fpr_ttbar_sum-pt2.csv",
    #            np.vstack((headers, np.column_stack((ttbar_baseline_tp, ttbar_baseline_fp)))), fmt='%s', delimiter=',')

    # Plot the ROC curves
    plot_roc_curve(tp_lst=[ttbar_baseline_tp, ttbar_tp],
                   fp_lst=[ttbar_baseline_fp, ttbar_fp],
                   labels=["Sum pt2", "Decoder"], title="TTBAR ROC Curves")


if __name__ == "__main__":
    # # 10 tracks, compact neural net
    # run_rec_err_model(Autoencoder(input_dim=10, code_dim=3, architecture=(8, 6)), FOLDER_10, PT_FILE_10)
    # 25 tracks, compact neural net
    run_rec_err_model(Autoencoder(input_dim=25, code_dim=3, architecture=(8, 5)), FOLDER_25, PT_FILE_25)
    # 50 tracks, compact neural net
    run_rec_err_model(Autoencoder(input_dim=50, code_dim=3, architecture=(8, 5)), FOLDER_50, PT_FILE_50)