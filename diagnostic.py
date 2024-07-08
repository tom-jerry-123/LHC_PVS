"""
Used to diagnose model bugs
"""
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
FEAT_FOLDER = "50_track_batches"
TTBAR_PT_FILE = "50_track_ttbar_pt_"
TTBAR_Z_FILE = "50_track_ttbar_z_"
VBF_PT_FILE = "50_track_vbf_pt_"
VBF_Z_FILE = "50_track_vbf_z_"

PT2_FOLDER = "sum_pt2_batches"
TTBAR_SUM_PT2 = "ttbar_sum_pt2_"
VBF_SUM_PT2 = "vbf_sum_pt2_"

TRAINING_BATCH_RANGE = (0, 8)
TESTING_BATCH_RANGE = (8, 12)
BATCH_SIZE = 500


def make_plots(model, ttbar_x, ttbar_y, ttbar_pt2, ttbar_reco_z, ttbar_hs_truth_z):
    ttbar_err = model.reconstruction_error(ttbar_x)

    # Get encodings and errors for hs and pu vertices separately for diagnostic purposes
    y_mask = ttbar_y == 1
    ttbar_hs_x = ttbar_x[y_mask]
    ttbar_pu_x = ttbar_x[~y_mask]
    hs_encodings = model.get_encodings(ttbar_hs_x)
    hs_err = ttbar_err[y_mask]
    hs_pt2 = ttbar_pt2[y_mask]
    chosen_idxs = np.random.choice(len(ttbar_pu_x), (int(0.05 * len(ttbar_pu_x)),), replace=False)
    pu_encodings = model.get_encodings(ttbar_pu_x)[chosen_idxs]
    pu_err = ttbar_err[~y_mask][chosen_idxs]
    pu_pt2 = ttbar_pt2[~y_mask][chosen_idxs]

    # Get efficiencies for TTBAR
    pt2_ttbar_eff, pt2_ttbar_midpts, pt2_ttbar_std = get_efficiencies_vs_density(ttbar_pt2, ttbar_y, ttbar_reco_z,
                                                                                 ttbar_hs_truth_z, num_bins=10,
                                                                                 plot_hist=False, algo_name="Sum-pt2")
    enc_ttbar_eff, enc_ttbar_midpts, enc_ttbar_std = get_efficiencies_vs_density(ttbar_err, ttbar_y, ttbar_reco_z,
                                                                                 ttbar_hs_truth_z, plot_hist=False,
                                                                                 algo_name="Autoencoder")

    # Time to make plots
    line_plot([pt2_ttbar_midpts, enc_ttbar_midpts], [pt2_ttbar_eff, enc_ttbar_eff], [pt2_ttbar_std, enc_ttbar_std],
              ['Sum-pt2', 'Autoencoder'], title="Efficiency vs. Density for TTBAR", xlabel="Vertex Density",
              ylabel="Efficiency")
    # plot_encodings([pu_encodings, hs_encodings], ["PU", "HS"])
    plot_err_vs_pt2([pu_err, hs_err], [pu_pt2, hs_pt2], ["PU", "HS"])


def make_reconstructions(model, ttbar_x, ttbar_y):
    y_mask = ttbar_y == 1
    ttbar_hs = ttbar_x[y_mask]
    ttbar_pu = ttbar_x[~y_mask]
    selected_pu_idxs = np.random.choice(len(ttbar_pu), (1000,), replace=False)
    selected_hs_idxs = np.random.choice(len(ttbar_hs), (1000,), replace=(len(ttbar_hs) < 1000))
    ttbar_hs = ttbar_hs[selected_hs_idxs]
    ttbar_pu = ttbar_pu[selected_pu_idxs]

    hs_reconstructions = model.get_reconstructions(ttbar_hs)
    pu_reconstructions = model.get_reconstructions(ttbar_pu)

    np.savetxt('diagnostic_data_files/ttbar_hs_sample.csv', ttbar_hs, delimiter=',', fmt='%f')
    np.savetxt('diagnostic_data_files/ttbar_pu_sample.csv', ttbar_pu, delimiter=',', fmt='%f')
    np.savetxt('diagnostic_data_files/ttbar_hs_reco.csv', hs_reconstructions, delimiter=',', fmt='%f')
    np.savetxt('diagnostic_data_files/ttbar_pu_reco.csv', pu_reconstructions, delimiter=',', fmt='%f')



def code_dim_test(ttbar_train, ttbar_x, ttbar_y, ttbar_pt2, ttbar_reco_z, ttbar_hs_truth_z):
    code_1d = Autoencoder(input_dim=50, code_dim=1, architecture=(32,), regularization="L2")
    code_2d = Autoencoder(input_dim=50, code_dim=2, architecture=(32,), regularization="L2")
    code_3d = Autoencoder(input_dim=50, code_dim=3, architecture=(32,), regularization="L2")

    code_1d.train_model(ttbar_train, epochs=10, plot_valid_loss=False)
    code_2d.train_model(ttbar_train, epochs=10, plot_valid_loss=False)
    code_3d.train_model(ttbar_train, epochs=10, plot_valid_loss=False)

    one_err = code_1d.reconstruction_error(ttbar_x)
    two_err = code_2d.reconstruction_error(ttbar_x)
    three_err = code_3d.reconstruction_error(ttbar_x)

    # Get efficiencies for TTBAR
    pt2_eff, pt2_midpts, pt2_std = get_efficiencies_vs_density(ttbar_pt2, ttbar_y, ttbar_reco_z,
                                                                                 ttbar_hs_truth_z, num_bins=10,
                                                                                 plot_hist=False, algo_name="Sum-pt2")
    one_eff, one_midpts, one_std = get_efficiencies_vs_density(one_err, ttbar_y, ttbar_reco_z,
                                                                                 ttbar_hs_truth_z, plot_hist=False,
                                                                                 algo_name="1d code")
    two_eff, two_midpts, two_std = get_efficiencies_vs_density(two_err, ttbar_y, ttbar_reco_z,
                                                               ttbar_hs_truth_z, plot_hist=False,
                                                               algo_name="1d code")
    three_eff, three_midpts, three_std = get_efficiencies_vs_density(three_err, ttbar_y, ttbar_reco_z,
                                                               ttbar_hs_truth_z, plot_hist=False,
                                                               algo_name="3d code")

    # Time to make plots
    line_plot([pt2_midpts, one_midpts, two_midpts, three_midpts], [pt2_eff, one_eff, two_eff, three_eff], [pt2_std, one_std, two_std, three_std],
              ['Sum-pt2', '1d code', '2d code', '3d code'], title="Efficiency vs. Density for TTBAR", xlabel="Vertex Density",
              ylabel="Efficiency")


def run_diagnostic_harness(autoencoder: Autoencoder):
    ttbar_train, ttbar_x, ttbar_y = load_train_test(FEAT_FOLDER + "/" + TTBAR_PT_FILE, TRAINING_BATCH_RANGE,
                                                    TESTING_BATCH_RANGE)

    # Load data for pt2
    ttbar_pt2, _trash = load_data(PT2_FOLDER + "/" + TTBAR_SUM_PT2, TESTING_BATCH_RANGE)

    # Load z-coordinate data
    ttbar_reco_z, _trash = load_data(FEAT_FOLDER + "/" + TTBAR_Z_FILE, TESTING_BATCH_RANGE)
    ttbar_hs_truth_z = load_truth_hs_z(file_paths.ROOT_PATH, TESTING_BATCH_RANGE[0] * BATCH_SIZE,
                                       TESTING_BATCH_RANGE[1] * BATCH_SIZE)

    # Use autoencoder to get reconstruction errors for classification
    model = autoencoder

    model.train_model(ttbar_train, epochs=10, plot_valid_loss=False)

    make_reconstructions(model, ttbar_x, ttbar_y)

if __name__ == "__main__":
    run_diagnostic_harness(Autoencoder(input_dim=50, code_dim=3, architecture=(32,), regularization="L2"))
