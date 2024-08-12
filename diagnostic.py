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

TRAINING_BATCH_RANGE = (0, 1)
TESTING_BATCH_RANGE = (8, 15)
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


def gather_encoder_mistakes(model, pt2, x_data, y_data, reco_zs, hs_truth_zs, start_event_num, outfile_prefix="", via_dist=False):
    rec_vecs = model.reconstructions(x_data)
    rec_errs = np.sum((rec_vecs - x_data) ** 2, axis=1)
    enc_yhat = get_classification(rec_errs, y_data)
    pt2_yhat = get_classification(pt2, y_data)
    enc_selected_zs = reco_zs[enc_yhat.astype(bool)]
    pt2_selected_zs = reco_zs[pt2_yhat.astype(bool)]
    hs_reco_zs = reco_zs[y_data.astype(bool)]

    mistake_x = []
    mistake_rec_x = []  # reconstruction of mistakenly chosen vertex
    correct_x = []
    correct_rec_x = []  # reconstruction of correct vertex
    event_nums = []
    vertex_nums = []
    other_data = []  # stores: mistake rec err, correct rec err, selected z, correct reco z, truth reco z, vertex num, event num

    e_y, e_enc_yhat, e_pt2_yhat, e_x, e_rec_vecs, e_rec_errs, e_pt2 = event_partition(y_data, enc_yhat, pt2_yhat, x_data, rec_vecs, rec_errs, pt2)
    N_events = len(hs_truth_zs)
    for i in range(N_events):
        vertex_num = np.where(e_enc_yhat[i] == 1)[0][0]
        cur_x = e_x[i]
        cur_enc_z = enc_selected_zs[i]
        cur_hs_z = hs_truth_zs[i]
        # check if each algorithm has selected successfully
        enc_unsuccessful = abs(cur_enc_z - cur_hs_z) > 1 if via_dist else vertex_num != 0
        pt2_successful = abs(pt2_selected_zs[i] - cur_hs_z) > 1 if via_dist else e_pt2_yhat[i][0] == 1.0
        if enc_unsuccessful and pt2_successful:
            # Then we consider selection unsuccessful. Record data on wrongly chosen vertex and correct vertex
            mistake_x.append(cur_x[vertex_num])
            mistake_rec_x.append(e_rec_vecs[i][vertex_num])
            event_nums.append(i + start_event_num)
            vertex_nums.append(vertex_num)
            correct_x.append(cur_x[0])
            correct_rec_x.append(e_rec_vecs[i][0])
            print(f"{e_rec_errs[i][vertex_num]:<10.4f}{e_rec_errs[i][0]:<10.4f}{cur_enc_z:<10.4f}{hs_reco_zs[i]:<10.4f}{cur_hs_z:<10.4f}")
            other_data.append([e_pt2[i][vertex_num], e_rec_errs[i][vertex_num], e_pt2[i][0], e_rec_errs[i][0], cur_enc_z, hs_reco_zs[i], hs_truth_zs[i], vertex_num, i+start_event_num])

    mistake_data = np.column_stack((mistake_x, vertex_nums, event_nums))
    correct_data = np.column_stack((correct_x, event_nums))
    mistake_rec = np.column_stack((mistake_rec_x, vertex_nums, event_nums))
    correct_rec = np.column_stack((correct_rec_x, event_nums))

    np.savetxt(outfile_prefix + "_mistakes.csv", mistake_data, delimiter=",", fmt="%.5f")
    np.savetxt(outfile_prefix + "_corrections.csv", correct_data, delimiter=",", fmt="%.5f")
    np.savetxt(outfile_prefix + "_mistakes_rec.csv", mistake_rec, delimiter=",", fmt="%.5f")
    np.savetxt(outfile_prefix + "_corrections_rec.csv", correct_rec, delimiter=",", fmt="%.5f")

    other_data = np.array(other_data)
    headers = "mistake pt2, rec err, coorect pt2, rec err, z,reco_z,truth_z,vert_#,event_#"
    np.savetxt(outfile_prefix + "_macro.csv", other_data, delimiter=",", header=headers, fmt="%.4f")


def gather_mistakes(algo_scores, x_data, y_data, reco_zs, hs_truth_zs, start_event_num, outfile_prefix="", via_dist=False):
    """
    For sake of memory efficiency, does not store headers.
    Note that second last column in mistakes is vertex number
    last column in mistakes is event number of mistake
    :param algo_scores:
    :param x_data:
    :param y_data:
    :param reco_zs:
    :param hs_truth_zs:
    :param start_event_num:
    :param outfile_prefix:
    :return:
    """
    yhat = get_classification(algo_scores, y_data)
    selected_zs = reco_zs[yhat.astype(bool)]
    hs_reco_zs = reco_zs[y_data.astype(bool)]

    mistake_x = []
    correct_x = []
    event_nums = []
    vertex_nums = []

    event_y, event_yhat, event_x = event_partition(y_data, yhat, x_data)

    z_coord_data = []
    N_events = len(hs_truth_zs)
    for i in range(N_events):
        cur_yhat = event_yhat[i]
        vertex_num = np.where(cur_yhat == 1)[0][0]
        cur_x = event_x[i]
        cur_z = selected_zs[i]
        cur_hs_z = hs_truth_zs[i]
        unsuccessful = abs(cur_z - cur_hs_z) > 1 if via_dist else vertex_num != 0
        if unsuccessful:
            # Then we consider selection unsuccessful. Record data on wrongly chosen vertex and correct vertex
            mistake_x.append(cur_x[vertex_num])
            event_nums.append(i + start_event_num)
            vertex_nums.append(vertex_num)
            correct_x.append(cur_x[0])
            print(f"{cur_z:<10.4f}{hs_reco_zs[i]:<10.4f}{cur_hs_z:<10.4f}")
            z_coord_data.append([cur_z, hs_reco_zs[i], hs_truth_zs[i], vertex_num, i+start_event_num])

    if outfile_prefix != "":
        mistake_data = np.column_stack((mistake_x, vertex_nums, event_nums))
        correct_data = np.column_stack((correct_x, event_nums))

        np.savetxt(outfile_prefix + "_mistakes.csv", mistake_data, delimiter=",", fmt="%.5f")
        np.savetxt(outfile_prefix + "_corrections.csv", correct_data, delimiter=",", fmt="%.5f")

    z_coord_data = np.array(z_coord_data)
    z_coord_headers = "selected_z,reco_z,truth_z,vert_#,event_#"
    np.savetxt(outfile_prefix + "_z.csv", z_coord_data, delimiter=",", header=z_coord_headers, fmt="%.4f")

    return np.array(mistake_x), np.array(correct_x)


def run_diagnostic_harness(autoencoder: Autoencoder):
    ttbar_train, ttbar_x, ttbar_y = load_train_test(FEAT_FOLDER + "/" + TTBAR_PT_FILE, TRAINING_BATCH_RANGE,
                                                    TESTING_BATCH_RANGE)

    # Load data for pt2
    ttbar_pt2, _trash = load_data(PT2_FOLDER + "/" + TTBAR_SUM_PT2, TESTING_BATCH_RANGE)

    # Load z-coordinate data
    ttbar_reco_z, _trash = load_data(FEAT_FOLDER + "/" + TTBAR_Z_FILE, TESTING_BATCH_RANGE)
    ttbar_hs_truth_z = load_truth_hs_z(file_paths.ROOT_PATH, TESTING_BATCH_RANGE[0] * BATCH_SIZE,
                                       TESTING_BATCH_RANGE[1] * BATCH_SIZE)

    model = autoencoder
    # model.train_model(ttbar_train, epochs=30, plot_valid_loss=True)
    model.load_weights("models/pt_massive_model.weights.h5")

    make_plots(model, ttbar_x, ttbar_y, ttbar_pt2, ttbar_reco_z, ttbar_hs_truth_z)

    # gather_encoder_mistakes(model, ttbar_pt2, ttbar_x, ttbar_y, ttbar_reco_z, ttbar_hs_truth_z, 4000, "mistakes/ttbar_pt_isHS", via_dist=False)

    # mistakes_x, correct_x = gather_mistakes(ttbar_pt2, ttbar_x, ttbar_y, ttbar_reco_z, ttbar_hs_truth_z, 4000, outfile_prefix="mistakes/ttbar_pt_isHS")
    #
    # # now, let's make some plots to understand the mistakes and corrections
    # wrong_vertex_pts = mistakes_x.flatten()
    # wrong_vertex_pts = wrong_vertex_pts[wrong_vertex_pts != 0]
    # correct_vertex_pts = correct_x.flatten()
    # correct_vertex_pts = correct_vertex_pts[correct_vertex_pts != 0]
    # # First, plot the pt distributions
    # plot_two_axis_hist(np.log(wrong_vertex_pts), np.log(correct_vertex_pts), 20, title="HS vs. Wrongly Chosen PU log track pts")
    #
    # bins = [i*25 for i in range(21)]
    # bins.append(3000)
    # wrong_vertex_sum_pt2 = np.sum(mistakes_x ** 2, axis=1)
    # correct_vertex_sum_pt2 = np.sum(correct_x ** 2, axis=1)
    # diff_pt2 = wrong_vertex_sum_pt2 - correct_vertex_sum_pt2
    # # Plot the difference in correct vertex sum pt2
    # plot_histogram(diff_pt2, bins, title="Difference in sum-pt2")
    # # Plot distribution of the sum pt2 of wrongly chosen vertices
    # plot_histogram(wrong_vertex_sum_pt2, bins, title="Sum pt2 distribution of wrongly chosen PU vertices")
    #
    # wrong_vertex_n_tracks = np.count_nonzero(mistakes_x, axis=1)
    # correct_vertex_n_tracks = np.count_nonzero(correct_x, axis=1)
    # # Plot distributions of the number of tracks
    # plot_two_axis_hist(wrong_vertex_n_tracks, correct_vertex_n_tracks, 20)


def quick_test(autoencoder: Autoencoder):
    model = autoencoder
    x_test = np.array([[25.55088,2.12126,1.73985,1.67074,1.44958,1.40417,1.12832,1.08985,0.96332,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000]])

    model.load_weights("models/pt_model.h5")
    print(model.reconstructions(x_test))


if __name__ == "__main__":
    run_diagnostic_harness(Autoencoder(input_dim=50, code_dim=50, architecture=(50,), regularization=None))
    # quick_test(Autoencoder(input_dim=50, code_dim=3, architecture=(17,), regularization="L2"))
