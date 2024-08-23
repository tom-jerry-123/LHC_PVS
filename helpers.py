"""
Miscellaneous helper functions for testing and analyzing models
for getting data classifications
for getting tp / fp rates
"""

import numpy as np
from vertex_density_calc import density_from_z_coord
from plotting import plot_histogram


def get_classification(vals, y_data):
    """
    :param vals: single float value from model for each data point
    :param y_data: data for correct label values (we need this to determine events)
    :return: classification results
    """
    if len(vals) != len(y_data):
        raise ValueError(f"Length of input values ({len(vals)}) not equal to length of labels ({len(y_data)})!")
    split_idxs = np.where(y_data == 1)[0]
    # We discard the first element of the split as it is the empty array
    event_vals = np.split(vals, split_idxs)[1:]
    event_labels = np.split(y_data, split_idxs)[1:]
    predictions = []
    N_events = len(event_vals)
    for i in range(N_events):
        cur_vals = event_vals[i]
        cur_predict = np.zeros(len(cur_vals))
        hs_idx = np.argmax(cur_vals)
        cur_predict[hs_idx] = 1
        predictions.append(cur_predict)
    prediction_arr = np.concatenate(predictions)
    return prediction_arr


def get_fp_tp(test_thresholds, pu_vals, hs_vals):
    pu_errors = np.array(pu_vals)
    hs_errors = np.array(hs_vals)
    num_hs = len(hs_errors)
    num_pu = len(pu_errors)
    tp_rates = []
    fp_rates = []

    for val in test_thresholds:
        tp_rate = np.sum(hs_errors >= val) / num_hs
        fp_rate = np.sum(pu_errors >= val) / num_pu
        if tp_rate < 0.8:
            break
        tp_rates.append(tp_rate)
        fp_rates.append(fp_rate)

    return np.array(tp_rates), np.array(fp_rates)


def get_efficiencies_vs_density(algo_scores, y_data, reco_zs, hs_truth_zs, num_bins=10, interpolate=True,
                                plot_hist=True, algo_name=""):
    """
    Returns efficiencies and bin midpoints calculated (in this order)
    :param algo_scores:
    :param y_data:
    :param reco_zs:
    :param hs_truth_zs:
    :param plot_hist:
    :param algo_name:
    :return:
    """
    # Get classifications of sum-pt2
    yhat = get_classification(algo_scores, y_data)

    # Get histogram of densities. First, densities of selected vertices
    yhat_zs = reco_zs[yhat.astype(bool)]
    densities = density_from_z_coord(yhat_zs)  # densities from selected vertex

    # calculate bins
    bins = num_bins
    if interpolate:
        sorted_data = np.sort(densities)
        # Use np.interp to calculate bin edges such that each bin contains an equal number of data points
        bins = np.interp(np.linspace(0, len(sorted_data), num_bins + 1),
                         np.arange(len(sorted_data)),
                         sorted_data)

    # now, calculate the histogram for selected vertices
    yhat_freq, bin_edges = np.histogram(densities, bins=bins)

    # Now, densities of correctly selected vertices, i.e. true positives
    tp_zs = yhat_zs[abs(yhat_zs - hs_truth_zs) < 1]
    tp_densities = density_from_z_coord(tp_zs)
    tp_freq, bin_edges = np.histogram(tp_densities, bins=bin_edges)

    if plot_hist:
        plot_histogram(densities, bins=bin_edges, title="Density Histograms for Selected Vertices, " + algo_name,
                       x_label="Density")
        plot_histogram(tp_densities, bins=bin_edges, title="Density Histogram for Successful Selection, " + algo_name, x_label="Density")

    # Compute efficiencies and midpoints, then return
    efficiencies = tp_freq / yhat_freq
    midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2
    eff_std = np.sqrt(efficiencies * (1 - efficiencies) / yhat_freq)

    return efficiencies, midpoints, eff_std


def event_partition(y_data, *args):
    split_idxs = np.where(y_data == 1)[0]
    # Splitting will generate empty subarray at start since first split index at 0; remove that empty arr
    partitioned_y = np.split(y_data, split_idxs)[1:]
    return_data_lst = [partitioned_y]
    for i, data_arr in enumerate(args):
        return_data_lst.append(np.split(data_arr, split_idxs)[1:])

    return return_data_lst


def shuffle_data(x_data, *args):
    """
    Shuffles the data passed here. accepts additional args as necessary.
    Assumes all input arrays have same length
    :param x_data: numpy array of data
    :param args:
    :return:
    """
    permute_idxs = np.random.choice(len(x_data), len(x_data), replace=False)
    x_data = x_data[permute_idxs]
    ret_data_lst = [x_data]
    for i, data_arr in enumerate(args):
        ret_data_lst.append(data_arr[permute_idxs])
    return ret_data_lst


