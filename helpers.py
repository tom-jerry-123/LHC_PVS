"""
Miscellaneous helper functions for testing and analyzing models
for getting data classifications
for getting tp / fp rates
"""

import numpy as np


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