"""
Reads data from root file and stores it in csv

Notes
noticed artifact in data: last vertex has zero pt, same coordinates as HS vertex
removing last vertex in data from data processing
"""
import file_paths
import numpy as np
import uproot
import csv
from data_loading import load_data, load_csv
from helpers import event_partition


"""
Functions for computing track feature data (and loading them, if stored)
Functions include
compute_track_pt
load_pt: for loading list of track pts from stored csv file
"""


def compute_track_pt(tree_file, out_path, event_range):
    with uproot.open(tree_file) as file:
        tree = file['EventTree;1']
        qOverP_array = tree['track_qOverP'].array()
        theta_array = tree['track_theta'].array()

    track_pt_data = []
    if event_range[0] < 0 or event_range[0] >= event_range[1] or event_range[1] > len(theta_array):
        raise ValueError("Event range is invalid!")
    print(f"Starting track pt calculations for events {event_range[0]} to {event_range[1]-1}")
    for i in range(event_range[0], event_range[1]):
        thetas = np.array(theta_array[i])
        qOverP_vals = np.array(qOverP_array[i])
        pts = np.sin(thetas) / np.abs(qOverP_vals) / 1000
        track_pt_data.append(pts)

    print(f"Completed pt calculations for {event_range[1] - event_range[0]} events.")

    with open(out_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(track_pt_data)



def load_pt(pt_path):
    track_pts = []
    with open(pt_path, mode='r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            track_pts.append([float(entry) for entry in row])

    # track_deltaRs = []
    # with open(r_path, mode='r') as file:
    #     csv_reader = csv.reader(file)
    #     for row in csv_reader:
    #         track_deltaRs.append([float(entry) for entry in row])

    print("Done loading csv files")
    return track_pts


def compute_track_eta(tree, event_range):
    theta_array = tree['track_theta'].array()
    eta_lst = []
    for i in range(event_range[0], event_range[1]):
        thetas = np.array(theta_array[i])
        etas = - np.log(np.tan(thetas/2))
        eta_lst.append(etas)
    print(f"Completed Eta Computations from event {event_range[0]} to event {event_range[1]}")
    return eta_lst


def compute_delta_R(tree, event_range):
    # First, compute track etas as we need them to find min delta r
    eta_lst = compute_track_eta(tree, event_range)
    # Collect other necessary data from the tree
    jet_pt_array = tree['jet_pt'].array()
    jet_phi_array = tree['jet_phi'].array()
    jet_eta_array = tree['jet_eta'].array()
    track_phi_array = tree['track_phi'].array()

    delta_R_data = []
    for i in range(event_range[0], event_range[1]):
        mask = jet_pt_array[i] > 30
        jet_phis = np.array(jet_phi_array[i][mask])
        jet_etas = np.array(jet_eta_array[i][mask])
        track_phis = np.array(track_phi_array[i])
        track_etas = np.array(eta_lst[i - event_range[0]])
        # Compute min r values
        Dphi_matrix = track_phis[np.newaxis, :] - jet_phis[:, np.newaxis]
        Dphi_matrix[Dphi_matrix > np.pi] -= 2*np.pi
        Dphi_matrix[Dphi_matrix < -np.pi] += 2*np.pi
        R_matrix = Dphi_matrix ** 2 + (track_etas[np.newaxis, :] - jet_etas[:, np.newaxis])**2
        R_matrix = R_matrix ** 0.5
        R_matrix = np.vstack((R_matrix, np.full((R_matrix.shape[1],), 5)))
        delta_Rs = np.min(R_matrix, axis=0)
        delta_R_data.append(delta_Rs)

    print(f"Completed delta-R computations from event {event_range[0]} to event {event_range[1]}")

    return delta_R_data


"""
For storing computations in csv files
Functions include
read_pt_to_csv
read_z_coord_to_csv
read_sum_pt2_to_csv
"""


def read_pt_to_csv(tree, track_pts, outfile_path, N_TRACKS=50, batch_range=None):
    """
    Reads the pts to a csv-file. Top N_TRACKS pt
    :param track_pts: list of computed track pts
    :param outfile_path: Name of output file
    :param N_TRACKS: number of pt tracks to store
    :param batch_range: two-tuple specifying event indices of start and end of range to process
    :return:
    """

    idx_array = tree['recovertex_tracks_idx'].array()
    weight_array = tree['recovertex_tracks_weight'].array()
    isHS_array = tree['recovertex_isHS'].array()
    vertex_z_array = tree['recovertex_z'].array()

    pt_data = []
    z_coords = []
    labels = []
    N_events = len(idx_array)
    if batch_range is None:
        batch_range = (0, N_events)
    if batch_range[1] > N_events or batch_range[0] < 0:
        raise ValueError("Invalid range request for processing batch!")
    print(f"Processing Batch from event {batch_range[0]} to event {batch_range[1]}")
    for i in range(batch_range[0], batch_range[1]):
        # Ignoring final vertex for labels and z-coordinate
        labels.extend(isHS_array[i][:-1])
        # Collect the data on the current event
        N_vertices = len(idx_array[i])
        event_pts = np.array(track_pts[i])
        event_weight_arrs = weight_array[i]
        event_idx_arrs = idx_array[i]
        # Note: we know the last vertex is an artifact so we ignore it
        for j in range(N_vertices-1):
            # Get proper tracks (i.e. weight > 0.75)
            vertex_weights = event_weight_arrs[j]
            vertex_idxs = event_idx_arrs[j]
            weight_mask = vertex_weights >= 0.75
            vertex_idxs = vertex_idxs[weight_mask]
            vertex_pts = event_pts[vertex_idxs]
            pt_mask = vertex_pts <= 50
            # Sort tracks by pt, in descending order
            vertex_pts = vertex_pts[pt_mask]
            vertex_pts = np.sort(vertex_pts)[::-1]
            # Pad with zeros
            if len(vertex_pts) >= N_TRACKS:
                vertex_pts = vertex_pts[:N_TRACKS]
            else:
                vertex_pts = np.pad(vertex_pts, (0, N_TRACKS - len(vertex_pts)), mode='constant')
            pt_data.append(vertex_pts)
        if i % 100 == 99:
            print(f"Done event {i}.")

    # Create headers
    headers = ["pt_" + str(i) for i in range(N_TRACKS)]
    headers.append("y")
    feat_data = np.column_stack((np.array(pt_data), labels))
    feat_data = np.vstack((headers, feat_data))

    np.savetxt(outfile_path, feat_data, delimiter=',', fmt='%s')
    print(f"Successfully saved data batch to '{outfile_path}'")


def read_z_coord_to_csv(tree, outfile_path, batch_range=None):
    isHS_array = tree['recovertex_isHS'].array()
    vertex_z_array = tree['recovertex_z'].array()

    z_coords = []
    labels = []
    N_events = len(vertex_z_array)
    if batch_range is None:
        batch_range = (0, N_events)
    if batch_range[1] > N_events or batch_range[0] < 0:
        raise ValueError("Invalid range request for processing batch!")
    print(f"Processing Batch from event {batch_range[0]} to event {batch_range[1]}")
    for i in range(batch_range[0], batch_range[1]):
        # Ignoring final vertex for labels and z-coordinate
        labels.extend(isHS_array[i][:-1])
        z_coords.extend(vertex_z_array[i][:-1])

    z_coord_data = np.column_stack((z_coords, labels))
    z_coord_data = np.vstack((["z_coord", "y"], z_coord_data))

    np.savetxt(outfile_path, z_coord_data, delimiter=',', fmt='%s')
    print(f"Successfully saved data batch to '{outfile_path}'")


def read_sum_pt2_to_csv(tree, track_pts, outfile_path, batch_range=None):
    """
    Reads sum of pt2 for each vertex to a csv file, along with y_label.
    :param track_pts:
    :return:
    """
    idx_array = tree['recovertex_tracks_idx'].array()
    weight_array = tree['recovertex_tracks_weight'].array()
    isHS_array = tree['recovertex_isHS'].array()

    vertex_data = []
    labels = []
    N_events = len(idx_array)
    if batch_range is None:
        batch_range = (0, N_events)
    if batch_range[1] > N_events or batch_range[0] < 0:
        raise ValueError("Invalid range request for processing batch!")
    print(f"Processing Batch from event {batch_range[0]} to event {batch_range[1]}")
    for i in range(batch_range[0], batch_range[1]):
        labels.extend(isHS_array[i][:-1])
        N_vertices = len(idx_array[i])
        event_pts = np.array(track_pts[i])
        event_weight_arrs = weight_array[i]
        event_idx_arrs = idx_array[i]
        # We ignore the last vertex as it is irrelevant
        for j in range(N_vertices-1):
            # Get proper tracks (i.e. weight > 0.75)
            vertex_weights = event_weight_arrs[j]
            vertex_idxs = event_idx_arrs[j]
            weight_mask = vertex_weights >= 0.75
            vertex_idxs = vertex_idxs[weight_mask]
            vertex_pts = event_pts[vertex_idxs]
            # get tracks with proper pt
            vertex_pts = vertex_pts[vertex_pts <= 50]
            sum_pt2 = np.sum(np.multiply(vertex_pts, vertex_pts))
            vertex_data.append(sum_pt2)
        if i % 100 == 0:
            print(f"Done event {i}.")

    headers = ["sum-pt2", 'y']
    final_data = np.column_stack((np.array(vertex_data), labels))
    final_data = np.vstack((headers, final_data))

    np.savetxt(outfile_path, final_data, delimiter=',', fmt='%s')
    print(f"Successfully saved data batch to '{outfile_path}'")


def read_features_to_csv(tree, out_path, n_tracks, event_range):
    """
    Reads the pts and Rs to a csv-file. Top N_tracks pt and their corresponding delta-R
    :param out_path: Name of output file
    :param batch_range: two-tuple specifying event indices of start and end of range to process
    :return:
    """
    idx_array = tree['recovertex_tracks_idx'].array()
    weight_array = tree['recovertex_tracks_weight'].array()
    isHS_array = tree['recovertex_isHS'].array()

    track_Drs = compute_delta_R(tree, event_range)
    track_pts = load_pt("other_data_files/track_pt_full_ttbar.csv")
    vertex_data = []
    labels = []
    N_events = len(idx_array)
    if event_range is None:
        event_range = (0, N_events)
    if event_range[1] > N_events or event_range[0] < 0:
        raise ValueError("Invalid range request for processing event!")
    print(f"Processing event from event {event_range[0]} to event {event_range[1]}")
    for i in range(event_range[0], event_range[1]):
        labels.extend(isHS_array[i][:-1])
        N_vertices = len(idx_array[i])
        event_pts = np.array(track_pts[i])
        event_Drs = np.array(track_Drs[i - event_range[0]])
        event_weight_arrs = weight_array[i]
        event_idx_arrs = idx_array[i]
        for j in range(N_vertices-1):
            # Get proper tracks (i.e. weight > 0.75)
            vertex_weights = event_weight_arrs[j]
            vertex_idxs = event_idx_arrs[j]
            weight_mask = vertex_weights >= 0.75
            vertex_idxs = vertex_idxs[weight_mask]
            vertex_pts = event_pts[vertex_idxs]
            vertex_dRs = event_Drs[vertex_idxs]
            # get tracks with proper pt
            pt_mask = vertex_pts <= 50
            vertex_pt_R = np.column_stack((vertex_pts[pt_mask], vertex_dRs[pt_mask]))
            # sort tracks by pt in descending order. Each min Dr immediately follows associated pt
            sorted_indices = np.argsort(vertex_pt_R[:, 0])[::-1]
            vertex_pt_R = vertex_pt_R[sorted_indices]
            vertex_pt_R = vertex_pt_R.flatten()
            if len(vertex_pt_R) >= n_tracks*2:
                vertex_pt_R = vertex_pt_R[:n_tracks*2]
            else:
                vertex_pt_R = np.pad(vertex_pt_R, (0, n_tracks*2 - len(vertex_pt_R)), mode='constant')
            vertex_data.append(vertex_pt_R)
        if i % 100 == 0:
            print(f"Done event {i}.")

    headers = [("pt_" if i % 2 == 0 else "Dr_") + str(i // 2) for i in range(n_tracks*2)]
    headers.append("y")
    final_data = np.column_stack((np.array(vertex_data), labels))
    final_data = np.vstack((headers, final_data))

    np.savetxt(out_path, final_data, delimiter=',', fmt='%s')
    print(f"Successfully saved data batch to '{out_path}'")


def make_supervised_sample():
    """
    Makes a sample for supervised learning by the autoencoder
    Uses first 8 batches from ttbar pt data
    :return:
    """
    # Load data
    x_data, y_data = load_data("50_track_batches/50_track_ttbar_pt_", (0, 8))

    # Now, randomly select 16000 pu vertices and 16000 hs vertices (hs vertices upsampled)
    hs_mask = y_data == 1
    hs_data = x_data[hs_mask]
    pu_data = x_data[~hs_mask]
    pu_idxs = np.random.choice(pu_data.shape[0], 16000, replace=False)
    selected_pu = pu_data[pu_idxs, :]
    hs_idxs = np.random.choice(hs_data.shape[0], selected_pu.shape[0], replace=True)
    selected_hs = hs_data[hs_idxs, :]

    # Now, concatenate and shuffle data
    selected_hs = np.column_stack((selected_hs, np.ones(selected_hs.shape[0], dtype='float32')))
    selected_pu = np.column_stack((selected_pu, np.zeros(selected_pu.shape[0], dtype='float32')))
    final_data = np.vstack((selected_hs, selected_pu))
    permute_idxs = np.random.choice(final_data.shape[0], final_data.shape[0], replace=False)
    final_data = final_data[permute_idxs]

    # Write to csv
    np.savetxt("supervised_ttbar_train.csv", final_data, delimiter=",", fmt="%.4f",
               header="Top 50 pts of each vertex (zero-padding). Last column is label.")


def compute_event_masks():
    with uproot.open(file_paths.ROOT_PATH) as file:
        ttbar_tree = file["EventTree;1"]
        print("Successfully loaded TTBAR tree")
    with uproot.open(file_paths.VBF_ROOT_PATH) as file:
        vbf_tree = file["EventTree;1"]
        print("Successfully loaded VBF tree")
    ttbar_jet_pts = ttbar_tree["jet_pt"].array()
    vbf_jet_pts = vbf_tree["jet_pt"].array()

    N_EVENTS = 7500
    ttbar_exclude_idxs = []
    vbf_exclude_idxs = []
    # Find all the events to exclude for ttbar and vbf
    for i in range(N_EVENTS):
        if np.sum(ttbar_jet_pts[i] >= 30.0) < 2:
            ttbar_exclude_idxs.append(i)
        if np.sum(vbf_jet_pts[i] >= 30.0) < 2:
            vbf_exclude_idxs.append(i)
    ttbar_exclude_idxs = np.array(ttbar_exclude_idxs)
    vbf_exclude_idxs = np.array(vbf_exclude_idxs)
    # Create boolean (zero-one) mask indicating valid events
    vbf_include_mask = np.ones(N_EVENTS)
    ttbar_include_mask = np.ones(N_EVENTS)
    vbf_include_mask[vbf_exclude_idxs] = 0
    ttbar_include_mask[ttbar_exclude_idxs] = 0

    # save the masks
    np.savetxt("other_data_files/event_inclusion_mask.csv", np.vstack((ttbar_include_mask, vbf_include_mask)), delimiter=',',
               header="First Row is TTBAR mask, second is VBF mask", fmt="%d")


def filter_events(outpath):
    """
    Using this to create new data files from older ones
    :return:
    """
    # Load necessary data
    ttbar_data, y_data = load_data("50_track_batches_old/50_track_ttbar_pt_dR_", (0, 15))
    reco_z_data, _trash = load_data("50_track_batches_old/50_track_ttbar_z_", (0, 15))
    event_masks, _trash = load_csv("other_data_files/event_inclusion_mask.csv", has_headers=True, has_y=False)
    # Type cast, and, if fewer events are loaded, only keep relevant part of event mask
    event_masks = event_masks.astype(bool)
    ttbar_mask = event_masks[0, :]
    y_data = y_data.astype(int)
    # Combine data. Last column is event index (starting from zero)
    X_data = np.column_stack((ttbar_data, reco_z_data, np.cumsum(y_data) - 1))
    # Partition over events
    part_y_data, part_X_data = event_partition(y_data, X_data)
    # Select the events to keep based on event mask
    kept_y_data = [sub_array for sub_array, mask in zip(part_y_data, ttbar_mask) if mask]
    kept_X_data = [sub_array for sub_array, mask in zip(part_X_data, ttbar_mask) if mask]
    # Concatenate data and write to csv
    final_y_data = np.concatenate(kept_y_data)
    final_X_data = np.concatenate(kept_X_data, axis=0)
    final_data = np.column_stack((final_X_data, final_y_data))
    headers = "First 100 columns pt and min dR alternating (pt0, dr0, pt1, dr1, etc.). 101st column reco z, 102nd event_#, 103rd y"
    np.savetxt(outpath, final_data, delimiter=",", header=headers, fmt="%.4f")



if __name__ == "__main__":
    # with uproot.open(file_paths.ROOT_PATH) as file:
    #     tree = file["EventTree;1"]
    #     print("Successfully loaded tree")
    # # compute_track_pt(VBF_ROOT_PATH, "other_data_files/track_pt_vbf_8000-16000.csv", (8000, 16000))
    # BATCH_SIZE = 500
    # N_TRACKS = 50
    # # track_pts = load_pt("other_data_files/track_pt_vbf.csv")
    # # new_track_pts = load_pt("other_data_files/track_pt_vbf_8000-16000.csv")
    # # track_pts.extend(new_track_pts)
    # # print("Successfully loaded all track pts.")

    # for k in range(0, 15):
    #     event_range = (k*BATCH_SIZE, k*BATCH_SIZE + BATCH_SIZE)
    #     read_features_to_csv(tree, f"50_track_batches/50_track_ttbar_pt_dR_{k}.csv", n_tracks=N_TRACKS, event_range=event_range)
    # compute_event_masks()
    filter_events(outpath="data_batches/ttbar_big_7500e.csv")
