"""
Reads data from root file and stores it in csv

Notes
noticed artifact in data: last vertex has zero pt, same coordinates as HS vertex
removing last vertex in data from data processing
"""

from file_paths import *
import numpy as np
import uproot
import csv


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


def read_pt_to_csv(track_pts, outfile_path, N_TRACKS=50, batch_range=None):
    """
    Reads the pts to a csv-file. Top N_TRACKS pt and their corresponding delta-R
    :param track_pts: list of computed track pts
    :param outfile_path: Name of output file
    :param N_TRACKS: number of pt tracks to store
    :param batch_range: two-tuple specifying event indices of start and end of range to process
    :return:
    """
    with uproot.open(ROOT_PATH) as file:
        tree = file["EventTree;1"]
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
        z_coords.extend(vertex_z_array[i][:-1])
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
    headers.append("z-coord")
    headers.append("y")
    final_data = np.column_stack((np.array(pt_data), z_coords, labels))
    final_data = np.vstack((headers, final_data))

    np.savetxt(outfile_path, final_data, delimiter=',', fmt='%s')
    print(f"Successfully saved data batch to '{outfile_path}'")


if __name__ == "__main__":
    BATCH_SIZE = 500
    N_TRACKS = 50
    track_pts = load_pt("other_data_files/track_pt_full_ttbar.csv")
    for k in range(0, 3):
        batch_range = (k*BATCH_SIZE, k*BATCH_SIZE + BATCH_SIZE)
        read_pt_to_csv(track_pts, f"{N_TRACKS}_track_batches/{N_TRACKS}_track_batch_{k}.csv", N_TRACKS=50, batch_range=batch_range)
