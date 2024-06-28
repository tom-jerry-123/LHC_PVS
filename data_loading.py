"""
Functions for loading data
"""

import numpy as np
import csv


def load_csv(csv_path, has_headers=True, has_y=True):
    with open(csv_path, 'r') as file:
        # Create a CSV reader object
        csv_reader = csv.reader(file)

        if has_headers:
            headers = next(csv_reader)

        x_data = []
        y_data = []
        # Iterate over each row in the CSV file
        for row in csv_reader:
            # Each row is a list where each element represents a column value
            row = [float(value) for value in row]
            if has_y:
                x_data.append(row[:-1])
                y_data.append(row[-1])
            else:
                x_data.append(row)

        return np.array(x_data), np.array(y_data)


def load_data(file_path, batch_range=(0, 8)):
    # Loads all events
    x_data_lst, y_data_lst = [], []
    for i in range(batch_range[0], batch_range[1]):
        cur_x, cur_y = load_csv(file_path + str(i) + '.csv')
        print(f"Done batch {i} from '{file_path}' files")
        x_data_lst.append(cur_x)
        y_data_lst.append(cur_y)
    x_data = np.vstack(x_data_lst)
    y_data = np.concatenate(y_data_lst)
    return x_data, y_data


def load_train_test(file_path, train_range, test_range):
    # Return training data, x_test, y_test
    x_train, y_train = load_data(file_path, train_range)
    training_data = x_train[y_train == 0]
    x_test, y_test = load_data(file_path, test_range)
    return training_data, x_test, y_test
